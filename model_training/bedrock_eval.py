import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv
from supabase import Client, create_client

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION - SINGLE BEDROCK JUDGE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TABLE = "benchmark_results"
DEFAULT_PAGE_SIZE = 500
DEFAULT_MAX_RETRIES = 5
VALID_USE_CASES = {"text-generation", "code-generation", "reasoning"}

@dataclass(frozen=True)
class EvaluatorModel:
    short_id: str
    provider: str
    kind: str
    model_id: str
    fmt: str | None = None

# We use Amazon Nova Micro, the absolute cheapest Bedrock model.
# ($0.035 / 1M input tokens. Extremely low cost, robust enough for json evaluation).
SINGLE_JUDGE = EvaluatorModel(
    short_id="nova-micro (Bedrock)",
    provider="Amazon",
    kind="bedrock",
    model_id="us.amazon.nova-micro-v1:0",
    fmt="nova",
    
)

_BEDROCK_CLIENT = None

def get_bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        _BEDROCK_CLIENT = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return _BEDROCK_CLIENT

def _build_bedrock_body(fmt: str, prompt: str) -> str:
    # Amazon Nova uses this message schema
    return json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 700},
    })

def _extract_bedrock_text(fmt: str, body_json: dict[str, Any]) -> str:
    # Extracting Amazon Nova response
    return (
        body_json.get("output", {})
        .get("message", {})
        .get("content", [{}])[0]
        .get("text", "")
    )


def call_bedrock(model: EvaluatorModel, content: str) -> str:
    bedrock = get_bedrock_client()
    raw = bedrock.invoke_model(
        modelId=model.model_id,
        body=_build_bedrock_body(model.fmt or "nova", content),
    )
    body_json = json.loads(raw["body"].read())
    return _extract_bedrock_text(model.fmt or "nova", body_json)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT RUBRICS
# ─────────────────────────────────────────────────────────────────────────────
BASE_EVALUATOR_INSTRUCTIONS = """You are an expert evaluator for benchmark datasets.

You will be given:
- the use case
- the original user prompt
- one model response

Your job is to score how well the response satisfies the prompt for that use case.

Before scoring, infer the expected answer scope from the prompt:
- concise: short factual or direct-answer prompts
- standard: normal explanatory prompts
- comprehensive: prompts that explicitly ask for depth, comparison, multi-step detail, or production-ready output

Scoring principles:
- Judge the response against what the prompt actually asked for.
- Do not reward extra length unless it clearly improves task fulfillment.
- A short correct answer can be excellent for a concise prompt.
- Irrelevant detail, over-explaining, scope drift, factual errors, broken code, or invalid reasoning must lower the score.
- Score the response on its own merits, not relative to some imagined ideal format.

Score meaning:
- 95-100: excellent, very hard to improve for this prompt
- 85-94: strong answer with minor issues
- 75-84: good answer with meaningful limitations
- 65-74: mixed quality with notable issues
- 50-64: weak or incomplete
- 30-49: poor, significantly flawed
- 10-29: very poor
- 0-9: no useful value

Return ONLY valid JSON in exactly this shape, and absolutely no other text.
{"accuracy_score": <integer 0-100>}
"""

USE_CASE_PROMPTS = {
    "text-generation": """USE CASE: text-generation
Rubric: Correctness/relevance (55%), Scope fit (25%), Completeness (15%), Clarity (5%).""",
    "code-generation": """USE CASE: code-generation
Rubric: Correctness/executability (45%), Requirement coverage (25%), Practical usefulness (20%), Clarity (10%).""",
    "reasoning": """USE CASE: reasoning
Rubric: Logical soundness (40%), Final answer correctness (30%), Completeness (20%), Clarity (10%)."""
}

def build_evaluator_prompt(use_case: str, prompt: str, response: str) -> str:
    normalized_use_case = (use_case or "text-generation").strip().lower()
    rubric = USE_CASE_PROMPTS.get(normalized_use_case, USE_CASE_PROMPTS["text-generation"])
    return (
        f"{BASE_EVALUATOR_INSTRUCTIONS}\n\n{rubric}\n\n"
        f"USER PROMPT:\n{prompt.strip()}\n\nMODEL RESPONSE:\n{response.strip()}\n"
    )

def _build_retry_prompt(base_content: str) -> str:
    return f"{base_content}\n\nFINAL REMINDER:\nReturn ONLY a JSON object: {{\"accuracy_score\": 87}}"


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION AND SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "accuracy_score" in data: return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find JSON object in output")
    data = json.loads(match.group(0))
    return data

def compute_prompt_hash(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    return create_client(url, key)

def fetch_rows(supabase: Client, table: str, limit: int | None, use_case: str | None, row_id: str | None, page_size: int):
    rows = []
    start = 0
    selected = "id,prompt,response,use_case,avg_accuracy_score,prompt_hash"
    while True:
        query = supabase.table(table).select(selected).order("created_at", desc=False)
        if use_case: query = query.eq("use_case", use_case)
        if row_id: query = query.eq("id", row_id)
        
        batch = query.range(start, start + page_size - 1).execute().data or []
        if not batch: break
        rows.extend(batch)
        if limit is not None and len(rows) >= limit: return rows[:limit]
        if len(batch) < page_size or row_id: break
        start += page_size
    return rows

def evaluate_with_model(model: EvaluatorModel, use_case: str, prompt: str, response: str) -> int:
    base_content = build_evaluator_prompt(use_case, prompt, response)
    
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            content = base_content if attempt == 0 else _build_retry_prompt(base_content)
            raw_text = call_bedrock(model, content)
            payload = _extract_json_object(raw_text)
            
            score = payload.get("accuracy_score")
            return max(0, min(100, int(round(float(score)))))
            
        except Exception as exc:
            err_str = str(exc).lower()
            is_rate = any(x in err_str for x in ["429", "throttling", "rate", "quota"])
            if attempt == DEFAULT_MAX_RETRIES - 1:
                raise RuntimeError(f"{model.short_id} failed permanently: {exc}")
            
            delay = (2 ** attempt) if is_rate else 1
            print(f"    [Retry {attempt+1}] Bedrock Throttling... wait {delay}s")
            time.sleep(delay)

def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    use_case = str(row.get("use_case", "text-generation") or "text-generation").strip().lower()
    if use_case not in VALID_USE_CASES: use_case = "text-generation"

    score = evaluate_with_model(SINGLE_JUDGE, use_case, prompt, response)

    payload = {
        # Directly map Nova score to avg_accuracy_score
        "avg_accuracy_score": float(score),
        "score_stdev": 0.0,
        "eval_conflict_flag": False,
        "high_conflict_flag": False,
        "low_confidence": False,
        "confidence_level": 1.0,
        "eval_count": 1,
        "prompt_hash": compute_prompt_hash(prompt),
        "_meta_judge": SINGLE_JUDGE.short_id 
    }

    return payload

def main() -> int:
    # Ensure AWS SDK picks up keys from .env
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    # Bedrock models can throttle, so default 4 max workers is safer
    parser.add_argument("--max-workers", type=int, default=4) 
    args = parser.parse_args()

    supabase = get_supabase_client()
    rows = fetch_rows(supabase, DEFAULT_TABLE, args.limit, None, None, DEFAULT_PAGE_SIZE)
    pending = [r for r in rows if args.force or r.get("avg_accuracy_score") is None]

    if not pending:
        print("No pending rows to update (use --force to recalculate).")
        return 0

    print(f"Fetched rows: {len(rows)} | Pending: {len(pending)}")
    print(f"Mode: Single Judge ({SINGLE_JUDGE.short_id}) -> avg_accuracy_score")

    completed, failures = 0, 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_row = {executor.submit(evaluate_row, r): r for r in pending}
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            row_id = str(row.get("id"))
            completed += 1
            try:
                payload = future.result()
                judge = payload.pop("_meta_judge", "unknown")
                supabase.table(DEFAULT_TABLE).update(payload).eq("id", row_id).execute()
                print(f"[{completed}/{len(pending)}] row={row_id[:8]} score={payload['avg_accuracy_score']} (judge={judge})")
            except Exception as exc:
                failures += 1
                print(f"[{completed}/{len(pending)}] row={row_id[:8]} FAILED: {exc}")

    print(f"\nDone. Failures: {failures}")
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
