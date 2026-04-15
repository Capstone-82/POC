from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv
from google import genai
from supabase import Client, create_client


DEFAULT_PLACEHOLDER_SCORE = 50
DEFAULT_TABLE = "benchmark_results"
DEFAULT_PAGE_SIZE = 500
DEFAULT_MAX_RETRIES = 4
VALID_USE_CASES = {"text-generation", "code-generation", "reasoning"}
DEFAULT_USE_CASE_EVALUATORS = {
    "text-generation": ["llama4-maverick", "mistral-large", "nova-premier"],
    "code-generation": ["llama4-maverick", "mistral-large"],
    "reasoning": ["llama4-maverick", "nova-premier"],
}
_BEDROCK_CLIENT = None


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

Return ONLY valid JSON in exactly this shape:
{"accuracy_score": <integer 0-100>}
"""


USE_CASE_PROMPTS = {
    "text-generation": """USE CASE: text-generation

Rubric:
1. Correctness and relevance (55%): Is the response factually correct and directly responsive?
2. Scope fit (25%): Is it appropriately concise or detailed for the prompt?
3. Completeness (15%): Does it fully answer the task at the right scope?
4. Clarity (5%): Is it easy to understand?

Extra rules:
- For direct factual prompts, brevity is a strength.
- Do not reward decorative filler, template-heavy phrasing, or generic padding.
- Penalize responses that sound polished but do not actually answer the question.
- If the prompt is open-ended creative writing, score for relevance, coherence, and fit to the requested style or constraints.""",
    "code-generation": """USE CASE: code-generation

Rubric:
1. Correctness and executability (45%): Is the code or technical answer correct and likely to work?
2. Requirement coverage (25%): Does it satisfy the requested functionality and constraints?
3. Practical usefulness (20%): Is it actionable, implementation-ready, and not missing critical pieces?
4. Clarity (10%): Is it organized and understandable?

Extra rules:
- Prefer working, grounded solutions over verbose explanation.
- Penalize fake APIs, broken syntax, missing imports when critical, unsafe hallucinations, and non-runnable pseudo-solutions presented as real code.
- If the prompt asks for explanation instead of code, evaluate the technical guidance by correctness and usefulness.""",
    "reasoning": """USE CASE: reasoning

Rubric:
1. Logical soundness (40%): Are the reasoning steps valid and coherent?
2. Final answer correctness (30%): Is the conclusion correct?
3. Completeness of reasoning (20%): Are the necessary steps covered without major gaps?
4. Clarity (10%): Is the reasoning understandable?

Extra rules:
- Reward valid reasoning, not confident wording.
- Penalize subtle logic gaps even when the final answer happens to be correct.
- Do not reward unnecessary verbosity unless it genuinely strengthens the reasoning.""",
}


@dataclass(frozen=True)
class EvaluatorModel:
    short_id: str
    provider: str
    kind: str
    model_id: str
    fmt: str | None = None


EVALUATOR_MODELS = {
    "gemini-2-5-flash": EvaluatorModel(
        short_id="gemini-2-5-flash",
        provider="Google",
        kind="vertex",
        model_id="gemini-2.5-flash",
    ),
    "nova-pro": EvaluatorModel(
        short_id="nova-pro",
        provider="Amazon",
        kind="bedrock",
        model_id="us.amazon.nova-pro-v1:0",
        fmt="nova",
    ),
    "mistral-large": EvaluatorModel(
        short_id="mistral-large",
        provider="Mistral AI",
        kind="bedrock",
        model_id="mistral.mistral-large-2402-v1:0",
        fmt="messages",
    ),
    "deepseek-r1": EvaluatorModel(
        short_id="deepseek-r1",
        provider="DeepSeek",
        kind="bedrock",
        model_id="us.deepseek.r1-v1:0",
        fmt="messages",
    ),
    "llama4-maverick": EvaluatorModel(
        short_id="llama4-maverick",
        provider="Meta",
        kind="bedrock",
        model_id=f"arn:aws:bedrock:{os.getenv('AWS_REGION', 'us-east-1')}:{os.getenv('AWS_ACCOUNT_ID', '')}:inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0",
        fmt="meta",
    ),
    "nova-premier": EvaluatorModel(
        short_id="nova-premier",
        provider="Amazon",
        kind="bedrock",
        model_id="us.amazon.nova-premier-v1:0",
        fmt="nova",
    ),
}


def _clean_json(text: str) -> str:
    return (
        (text or "")
        .strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _clean_json(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in evaluator output: {cleaned[:300]}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Evaluator returned JSON that was not an object")
    return data


def _normalize_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid accuracy score: {value}") from exc
    return max(0, min(100, score))


def _coerce_existing_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_prompt_hash(prompt: str) -> str:
    """SHA-256 of lowercased, whitespace-normalized prompt."""
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def compute_quality_flags(scores: list[float]) -> dict[str, Any]:
    """
    Compute evaluator agreement and confidence flags from named evaluator scores.
    """
    if len(scores) < 2:
        return {
            "score_stdev": None,
            "eval_conflict_flag": False,
            "high_conflict_flag": False,
            "low_confidence": True,
            "confidence_level": 0.3,
            "eval_count": len(scores),
        }

    score_range = max(scores) - min(scores)
    stdev = statistics.stdev(scores)
    conflict = score_range >= 25
    high_conflict = score_range >= 40
    confidence = max(0.0, round(1.0 - stdev / 50.0, 3))

    return {
        "score_stdev": round(stdev, 3),
        "eval_conflict_flag": conflict,
        "high_conflict_flag": high_conflict,
        "low_confidence": high_conflict or len(scores) < 2,
        "confidence_level": confidence,
        "eval_count": len(scores),
    }


def build_evaluator_prompt(use_case: str, prompt: str, response: str) -> str:
    normalized_use_case = (use_case or "text-generation").strip().lower()
    rubric = USE_CASE_PROMPTS.get(normalized_use_case, USE_CASE_PROMPTS["text-generation"])
    return (
        f"{BASE_EVALUATOR_INSTRUCTIONS}\n\n"
        f"{rubric}\n\n"
        f"USER PROMPT:\n{prompt.strip()}\n\n"
        f"MODEL RESPONSE:\n{response.strip()}\n"
    )


def _build_bedrock_body(fmt: str, prompt: str) -> str:
    if fmt == "meta":
        formatted = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return json.dumps({"prompt": formatted, "max_gen_len": 700})
    if fmt == "nova":
        return json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 700},
        })
    return json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
    })


def _extract_bedrock_text(fmt: str, body_json: dict[str, Any]) -> str:
    if fmt == "meta":
        return body_json.get("generation", "")
    if fmt == "nova":
        return (
            body_json.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
        )
    return (
        body_json.get("content", [{}])[0].get("text", "")
        or body_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        or body_json.get("outputs", [{}])[0].get("text", "")
        or ""
    )


def _get_vertex_client() -> genai.Client:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex evaluation")
    location = os.getenv("AVG_ACCURACY_VERTEX_LOCATION", "global").strip() or "global"
    return genai.Client(vertexai=True, project=project, location=location)


def _get_bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        _BEDROCK_CLIENT = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return _BEDROCK_CLIENT


def call_evaluator(model: EvaluatorModel, content: str) -> str:
    if model.kind == "vertex":
        client = _get_vertex_client()
        response = client.models.generate_content(
            model=model.model_id,
            contents=content,
        )
        return response.text or ""

    bedrock = _get_bedrock_client()
    raw = bedrock.invoke_model(
        modelId=model.model_id,
        body=_build_bedrock_body(model.fmt or "messages", content),
    )
    body_json = json.loads(raw["body"].read())
    return _extract_bedrock_text(model.fmt or "messages", body_json)


def _build_retry_prompt(base_content: str) -> str:
    return (
        f"{base_content}\n\n"
        "FINAL REMINDER:\n"
        'Return ONLY a single JSON object like {"accuracy_score": 87}. '
        "Do not add explanation, analysis, or markdown."
    )


def evaluate_with_model(
    model: EvaluatorModel,
    use_case: str,
    prompt: str,
    response: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> int:
    base_content = build_evaluator_prompt(use_case=use_case, prompt=prompt, response=response)

    for attempt in range(max_retries):
        try:
            content = base_content if attempt == 0 else _build_retry_prompt(base_content)
            raw_text = call_evaluator(model, content)
            if not (raw_text or "").strip():
                raise ValueError("Evaluator returned empty output")
            payload = _extract_json_object(raw_text)
            return _normalize_score(payload.get("accuracy_score"))
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"{model.short_id} failed after {max_retries} attempts: {exc}") from exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to evaluate with {model.short_id}")


def make_score_column_name(model_short_id: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_short_id).strip("_").lower()
    return f"eval_{safe_name}_score"


def resolve_models(names: list[str]) -> list[EvaluatorModel]:
    resolved = []
    for name in names:
        normalized = name.strip().lower()
        if normalized not in EVALUATOR_MODELS:
            raise ValueError(
                f"Unknown evaluator model '{name}'. Available: {', '.join(sorted(EVALUATOR_MODELS))}"
            )
        resolved.append(EVALUATOR_MODELS[normalized])
    return resolved


def resolve_evaluators_for_use_case(use_case: str, override_names: list[str] | None = None) -> list[EvaluatorModel]:
    if override_names:
        return resolve_models(override_names)
    return resolve_models(DEFAULT_USE_CASE_EVALUATORS[use_case])


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required")
    return create_client(url, key)


def fetch_rows(
    supabase: Client,
    table: str,
    limit: int | None,
    use_case: str | None,
    row_id: str | None,
    page_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    selected_columns = "id,prompt,response,use_case,accuracy_score,avg_accuracy_score,prompt_hash"

    while True:
        query = supabase.table(table).select(selected_columns).order("created_at", desc=False)
        if use_case:
            query = query.eq("use_case", use_case)
        if row_id:
            query = query.eq("id", row_id)

        batch_end = start + page_size - 1
        response = query.range(start, batch_end).execute()
        batch = response.data or []
        if not batch:
            break

        rows.extend(batch)
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
        if len(batch) < page_size or row_id:
            break
        start += page_size

    return rows


def should_process_row(row: dict[str, Any], force: bool) -> bool:
    if force:
        return True
    return row.get("avg_accuracy_score") is None


def evaluate_row(
    row: dict[str, Any],
    placeholder_score: float,
    override_names: list[str] | None = None,
) -> dict[str, Any]:
    prompt = str(row.get("prompt", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    use_case = str(row.get("use_case", "text-generation") or "text-generation").strip().lower()
    if use_case not in VALID_USE_CASES:
        use_case = "text-generation"
    if not prompt or not response:
        raise ValueError("Row must contain non-empty prompt and response")

    evaluators = resolve_evaluators_for_use_case(use_case, override_names=override_names)
    update_payload: dict[str, Any] = {}
    scores_for_average: list[float] = []
    successful_new_scores = 0
    evaluator_failures: list[str] = []

    for evaluator in evaluators:
        try:
            score = evaluate_with_model(
                model=evaluator,
                use_case=use_case,
                prompt=prompt,
                response=response,
            )
            column_name = make_score_column_name(evaluator.short_id)
            update_payload[column_name] = score
            scores_for_average.append(float(score))
            successful_new_scores += 1
        except Exception as exc:
            evaluator_failures.append(f"{evaluator.short_id}: {exc}")

    if successful_new_scores == 0:
        failure_summary = " | ".join(evaluator_failures) if evaluator_failures else "no evaluator scores returned"
        raise ValueError(f"All new evaluators failed. {failure_summary}")

    update_payload["avg_accuracy_score"] = round(sum(scores_for_average) / len(scores_for_average), 2)
    update_payload.update(compute_quality_flags(scores_for_average))
    update_payload["prompt_hash"] = compute_prompt_hash(prompt)
    update_payload["_meta_successful_new_scores"] = successful_new_scores
    update_payload["_meta_evaluator_failures"] = evaluator_failures
    return update_payload


def update_row(supabase: Client, table: str, row_id: str, payload: dict[str, Any]) -> None:
    supabase.table(table).update(payload).eq("id", row_id).execute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch benchmark rows from Supabase, score each with 2 new evaluators, and update avg_accuracy_score in place."
    )
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=None,
        help="Optional override evaluator model short ids. If omitted, use-case-specific evaluator sets are used.",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Supabase table name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel row workers.")
    parser.add_argument("--use-case", choices=sorted(VALID_USE_CASES), help="Optional use-case filter.")
    parser.add_argument("--row-id", help="Optional single row id to update.")
    parser.add_argument(
        "--placeholder-score",
        type=float,
        default=DEFAULT_PLACEHOLDER_SCORE,
        help="Deprecated; legacy accuracy_score is never included in avg_accuracy_score.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute rows even if avg_accuracy_score already exists.",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print the evaluator prompts/rubrics for each use case and exit.",
    )
    return parser.parse_args()


def print_prompts() -> None:
    for use_case in ("text-generation", "code-generation", "reasoning"):
        print("=" * 88)
        print(use_case)
        print("=" * 88)
        print(BASE_EVALUATOR_INSTRUCTIONS)
        print()
        print(USE_CASE_PROMPTS[use_case])
        print()


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    args = parse_args()

    if args.print_prompts:
        print_prompts()
        return 0

    supabase = get_supabase_client()
    rows = fetch_rows(
        supabase=supabase,
        table=args.table,
        limit=args.limit,
        use_case=args.use_case,
        row_id=args.row_id,
        page_size=DEFAULT_PAGE_SIZE,
    )

    if not rows:
        print("No rows found.")
        return 0

    pending_rows = [row for row in rows if should_process_row(row, force=args.force)]
    if not pending_rows:
        print("No pending rows to update.")
        return 0

    print(f"Fetched rows: {len(rows)}")
    print(f"Pending rows: {len(pending_rows)}")
    print(f"Table: {args.table}")
    if args.evaluators:
        print(f"Evaluator override: {', '.join(args.evaluators)}")
    else:
        print(
            "Use-case evaluators: "
            + "; ".join(
                f"{use_case}={','.join(models)}"
                for use_case, models in DEFAULT_USE_CASE_EVALUATORS.items()
            )
        )
    print("Legacy accuracy_score excluded from multi-evaluator average.")

    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_row = {
            executor.submit(
                evaluate_row,
                row,
                args.placeholder_score,
                args.evaluators,
            ): row
            for row in pending_rows
        }

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            completed += 1
            row_id = str(row.get("id"))
            try:
                payload = future.result()
                successful_new_scores = payload.pop("_meta_successful_new_scores", 0)
                evaluator_failures = payload.pop("_meta_evaluator_failures", [])
                update_row(supabase=supabase, table=args.table, row_id=row_id, payload=payload)
                if evaluator_failures:
                    print(
                        f"[{completed}/{len(pending_rows)}] updated row={row_id} "
                        f"avg={payload['avg_accuracy_score']} "
                        f"new_scores={successful_new_scores} "
                        f"partial_failures={' || '.join(evaluator_failures)}"
                    )
                else:
                    print(
                        f"[{completed}/{len(pending_rows)}] updated row={row_id} "
                        f"avg={payload['avg_accuracy_score']} new_scores={successful_new_scores}"
                    )
            except Exception as exc:
                failures += 1
                print(f"[{completed}/{len(pending_rows)}] row={row_id} FAILED: {exc}", file=sys.stderr)

    print(f"Finished. Failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
