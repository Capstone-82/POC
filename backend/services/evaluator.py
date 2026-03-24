"""
Evaluator service — Gemini via Vertex AI (google-genai SDK).

Smart batching: groups responses dynamically by size to minimize API calls
while keeping evaluator context manageable for accurate scoring.
  - Short responses → batch 8-10 per call
  - Medium responses → batch 3-4 per call
  - Long responses → 1-2 per call
Sequential calls with delay between batches to avoid 429s.
"""

import json
import os
import asyncio
from google import genai

# ─── Vertex AI Client ─────────────────────────────────────────

client = genai.Client(
    vertexai=True,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

EVALUATOR_MODELS = {
    "gemini-2.0-flash":      "gemini-2.0-flash",
    "gemini-2.5-flash":      "gemini-2.5-flash",
    "gemini-2.5-pro":        "gemini-2.5-pro",
}

DEFAULT_EVALUATOR = "gemini-2.0-flash"

# ─── Config ───────────────────────────────────────────────────
MAX_TOKENS_PER_BATCH = 3000   # ~3000 tokens of response text per batch
DELAY_BETWEEN_BATCHES = 2.0   # seconds between evaluator calls
MAX_RETRIES = 3
RETRY_BASE_DELAY = 4.0        # 4s → 8s → 16s


def _clean_json(text: str) -> str:
    return (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


# ─── Evaluate the prompt (kept for inference route) ──────────

PROMPT_EVAL_SYSTEM = """
You are an evaluator. Given a prompt, return a JSON object with exactly these fields:

{
  "prompt_complexity": "low" | "mid" | "high",
  "prompt_quality_score": <integer 0-100>
}

Return ONLY the JSON object. No explanation, no markdown.
"""


async def evaluate_prompt(prompt: str, evaluator_model: str = DEFAULT_EVALUATOR) -> dict:
    model_name = EVALUATOR_MODELS.get(evaluator_model)
    if not model_name:
        raise ValueError(f"Unsupported evaluator: {evaluator_model}")

    response = client.models.generate_content(
        model=model_name,
        contents=f"{PROMPT_EVAL_SYSTEM}\n\nPrompt to evaluate:\n{prompt}",
    )
    return json.loads(_clean_json(response.text))


# ─── Smart batching logic ────────────────────────────────────

def _create_batches(responses: list[dict]) -> list[list[dict]]:
    """
    Group responses into batches based on token count.
    Each batch stays under MAX_TOKENS_PER_BATCH total response tokens.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    # Sort by response length (short first) so we pack efficiently
    sorted_responses = sorted(responses, key=lambda r: len(r["response"]))

    for r in sorted_responses:
        truncated = r["response"][:3000]
        tokens = _estimate_tokens(truncated)

        # If adding this response would exceed budget, start a new batch
        if current_batch and (current_tokens + tokens > MAX_TOKENS_PER_BATCH):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(r)
        current_tokens += tokens

    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)

    return batches


# ─── Batch evaluation prompt ─────────────────────────────────

BATCH_EVAL_SYSTEM = """You are an expert AI response evaluator. You will be given a prompt and one or more model responses. Score EACH response independently.

SCORING CRITERIA (apply to EACH response separately):
1. CORRECTNESS (40%): Facts, code, logic accurate? Code compiles? Sound reasoning?
2. COMPLETENESS (25%): Fully addresses ALL parts of the prompt?
3. DEPTH & QUALITY (20%): Insightful, well-structured, expert-level?
4. PRACTICAL USEFULNESS (15%): Runnable code? Actionable instructions?

SCORING SCALE — you MUST differentiate between responses:
  95-100 = Exceptional. Zero errors, comprehensive, novel insights
  85-94  = Excellent. Correct, complete, minor imperfections
  75-84  = Very Good. Mostly correct, a few small gaps
  65-74  = Good. Correct overall, notable omissions
  55-64  = Adequate. Core right, significant gaps or some errors
  45-54  = Mediocre. Partially correct, missing important aspects
  35-44  = Below Average. Major errors or very incomplete
  25-34  = Poor. Mostly incorrect or off-topic
  15-24  = Very Poor. Fundamentally wrong approach
  5-14   = Terrible. Gibberish or completely off-topic
  0-4    = No value. Empty or nonsensical

MANDATORY RULES:
- Score EACH response on its own merits — do NOT give the same score to all
- Working, correct code that solves the prompt = 75+
- Broken, nonsensical, or hallucinated code = below 30
- Partial answer with gaps = 45-65
- If response A is clearly better than B, their scores MUST differ by at least 10 points
- A response with gibberish or invalid syntax MUST score below 20

Return a JSON array with one object per response, in the SAME order as given:
[{"model_id": "<model_id>", "accuracy_score": <integer 0-100>}, ...]

Return ONLY the JSON array. No explanation, no markdown."""


async def _evaluate_batch_with_retry(
    prompt: str,
    batch: list[dict],
    model_name: str,
) -> list[dict]:
    """Evaluate a batch of responses with retry logic."""
    # Build the numbered response list
    responses_text = "\n\n---\n\n".join(
        f"[Response {i+1}] model_id: {r['model_id']}\n{r['response'][:3000]}"
        for i, r in enumerate(batch)
    )

    content = (
        f"{BATCH_EVAL_SYSTEM}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL RESPONSES ({len(batch)} total):\n\n{responses_text}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.generate_content(model=model_name, contents=content)
            scores = json.loads(_clean_json(result.text))

            # Validate we got the right number of scores
            if isinstance(scores, list) and len(scores) >= len(batch):
                return scores[:len(batch)]

            # If fewer scores returned, pad with defaults
            existing_ids = {s.get("model_id") for s in scores}
            for r in batch:
                if r["model_id"] not in existing_ids:
                    scores.append({"model_id": r["model_id"], "accuracy_score": 50})
            return scores

        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(k in err_str for k in ["429", "rate", "quota", "503", "overload", "resource"])

            if is_retryable and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[EVAL RETRY] batch({','.join(r['model_id'] for r in batch)}) attempt {attempt+1}/{MAX_RETRIES}, wait {delay}s")
                await asyncio.sleep(delay)
            else:
                print(f"[EVAL ERROR] batch: {e}")
                return [{"model_id": r["model_id"], "accuracy_score": 50} for r in batch]

    return [{"model_id": r["model_id"], "accuracy_score": 50} for r in batch]


async def evaluate_all_responses(
    prompt: str,
    responses: list[dict],
    evaluator_model: str = DEFAULT_EVALUATOR,
) -> list[dict]:
    """
    Smart batched evaluation:
      1. Group responses by size into optimal batches
      2. Send each batch sequentially with delay between calls
      3. Collect all scores
    """
    model_name = EVALUATOR_MODELS.get(evaluator_model)
    if not model_name:
        raise ValueError(f"Unsupported evaluator: {evaluator_model}")

    batches = _create_batches(responses)
    print(f"[EVALUATOR] {len(responses)} responses → {len(batches)} batches: {[len(b) for b in batches]}")

    all_scores = []
    for i, batch in enumerate(batches):
        scores = await _evaluate_batch_with_retry(prompt, batch, model_name)
        all_scores.extend(scores)

        # Delay between batches (skip delay after last batch)
        if i < len(batches) - 1:
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)

    return all_scores


# ─── Single response evaluator (inference route) ─────────────

async def evaluate_response(
    prompt: str, response: str, evaluator_model: str = DEFAULT_EVALUATOR
) -> dict:
    model_name = EVALUATOR_MODELS.get(evaluator_model)
    if not model_name:
        raise ValueError(f"Unsupported evaluator: {evaluator_model}")

    content = (
        f"{BATCH_EVAL_SYSTEM}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL RESPONSES (1 total):\n\n[Response 1] model_id: single\n{response[:3000]}"
    )
    result = client.models.generate_content(model=model_name, contents=content)
    scores = json.loads(_clean_json(result.text))
    if isinstance(scores, list) and scores:
        return {"accuracy_score": scores[0].get("accuracy_score", 50)}
    return {"accuracy_score": 50}
