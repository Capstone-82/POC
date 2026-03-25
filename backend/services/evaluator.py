"""
Evaluator service with Gemini round-robin client pool.

The evaluator now uses the user-selected use case directly:
  - text-generation
  - code-generation
  - reasoning

It keeps the existing batching and parallel execution model, but applies a
use-case-specific rubric so concise factual answers are not penalized for
being appropriately brief.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from services.gemini_clients import get_client, get_client_count

MAX_TOKENS_PER_BATCH = 3000
MAX_RETRIES = 3
RETRY_BASE_DELAY = 4.0
DEFAULT_USE_CASE = "text-generation"


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


PROMPT_EVAL_SYSTEM = """
You are an evaluator. Given a prompt, return a JSON object with exactly these fields:

{
  "prompt_complexity": "low" | "mid" | "high",
  "prompt_quality_score": <integer 0-100>
}

Return ONLY the JSON object. No explanation, no markdown.
"""


BASE_EVAL_RULES = """You are an expert AI response evaluator.

You will be given:
- the user-selected use case
- the original prompt
- one or more model responses

Score EACH response independently for how well it satisfies the prompt for that use case.

Before scoring, infer the EXPECTED ANSWER SCOPE from the prompt:
- concise: short factual or direct-answer prompts
- standard: normal explanatory prompts
- comprehensive: prompts that explicitly ask for depth, breadth, comparison, step-by-step detail, or production-ready output

Core evaluation principles:
- Judge responses against what the prompt actually asked for, not against the longest possible answer.
- A short correct answer can be the best answer for a concise prompt.
- Do NOT reward extra detail unless it clearly improves fulfillment of the prompt.
- Irrelevant detail, over-explaining, or scope drift should lower the score.
- "Completeness" means fully answering the task at the right scope, not adding unrelated information.

Scoring scale:
  95-100 = Excellent for this prompt and use case; correct, appropriately scoped, and hard to improve
  85-94  = Strong answer with only minor issues
  75-84  = Good answer with some limitations or unnecessary detail
  65-74  = Mixed quality; correct in core but with notable issues
  50-64  = Weak; partially correct, incomplete, or poorly scoped
  30-49  = Poor; significant errors or major mismatch to the task
  10-29  = Very poor; mostly wrong, broken, or badly off-target
  0-9    = No useful value

Mandatory scoring rules:
- Score EACH response on its own merits.
- Do NOT give a higher score just because a response is longer.
- Penalize irrelevant elaboration for concise prompts.
- A concise, correct answer to a concise factual question should typically score 95+.
- Incorrect factual content, broken code, invalid reasoning, or obvious hallucinations must be scored low.
- If one response is clearly better than another, their scores should differ meaningfully.

Return a JSON array with one object per response, in the SAME order as given:
[{"model_id": "<model_id>", "accuracy_score": <integer 0-100>}, ...]

Return ONLY the JSON array. No explanation, no markdown."""


USE_CASE_RUBRICS = {
    "text-generation": """USE CASE: text-generation

Apply this rubric for text-generation prompts:
1. CORRECTNESS & RELEVANCE (55%): Is the answer factually correct and directly responsive?
2. SCOPE FIT (25%): Is the answer appropriately concise or detailed for the prompt?
3. COMPLETENESS (15%): Does it fully answer the asked question at the right scope?
4. CLARITY (5%): Is it clear and easy to understand?

Extra rules for text-generation:
- For direct factual prompts, brevity is a strength.
- Unnecessary trivia, background, or decorative detail should reduce the score.
- Depth matters only when the prompt asks for explanation, comparison, nuance, or expansion.""",
    "code-generation": """USE CASE: code-generation

Apply this rubric for code-generation prompts:
1. CORRECTNESS & EXECUTABILITY (45%): Is the code or technical answer correct, consistent, and likely to work?
2. REQUIREMENT COVERAGE (25%): Does it satisfy the requested functionality and constraints?
3. PRACTICAL USEFULNESS (20%): Is it actionable, runnable, and implementation-ready?
4. CLARITY (10%): Is it organized and understandable?

Extra rules for code-generation:
- Working code that solves the prompt should score high even if concise.
- Broken code, fake APIs, missing critical pieces, or unsafe hallucinated behavior should score low.
- Extra explanation should help the solution; otherwise it should not improve the score.""",
    "reasoning": """USE CASE: reasoning

Apply this rubric for reasoning prompts:
1. LOGICAL SOUNDNESS (40%): Are the reasoning steps valid and coherent?
2. FINAL ANSWER CORRECTNESS (30%): Is the final answer right?
3. COMPLETENESS OF REASONING (20%): Are the necessary steps covered without major gaps?
4. CLARITY (10%): Is the reasoning understandable?

Extra rules for reasoning:
- Reward correct reasoning, not just confident wording.
- Penalize invalid logic even if the answer sounds polished.
- Do not reward unnecessary elaboration that does not strengthen the reasoning.""",
}


def _normalize_use_case(use_case: str | None) -> str:
    normalized = (use_case or DEFAULT_USE_CASE).strip().lower()
    if normalized not in USE_CASE_RUBRICS:
        return DEFAULT_USE_CASE
    return normalized


def _build_batch_eval_system(use_case: str | None) -> str:
    normalized = _normalize_use_case(use_case)
    return f"{BASE_EVAL_RULES}\n\n{USE_CASE_RUBRICS[normalized]}"


async def evaluate_prompt(prompt: str, evaluator_model: str = None) -> dict:
    """Evaluate a prompt's complexity and quality for inference routing."""
    pool_entry = get_client()
    client = pool_entry["client"]
    model = pool_entry["model"]

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=f"{PROMPT_EVAL_SYSTEM}\n\nPrompt to evaluate:\n{prompt}",
        )
        return json.loads(_clean_json(response.text))

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call)


def _create_batches(responses: list[dict]) -> list[list[dict]]:
    """
    Group responses into batches based on token count.
    Each batch stays under MAX_TOKENS_PER_BATCH total response tokens.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    sorted_responses = sorted(responses, key=lambda r: len(r["response"]))

    for response in sorted_responses:
        truncated = response["response"][:3000]
        tokens = _estimate_tokens(truncated)

        if current_batch and (current_tokens + tokens > MAX_TOKENS_PER_BATCH):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(response)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _evaluate_batch_sync(
    prompt: str,
    use_case: str,
    batch: list[dict],
    pool_entry: dict,
) -> list[dict]:
    """Evaluate a batch of responses synchronously (runs in a thread pool)."""
    client = pool_entry["client"]
    model = pool_entry["model"]
    label = pool_entry["label"]
    system_prompt = _build_batch_eval_system(use_case)
    normalized_use_case = _normalize_use_case(use_case)

    responses_text = "\n\n---\n\n".join(
        f"[Response {i + 1}] model_id: {response['model_id']}\n{response['response'][:3000]}"
        for i, response in enumerate(batch)
    )

    content = (
        f"{system_prompt}\n\n"
        f"USER-SELECTED USE CASE:\n{normalized_use_case}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL RESPONSES ({len(batch)} total):\n\n{responses_text}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.generate_content(
                model=model,
                contents=content,
            )
            scores = json.loads(_clean_json(result.text))

            if isinstance(scores, list) and len(scores) >= len(batch):
                return scores[: len(batch)]

            existing_ids = {score.get("model_id") for score in scores} if isinstance(scores, list) else set()
            normalized_scores = scores if isinstance(scores, list) else []
            for response in batch:
                if response["model_id"] not in existing_ids:
                    normalized_scores.append({"model_id": response["model_id"], "accuracy_score": 50})
            return normalized_scores

        except Exception as exc:
            err_str = str(exc).lower()
            is_retryable = any(
                token in err_str for token in ["429", "rate", "quota", "503", "overload", "resource"]
            )

            if is_retryable and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                batch_ids = ",".join(response["model_id"] for response in batch)
                print(
                    f"[EVAL RETRY] {label} {normalized_use_case} batch({batch_ids}) "
                    f"attempt {attempt + 1}/{MAX_RETRIES}, wait {delay}s"
                )
                import time

                time.sleep(delay)
            else:
                print(f"[EVAL ERROR] {label} {normalized_use_case}: {exc}")
                return [{"model_id": response["model_id"], "accuracy_score": 50} for response in batch]

    return [{"model_id": response["model_id"], "accuracy_score": 50} for response in batch]


async def evaluate_all_responses(
    prompt: str,
    responses: list[dict],
    use_case: str = DEFAULT_USE_CASE,
    evaluator_model: str = None,
) -> list[dict]:
    """
    Parallel batched evaluation using the round-robin client pool.

    The user-selected use case is passed directly into the evaluator prompt so
    the model grades with the right rubric instead of guessing task type.
    """
    batches = _create_batches(responses)
    num_clients = get_client_count()
    normalized_use_case = _normalize_use_case(use_case)
    print(
        f"[EVALUATOR] {len(responses)} responses -> {len(batches)} batches "
        f"across {num_clients} clients for use_case={normalized_use_case}"
    )

    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = []
        for batch in batches:
            pool_entry = get_client()
            futures.append(
                loop.run_in_executor(
                    executor,
                    _evaluate_batch_sync,
                    prompt,
                    normalized_use_case,
                    batch,
                    pool_entry,
                )
            )
        results = await asyncio.gather(*futures, return_exceptions=True)

    all_scores = []
    for index, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[EVAL ERROR] batch {index}: {result}")
            all_scores.extend(
                [{"model_id": response["model_id"], "accuracy_score": 50} for response in batches[index]]
            )
        else:
            all_scores.extend(result)

    return all_scores


async def evaluate_response(
    prompt: str,
    response: str,
    use_case: str = DEFAULT_USE_CASE,
    evaluator_model: str = None,
) -> dict:
    """Evaluate a single response using the same use-case-aware rubric."""
    pool_entry = get_client()
    client = pool_entry["client"]
    model = pool_entry["model"]
    system_prompt = _build_batch_eval_system(use_case)
    normalized_use_case = _normalize_use_case(use_case)

    def _call():
        content = (
            f"{system_prompt}\n\n"
            f"USER-SELECTED USE CASE:\n{normalized_use_case}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"MODEL RESPONSES (1 total):\n\n[Response 1] model_id: single\n{response[:3000]}"
        )
        result = client.models.generate_content(
            model=model,
            contents=content,
        )
        scores = json.loads(_clean_json(result.text))
        if isinstance(scores, list) and scores:
            return {"accuracy_score": scores[0].get("accuracy_score", 50)}
        return {"accuracy_score": 50}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call)
