"""
Evaluator service — all calls go to Gemini via Vertex AI (google-genai SDK).

Two functions:
  evaluate_prompt()        → 1 call → {prompt_complexity, prompt_quality_score}
  evaluate_all_responses() → 1 call → [{model_id, accuracy_score}, ...] for ALL models at once
"""

import json
import os
from google import genai

# ─── Vertex AI Client ─────────────────────────────────────────

client = genai.Client(
    vertexai=True,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

EVALUATOR_MODELS = {
    "gemini-3.1-pro":        "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini-2.0-flash":      "gemini-2.0-flash",
    "gemini-2.5-flash":      "gemini-2.5-flash",
    "gemini-2.5-pro":        "gemini-2.5-pro",
}

DEFAULT_EVALUATOR = "gemini-2.0-flash"


def _clean_json(text: str) -> str:
    """Strip markdown fences if the model wraps JSON in ```json ... ```."""
    return (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


# ─── CALL 1: Evaluate the prompt ─────────────────────────────

PROMPT_EVAL_SYSTEM = """
You are an evaluator. Given a prompt, return a JSON object with exactly these fields:

{
  "prompt_complexity": "low" | "mid" | "high",
  "prompt_quality_score": <integer 0-100>
}

Complexity scoring:
  low  = simple factual questions, basic tasks, one-step instructions
  mid  = multi-step tasks, moderate reasoning, some domain knowledge needed
  high = complex reasoning, expert domain, multi-constraint problems

Quality scoring bands:
  90-100 = perfectly clear, specific, unambiguous
  70-89  = mostly clear, minor gaps
  40-69  = somewhat clear, notable ambiguity
  10-39  = vague or poorly structured
  0-9    = incomprehensible

Return ONLY the JSON object. No explanation, no markdown.
"""


async def evaluate_prompt(prompt: str, evaluator_model: str = DEFAULT_EVALUATOR) -> dict:
    """
    Single evaluator call → returns {prompt_complexity, prompt_quality_score}.
    """
    model_name = EVALUATOR_MODELS.get(evaluator_model)
    if not model_name:
        raise ValueError(f"Unsupported evaluator: {evaluator_model}")

    response = client.models.generate_content(
        model=model_name,
        contents=f"{PROMPT_EVAL_SYSTEM}\n\nPrompt to evaluate:\n{prompt}",
    )
    return json.loads(_clean_json(response.text))


# ─── CALL 2: Evaluate ALL responses in one shot ───────────────

BATCH_RESPONSE_EVAL_SYSTEM = """
You are an evaluator. Given a prompt and a list of model responses, score each response for accuracy.

Return a JSON array where each element has exactly:
  { "model_id": "<the model_id you were given>", "accuracy_score": <integer 0-100> }

Scoring bands:
  90-100 = completely correct, complete, directly addresses the prompt
  70-89  = mostly correct, minor omissions or small errors
  40-69  = partially correct, notable gaps or errors
  10-39  = mostly incorrect or off-topic
  0-9    = completely wrong or irrelevant

Return ONLY the JSON array. No explanation, no markdown. One entry per response, in the same order.
"""


async def evaluate_all_responses(
    prompt: str,
    responses: list[dict],   # [{"model_id": "...", "response": "..."}, ...]
    evaluator_model: str = DEFAULT_EVALUATOR,
) -> list[dict]:
    """
    Single evaluator call for ALL model responses at once.
    Returns [{model_id, accuracy_score}, ...] in the same order as input.
    """
    model_name = EVALUATOR_MODELS.get(evaluator_model)
    if not model_name:
        raise ValueError(f"Unsupported evaluator: {evaluator_model}")

    # Build the numbered response list
    responses_text = "\n\n".join(
        f"[{i+1}] model_id: {r['model_id']}\nResponse: {r['response'][:2000]}"
        for i, r in enumerate(responses)
    )

    content = (
        f"{BATCH_RESPONSE_EVAL_SYSTEM}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Model Responses:\n{responses_text}"
    )

    result = client.models.generate_content(model=model_name, contents=content)
    scores = json.loads(_clean_json(result.text))

    # Safety: if model returns fewer scores than responses, pad with 50
    if len(scores) < len(responses):
        existing_ids = {s["model_id"] for s in scores}
        for r in responses:
            if r["model_id"] not in existing_ids:
                scores.append({"model_id": r["model_id"], "accuracy_score": 50})

    return scores


# ─── LEGACY: single response evaluator (kept for inference route) ─

RESPONSE_EVAL_SYSTEM = """
You are an evaluator. Given a prompt and a model's response, return a JSON object:

{
  "accuracy_score": <integer 0-100>
}

Scoring bands:
  90-100 = completely correct, complete, directly addresses the prompt
  70-89  = mostly correct, minor omissions or small errors
  40-69  = partially correct, notable gaps or errors
  10-39  = mostly incorrect or off-topic
  0-9    = completely wrong or irrelevant

Return ONLY the JSON object. No explanation, no markdown.
"""


async def evaluate_response(
    prompt: str, response: str, evaluator_model: str = DEFAULT_EVALUATOR
) -> dict:
    """Single response evaluator — used by the inference route."""
    model_name = EVALUATOR_MODELS.get(evaluator_model)
    if not model_name:
        raise ValueError(f"Unsupported evaluator: {evaluator_model}")

    content = (
        f"{RESPONSE_EVAL_SYSTEM}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Response:\n{response}"
    )
    result = client.models.generate_content(model=model_name, contents=content)
    return json.loads(_clean_json(result.text))
