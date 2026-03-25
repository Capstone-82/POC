import json
import os
from typing import Any

import httpx


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_CLARITY_MODEL", "gpt-4.1").strip() or "gpt-4.1"

VALID_LABELS = {"CLEAR", "PARTIAL", "UNCLEAR"}

CLARITY_SYSTEM_PROMPT = """You are a dataset labeling assistant.

Your task is to classify each prompt into exactly one clarity label:
- CLEAR
- PARTIAL
- UNCLEAR

Label definitions:

CLEAR:
- The prompt is specific, understandable, and actionable.
- The task, intent, and expected output are mostly clear.
- A capable model can respond well without making major assumptions.
- Short prompts can still be CLEAR if they ask for a simple direct task.

PARTIAL:
- The prompt is understandable enough to attempt, but it contains ambiguity, missing constraints, or underspecified intent.
- A capable model can respond, but would need to make some assumptions.
- The main goal is visible, but important details are fuzzy or incomplete.

UNCLEAR:
- The prompt is too vague, fragmented, contradictory, context-dependent, or incomplete to answer reliably.
- A capable model would have to guess too much about what the user wants.
- The task itself is not meaningfully understandable from the prompt alone.

Decision rules:
- Judge the prompt itself, not whether the topic is easy or difficult.
- Do not reward length. Long prompts can still be PARTIAL or UNCLEAR.
- Do not penalize brevity. A short prompt can be CLEAR if the requested task is explicit.
- If the main task is understandable but some constraints or details are missing, choose PARTIAL.
- If the prompt depends on missing external context like "fix this", "summarize this", or "do it like before" with no usable context, choose UNCLEAR.
- Be conservative and consistent across the whole batch.
- Return exactly one label per prompt.

Examples:
- "What is the capital of France?" -> CLEAR
- "Write a Python function to remove duplicates from a list." -> CLEAR
- "Write an email for client update, make it good." -> PARTIAL
- "Create a dashboard for sales with charts and export support." -> PARTIAL
- "Fix this code." -> UNCLEAR
- "Summarize this and make it shorter." -> UNCLEAR

Return JSON only."""


def _build_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "clarity_batch_labels",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "prompt_id": {"type": "string"},
                                "clarity": {
                                    "type": "string",
                                    "enum": ["CLEAR", "PARTIAL", "UNCLEAR"],
                                },
                            },
                            "required": ["prompt_id", "clarity"],
                        },
                    }
                },
                "required": ["results"],
            },
        },
    }


def _extract_json_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("OpenAI response did not contain choices")

    message = choices[0].get("message", {})
    parsed = message.get("parsed")
    if parsed:
        return json.dumps(parsed)

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                text_parts.append(item.get("text", ""))
        if text_parts:
            return "".join(text_parts)

    raise ValueError("Could not extract JSON content from OpenAI response")


async def classify_prompt_batch(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if not batch:
        return []

    prompt_lines = []
    for item in batch:
        prompt_lines.append(
            f'prompt_id: "{item["prompt_id"]}"\nprompt: """{item["prompt"]}"""'
        )

    user_prompt = (
        "Classify the following prompts for clarity.\n\n"
        "Return one result per prompt_id.\n\n"
        + "\n\n---\n\n".join(prompt_lines)
    )

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": CLARITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": _build_response_schema(),
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "OpenAI authentication failed for the clarity pipeline. "
                "Check that backend/.env contains one valid OPENAI_API_KEY and restart the backend."
            )
        response.raise_for_status()
        payload = response.json()

    data = json.loads(_extract_json_content(payload))
    results = data.get("results", [])
    if len(results) != len(batch):
        raise ValueError(f"Expected {len(batch)} clarity labels, got {len(results)}")

    expected_ids = [item["prompt_id"] for item in batch]
    labels_by_id: dict[str, str] = {}
    for item in results:
        prompt_id = str(item.get("prompt_id", ""))
        clarity = str(item.get("clarity", "")).upper()
        if prompt_id not in expected_ids:
            raise ValueError(f"Unexpected prompt_id returned by OpenAI: {prompt_id}")
        if clarity not in VALID_LABELS:
            raise ValueError(f"Invalid clarity label returned by OpenAI: {clarity}")
        labels_by_id[prompt_id] = clarity

    missing_ids = [prompt_id for prompt_id in expected_ids if prompt_id not in labels_by_id]
    if missing_ids:
        raise ValueError(f"Missing clarity labels for prompt_ids: {', '.join(missing_ids)}")

    return [
        {"prompt_id": prompt_id, "clarity": labels_by_id[prompt_id]}
        for prompt_id in expected_ids
    ]
