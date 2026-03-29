"""
Relabel prompt complexity from the canonical deduplicated prompt CSV.

Designed for safe incremental runs:
- batches prompts
- writes output after every batch
- skips rows already present in the output CSV
- retries on rate limits / transient failures

Example test run:
  python relabel_complexity_test.py --limit 30

Full run:
  python relabel_complexity_test.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "Supabase Snippet Deduplicate Benchmark Results by Prompt.csv"
OUTPUT_CSV = BASE_DIR / "prompt_complexity_relabeled.csv"
DEFAULT_MODEL = os.getenv("OPENAI_COMPLEXITY_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

VALID_USE_CASES = {"text-generation", "code-generation", "reasoning"}
VALID_CLARITIES = {"CLEAR", "PARTIAL", "UNCLEAR"}
VALID_COMPLEXITIES = {"low", "mid", "high"}

SYSTEM_PROMPT = """You are a dataset labeling assistant.

Your job is to assign one prompt complexity label to each prompt:
- low
- mid
- high

You will be given:
- prompt_id
- use_case
- clarity
- prompt

Complexity means the expected difficulty for a capable LLM to satisfy the prompt well.
Judge complexity from the prompt itself, not from any existing human label.

Use these definitions:

low:
- direct factual questions
- simple writing prompts with minimal constraints
- basic code tasks such as a small function or easy algorithm
- short reasoning with one or two straightforward steps

mid:
- moderate reasoning or constraint handling
- writing prompts with multiple instructions or formatting requirements
- code prompts needing a complete but bounded solution
- prompts that require several meaningful steps, but not system-level design

high:
- system design, architecture, or production-grade implementation
- complex multi-step reasoning with many constraints
- advanced technical design, tradeoff analysis, or broad solution scope
- prompts that require deep coverage, robustness, or multiple interacting components

Important rules:
- Do not use prompt length alone as a proxy for complexity.
- A short prompt can still be high if it implies deep design or advanced reasoning.
- A long prompt can still be low or mid if the task itself is straightforward.
- Complexity is use-case aware. The same wording may imply different difficulty under different use cases.
- Clarity is provided as context but should not dominate the decision.
- Be strict and consistent across the entire dataset.

Return JSON only."""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "complexity_batch_labels",
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
                                "prompt_complexity": {
                                    "type": "string",
                                    "enum": ["low", "mid", "high"],
                                },
                                "complexity_reason": {"type": "string"},
                            },
                            "required": ["prompt_id", "prompt_complexity", "complexity_reason"],
                        },
                    }
                },
                "required": ["results"],
            },
        },
    }


def build_user_prompt(batch: list[dict[str, str]]) -> str:
    parts = []
    for item in batch:
        parts.append(
            "\n".join(
                [
                    f'prompt_id: "{item["prompt_id"]}"',
                    f'use_case: "{item["use_case"]}"',
                    f'clarity: "{item["clarity"]}"',
                    f'prompt: """{item["prompt"]}"""',
                ]
            )
        )
    return (
        "Label the following prompts for prompt complexity.\n"
        "Return one result per prompt_id.\n\n"
        + "\n\n---\n\n".join(parts)
    )


def extract_json_content(payload: dict[str, Any]) -> str:
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


def load_input_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    cleaned = []
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt", "")).strip()
        use_case = str(row.get("use_case", "")).strip().lower()
        clarity = str(row.get("clarity", "")).strip().upper()
        old_complexity = str(row.get("prompt_complexity", "")).strip().lower()
        if not prompt or use_case not in VALID_USE_CASES or clarity not in VALID_CLARITIES:
            continue
        cleaned.append(
            {
                "prompt_id": str(index),
                "prompt": prompt,
                "use_case": use_case,
                "clarity": clarity,
                "old_prompt_complexity": old_complexity,
            }
        )
    return cleaned


def load_completed_prompt_keys() -> set[tuple[str, str]]:
    if not OUTPUT_CSV.exists():
        return set()
    with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return {
            (str(row.get("prompt", "")).strip(), str(row.get("use_case", "")).strip().lower())
            for row in rows
            if row.get("prompt") and row.get("use_case")
        }


def append_results(rows: list[dict[str, str]]) -> None:
    write_header = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "prompt",
                "use_case",
                "prompt_complexity",
                "clarity",
                "old_prompt_complexity",
                "complexity_reason",
                "label_source",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def classify_batch(
    batch: list[dict[str, str]],
    model: str,
    timeout_seconds: float,
    max_retries: int,
) -> list[dict[str, str]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(batch)},
        ],
        "response_format": response_schema(),
        "temperature": 0,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"Transient OpenAI error {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            payload = response.json()
            data = json.loads(extract_json_content(payload))
            results = data.get("results", [])
            if len(results) != len(batch):
                raise ValueError(f"Expected {len(batch)} results, got {len(results)}")

            expected_ids = {item["prompt_id"] for item in batch}
            normalized = []
            for item in results:
                prompt_id = str(item.get("prompt_id", "")).strip()
                prompt_complexity = str(item.get("prompt_complexity", "")).strip().lower()
                complexity_reason = str(item.get("complexity_reason", "")).strip()
                if prompt_id not in expected_ids:
                    raise ValueError(f"Unexpected prompt_id returned: {prompt_id}")
                if prompt_complexity not in VALID_COMPLEXITIES:
                    raise ValueError(f"Invalid complexity returned: {prompt_complexity}")
                normalized.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_complexity": prompt_complexity,
                        "complexity_reason": complexity_reason,
                    }
                )
            return normalized
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            sleep_seconds = min(30, 2 ** attempt * 2)
            print(f"[retry] batch failed ({exc}); sleeping {sleep_seconds}s")
            time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel prompt complexity with an LLM.")
    parser.add_argument("--input", default=str(INPUT_CSV), help="Input canonical prompt CSV.")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use.")
    parser.add_argument("--batch-size", type=int, default=12, help="Prompts per API call.")
    parser.add_argument("--delay-seconds", type=float, default=1.5, help="Delay between successful batches.")
    parser.add_argument("--limit", type=int, help="Only process the first N remaining prompts.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per batch.")
    parser.add_argument("--timeout-seconds", type=float, default=90.0, help="Request timeout.")
    parser.add_argument("--dry-run", action="store_true", help="Print a sample batch payload and exit.")
    return parser.parse_args()


def main() -> None:
    load_env_file(BASE_DIR.parent / "backend" / ".env")

    global INPUT_CSV, OUTPUT_CSV, OPENAI_API_KEY
    args = parse_args()
    INPUT_CSV = Path(args.input)
    OUTPUT_CSV = Path(args.output)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

    rows = load_input_rows()
    completed_keys = load_completed_prompt_keys()
    pending = [
        row for row in rows
        if (row["prompt"], row["use_case"]) not in completed_keys
    ]

    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print("No pending prompts to process.")
        return

    if args.dry_run:
        sample = pending[: min(args.batch_size, len(pending))]
        print(json.dumps({"model": args.model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(sample)}]}, indent=2))
        return

    print(f"Input rows: {len(rows)}")
    print(f"Already completed: {len(completed_keys)}")
    print(f"Pending this run: {len(pending)}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_results = classify_batch(
            batch=batch,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        result_by_id = {item["prompt_id"]: item for item in batch_results}

        rows_to_write = []
        for row in batch:
            labeled = result_by_id[row["prompt_id"]]
            rows_to_write.append(
                {
                    "prompt": row["prompt"],
                    "use_case": row["use_case"],
                    "prompt_complexity": labeled["prompt_complexity"],
                    "clarity": row["clarity"],
                    "old_prompt_complexity": row["old_prompt_complexity"],
                    "complexity_reason": labeled["complexity_reason"],
                    "label_source": f"openai:{args.model}",
                }
            )

        append_results(rows_to_write)
        print(f"[ok] processed {min(start + len(batch), len(pending))}/{len(pending)} prompts")

        if start + args.batch_size < len(pending):
            time.sleep(max(0.0, args.delay_seconds))

    print(f"Done. Wrote labels to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
