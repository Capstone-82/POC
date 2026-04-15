from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from generate_avg_accuracy_scores import (
    DEFAULT_USE_CASE_EVALUATORS,
    _extract_json_object,
    build_evaluator_prompt,
    call_evaluator,
    get_supabase_client,
    resolve_evaluators_for_use_case,
)

DEFAULT_SAMPLE_USE_CASE = "reasoning"
DEFAULT_SAMPLE_PROMPT = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
DEFAULT_SAMPLE_RESPONSE = (
    "Let the ball cost x dollars. Then the bat costs x + 1.00. "
    "So x + (x + 1.00) = 1.10, which means 2x = 0.10 and x = 0.05. "
    "The ball costs $0.05."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug evaluator model outputs by printing raw responses before JSON parsing."
    )
    parser.add_argument(
        "--use-case",
        choices=sorted(DEFAULT_USE_CASE_EVALUATORS),
        default="text-generation",
        help="Use case to test.",
    )
    parser.add_argument("--row-id", help="Optional benchmark_results row id from Supabase.")
    parser.add_argument("--prompt", help="Prompt text to test directly.")
    parser.add_argument("--response", help="Response text to test directly.")
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=None,
        help="Optional evaluator override. Otherwise use the default set for the chosen use case.",
    )
    return parser.parse_args()


def load_case_from_supabase(row_id: str) -> tuple[str, str, str]:
    supabase = get_supabase_client()
    result = (
        supabase.table("benchmark_results")
        .select("id,prompt,response,use_case")
        .eq("id", row_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise SystemExit(f"No row found for id {row_id}")

    row = rows[0]
    return (
        str(row.get("use_case") or "text-generation"),
        str(row.get("prompt") or ""),
        str(row.get("response") or ""),
    )


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    args = parse_args()

    if args.row_id:
        use_case, prompt, response = load_case_from_supabase(args.row_id)
    else:
        if args.prompt and args.response:
            use_case = args.use_case
            prompt = args.prompt
            response = args.response
        else:
            use_case = DEFAULT_SAMPLE_USE_CASE
            prompt = DEFAULT_SAMPLE_PROMPT
            response = DEFAULT_SAMPLE_RESPONSE

    evaluators = resolve_evaluators_for_use_case(use_case, override_names=args.evaluators)
    evaluator_prompt = build_evaluator_prompt(use_case=use_case, prompt=prompt, response=response)

    print("=" * 100)
    print(f"use_case: {use_case}")
    print(f"evaluators: {', '.join(model.short_id for model in evaluators)}")
    print("=" * 100)
    print("PROMPT SENT TO EVALUATOR")
    print("-" * 100)
    print(evaluator_prompt)
    print("-" * 100)

    for model in evaluators:
        print()
        print("=" * 100)
        print(f"MODEL: {model.short_id} ({model.provider})")
        print("=" * 100)
        try:
            raw_output = call_evaluator(model, evaluator_prompt)
            print("RAW OUTPUT")
            print("-" * 100)
            print(raw_output if raw_output else "<EMPTY STRING>")
            print("-" * 100)

            try:
                parsed = _extract_json_object(raw_output)
                print("PARSED JSON")
                print(json.dumps(parsed, indent=2))
            except Exception as exc:
                print(f"PARSE FAILED: {exc}")
        except Exception as exc:
            print(f"CALL FAILED: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
