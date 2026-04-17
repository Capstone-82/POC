from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from generate_avg_accuracy_scores import EVALUATOR_MODELS, call_evaluator, get_supabase_client


DEFAULT_JUDGES = ("llama4-maverick", "mistral-large", "nova-premier")
DEFAULT_TOP_K = 5
DEFAULT_MAX_WORKERS = 6
VALID_WINNERS = {"A", "B", "TIE"}


PAIRWISE_SYSTEM = """You are evaluating two LLM responses. Be critical and decisive.

You will be given:
- the original user prompt
- Response A
- Response B

Choose which response better addresses the prompt.
Consider:
- accuracy
- completeness
- conciseness

Return ONLY valid JSON in exactly this shape:
{"winner": "A" | "B", "reason": "one sentence"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise evaluation on benchmark_results.")
    parser.add_argument("--limit-prompts", type=int, default=None, help="Optional prompt_hash limit.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top models per prompt to compare.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel pair workers.")
    parser.add_argument(
        "--judges",
        nargs="+",
        default=list(DEFAULT_JUDGES),
        help=f"Judge model short ids. Default: {', '.join(DEFAULT_JUDGES)}",
    )
    return parser.parse_args()


def clean_json(text: str) -> str:
    return (
        (text or "")
        .strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def extract_pairwise_payload(text: str) -> dict[str, Any]:
    cleaned = clean_json(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse pairwise JSON: {cleaned[:300]}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Pairwise evaluator output was not a JSON object")
    return payload


def build_prompt(prompt: str, response_a: str, response_b: str, model_a: str, model_b: str) -> str:
    return (
        f"{PAIRWISE_SYSTEM}\n\n"
        f"Prompt given to both models:\n{prompt.strip()}\n\n"
        f"Response A ({model_a}):\n{response_a.strip()}\n\n"
        f"Response B ({model_b}):\n{response_b.strip()}\n"
    )


def judge_pair(judge_name: str, prompt: str, response_a: str, response_b: str, model_a: str, model_b: str) -> dict:
    model = EVALUATOR_MODELS[judge_name]
    raw = call_evaluator(model, build_prompt(prompt, response_a, response_b, model_a, model_b))
    payload = extract_pairwise_payload(raw)
    winner = str(payload.get("winner", "")).strip().upper()
    if winner not in {"A", "B"}:
        raise ValueError(f"Invalid pairwise winner from {judge_name}: {winner}")
    return {
        "judge_model": judge_name,
        "winner": winner,
        "reason": str(payload.get("reason", "")).strip()[:500],
    }


def fetch_benchmark_rows(supabase, limit_prompts: int | None) -> list[dict]:
    page_size = 1000
    start = 0
    rows: list[dict] = []

    while True:
        batch = (
            supabase.table("benchmark_results")
            .select(
                "id,prompt_hash,prompt,use_case,prompt_complexity,model_id,response,avg_accuracy_score"
            )
            .not_.is_("prompt_hash", "null")
            .not_.is_("response", "null")
            .not_.is_("avg_accuracy_score", "null")
            .range(start, start + page_size - 1)
            .execute()
            .data
            or []
        )
        if not batch:
            break
        rows.extend(batch)
        start += page_size
        if limit_prompts and len({str(r.get("prompt_hash")) for r in rows}) >= limit_prompts:
            break

    if not limit_prompts:
        return rows

    allowed_hashes: set[str] = set()
    filtered: list[dict] = []
    for row in rows:
        prompt_hash = str(row.get("prompt_hash", "")).strip()
        if not prompt_hash:
            continue
        if prompt_hash in allowed_hashes or len(allowed_hashes) < limit_prompts:
            allowed_hashes.add(prompt_hash)
            filtered.append(row)
    return filtered


def select_candidate_rows(group: list[dict], top_k: int, seed: int) -> list[dict]:
    """
    Avoid relying entirely on avg_accuracy_score, which is the legacy noisy signal.

    Strategy:
    - keep 1 "anchor" model with the highest avg_accuracy_score so we retain a strong baseline
    - sample the remaining candidates randomly from the rest of the group
    """
    if len(group) <= top_k:
        return list(group)

    ordered = sorted(
        group,
        key=lambda item: float(item.get("avg_accuracy_score") or 0.0),
        reverse=True,
    )
    anchor = ordered[0]
    remainder = ordered[1:]
    rng = random.Random(seed)
    sampled = rng.sample(remainder, k=min(top_k - 1, len(remainder)))
    selected = [anchor, *sampled]
    return sorted(
        selected,
        key=lambda item: str(item.get("model_id", "")),
    )


def group_top_models(rows: list[dict], top_k: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        prompt_hash = str(row.get("prompt_hash", "")).strip()
        if prompt_hash:
            grouped.setdefault(prompt_hash, []).append(row)

    prompt_groups: list[dict] = []
    for group_index, (prompt_hash, group) in enumerate(grouped.items()):
        selected = select_candidate_rows(group, top_k=top_k, seed=group_index)
        if len(selected) < 2:
            continue
        sample = selected[0]
        prompt_groups.append(
            {
                "prompt_hash": prompt_hash,
                "prompt": str(sample.get("prompt", "")),
                "use_case": str(sample.get("use_case", "")),
                "complexity": str(sample.get("prompt_complexity", "")),
                "rows": selected,
            }
        )
    return prompt_groups


def upsert_pairwise_result(supabase, payload: dict[str, Any]) -> None:
    existing = (
        supabase.table("pairwise_results")
        .select("id")
        .eq("prompt_hash", payload["prompt_hash"])
        .eq("model_a", payload["model_a"])
        .eq("model_b", payload["model_b"])
        .limit(1)
        .execute()
        .data
        or []
    )
    if existing:
        supabase.table("pairwise_results").update(payload).eq("id", existing[0]["id"]).execute()
    else:
        supabase.table("pairwise_results").insert(payload).execute()


def evaluate_pair(prompt_group: dict, row_a: dict, row_b: dict, judges: list[str], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    swap = rng.choice([False, True])
    left = row_b if swap else row_a
    right = row_a if swap else row_b

    judge_results = [
        judge_pair(
            judge_name=judge_name,
            prompt=prompt_group["prompt"],
            response_a=str(left.get("response", "")),
            response_b=str(right.get("response", "")),
            model_a=str(left.get("model_id", "")),
            model_b=str(right.get("model_id", "")),
        )
        for judge_name in judges
    ]

    translated_winners: list[str] = []
    reasons: list[str] = []
    for result in judge_results:
        winner_model = str(left.get("model_id")) if result["winner"] == "A" else str(right.get("model_id"))
        translated_winners.append(winner_model)
        if result["reason"]:
            reasons.append(f"{result['judge_model']}: {result['reason']}")

    winner_counts: dict[str, int] = {}
    for winner_model in translated_winners:
        winner_counts[winner_model] = winner_counts.get(winner_model, 0) + 1

    sorted_counts = sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))
    if len(sorted_counts) == 1 or (
        len(sorted_counts) > 1 and sorted_counts[0][1] > sorted_counts[1][1]
    ):
        winner_model = sorted_counts[0][0]
        loser_model = (
            str(row_b.get("model_id"))
            if winner_model == str(row_a.get("model_id"))
            else str(row_a.get("model_id"))
        )
        winner = "A" if winner_model == str(row_a.get("model_id")) else "B"
    else:
        winner_model = "TIE"
        loser_model = ""
        winner = "TIE"

    if winner not in VALID_WINNERS:
        raise ValueError(f"Unexpected winner value: {winner}")

    return {
        "prompt_hash": prompt_group["prompt_hash"],
        "use_case": prompt_group["use_case"],
        "complexity": prompt_group["complexity"],
        "model_a": str(row_a.get("model_id")),
        "model_b": str(row_b.get("model_id")),
        "response_a": str(row_a.get("response", ""))[:2000],
        "response_b": str(row_b.get("response", ""))[:2000],
        "winner": winner,
        "winner_model": winner_model,
        "loser_model": loser_model,
        "judge_model": ",".join(judges),
        "reason": " | ".join(reasons)[:1000],
    }


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")
    args = parse_args()
    judges = [name.strip().lower() for name in args.judges]

    missing = [name for name in judges if name not in EVALUATOR_MODELS]
    if missing:
        raise SystemExit(f"Unknown judge(s): {', '.join(missing)}")

    supabase = get_supabase_client()
    rows = fetch_benchmark_rows(supabase, limit_prompts=args.limit_prompts)
    prompt_groups = group_top_models(rows, top_k=max(2, args.top_k))

    print(f"Loaded {len(rows)} benchmark rows across {len(prompt_groups)} prompt groups.")
    total_pairs = sum(len(list(itertools.combinations(group["rows"], 2))) for group in prompt_groups)
    print(f"Evaluating {total_pairs} model pairs with judges: {', '.join(judges)}")

    completed = 0
    failures = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        for group_index, prompt_group in enumerate(prompt_groups):
            for pair_index, (row_a, row_b) in enumerate(itertools.combinations(prompt_group["rows"], 2)):
                seed = group_index * 1000 + pair_index
                future = executor.submit(evaluate_pair, prompt_group, row_a, row_b, judges, seed)
                futures[future] = (prompt_group["prompt_hash"], row_a["model_id"], row_b["model_id"])

        for future in as_completed(futures):
            completed += 1
            prompt_hash, model_a, model_b = futures[future]
            try:
                payload = future.result()
                upsert_pairwise_result(supabase, payload)
                print(
                    f"[{completed}/{len(futures)}] prompt={prompt_hash[:8]} "
                    f"{model_a} vs {model_b} -> {payload['winner_model']}"
                )
            except Exception as exc:
                failures += 1
                print(f"[{completed}/{len(futures)}] FAILED prompt={prompt_hash[:8]} {model_a} vs {model_b}: {exc}")
                time.sleep(0.2)

    print(f"Finished pairwise evaluation. Failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
