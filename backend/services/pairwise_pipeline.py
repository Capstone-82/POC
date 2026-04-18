from __future__ import annotations

import itertools
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from services.bedrock import BEDROCK_MODELS, _build_body, _extract_text, bedrock as bedrock_client
from services.supabase_client import supabase

DEFAULT_JUDGES = ("llama4-maverick", "mistral-large", "nova-premier")
VALID_WINNERS = {"A", "B", "TIE"}
MIN_DECISIVE_MATCHES = 5
FULL_CONFIDENCE_MATCHES = 10

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


def _clean_json(text: str) -> str:
    return (
        (text or "")
        .strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def _extract_pairwise_payload(text: str) -> dict[str, Any]:
    cleaned = _clean_json(text)
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


def _find_bedrock_model(short_id: str) -> dict[str, Any]:
    match = next((model for model in BEDROCK_MODELS if model["short_id"] == short_id), None)
    if not match:
        raise ValueError(f"Evaluator '{short_id}' not found in Bedrock model registry")
    return match


def _build_pairwise_prompt(prompt: str, response_a: str, response_b: str, model_a: str, model_b: str) -> str:
    return (
        f"{PAIRWISE_SYSTEM}\n\n"
        f"Prompt given to both models:\n{prompt.strip()}\n\n"
        f"Response A ({model_a}):\n{response_a.strip()}\n\n"
        f"Response B ({model_b}):\n{response_b.strip()}\n"
    )


def _judge_pair(
    judge_name: str,
    prompt: str,
    response_a: str,
    response_b: str,
    model_a: str,
    model_b: str,
) -> dict[str, str]:
    judge = _find_bedrock_model(judge_name)
    raw = bedrock_client.invoke_model(
        modelId=judge["model_id"],
        body=_build_body(
            judge["fmt"],
            _build_pairwise_prompt(prompt, response_a, response_b, model_a, model_b),
        ),
        contentType="application/json",
        accept="application/json",
    )
    body_json = json.loads(raw["body"].read())
    text = _extract_text(judge["fmt"], body_json)
    payload = _extract_pairwise_payload(text)
    winner = str(payload.get("winner", "")).strip().upper()
    if winner not in {"A", "B"}:
        raise ValueError(f"Invalid pairwise winner from {judge_name}: {winner}")
    return {
        "judge_model": judge_name,
        "winner": winner,
        "reason": str(payload.get("reason", "")).strip()[:500],
    }


def _fetch_benchmark_rows_for_hashes(prompt_hashes: list[str]) -> list[dict[str, Any]]:
    if not prompt_hashes:
        return []

    rows: list[dict[str, Any]] = []
    for prompt_hash in prompt_hashes:
        batch = (
            supabase.table("benchmark_results")
            .select("prompt_hash,prompt,use_case,prompt_complexity,model_id,response")
            .eq("prompt_hash", prompt_hash)
            .not_.is_("response", "null")
            .execute()
            .data
            or []
        )
        rows.extend(batch)
    return rows


def _upsert_pairwise_result(payload: dict[str, Any]) -> None:
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


def _evaluate_pair(prompt_group: dict[str, Any], row_a: dict[str, Any], row_b: dict[str, Any], judges: tuple[str, ...], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    swap = rng.choice([False, True])
    left = row_b if swap else row_a
    right = row_a if swap else row_b

    judge_results = [
        _judge_pair(
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
    if len(sorted_counts) == 1 or (len(sorted_counts) > 1 and sorted_counts[0][1] > sorted_counts[1][1]):
        winner_model = sorted_counts[0][0]
        loser_model = str(row_b.get("model_id")) if winner_model == str(row_a.get("model_id")) else str(row_a.get("model_id"))
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


def _update_bucket(bucket: dict[str, Any], model_id: str, winner: str, winner_model: str, judge_model: str) -> None:
    bucket["total_matches"] += 1
    bucket["total_participations"] += 1

    if judge_model:
        for judge in judge_model.split(","):
            cleaned = judge.strip()
            if cleaned:
                bucket["judges"].add(cleaned)

    if winner == "TIE":
        bucket["ties"] += 1
    elif winner_model == model_id:
        bucket["wins"] += 1
    else:
        bucket["losses"] += 1


def run_pairwise_for_prompt_hashes(
    prompt_hashes: list[str],
    judges: tuple[str, ...] = DEFAULT_JUDGES,
    max_workers: int = 4,
) -> dict[str, Any]:
    rows = _fetch_benchmark_rows_for_hashes(prompt_hashes)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        prompt_hash = str(row.get("prompt_hash", "")).strip()
        if prompt_hash:
            grouped.setdefault(prompt_hash, []).append(row)

    prompt_groups: list[dict[str, Any]] = []
    for prompt_hash, group in grouped.items():
        if len(group) < 2:
            continue
        sample = group[0]
        prompt_groups.append(
            {
                "prompt_hash": prompt_hash,
                "prompt": str(sample.get("prompt", "")),
                "use_case": str(sample.get("use_case", "")),
                "complexity": str(sample.get("prompt_complexity", "")),
                "rows": sorted(group, key=lambda item: str(item.get("model_id", ""))),
            }
        )

    futures = {}
    completed = 0
    failures = 0
    pair_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        for group_index, prompt_group in enumerate(prompt_groups):
            for pair_index, (row_a, row_b) in enumerate(itertools.combinations(prompt_group["rows"], 2)):
                seed = group_index * 1000 + pair_index
                future = executor.submit(_evaluate_pair, prompt_group, row_a, row_b, judges, seed)
                futures[future] = (prompt_group["prompt_hash"], row_a["model_id"], row_b["model_id"])

        for future in as_completed(futures):
            completed += 1
            try:
                payload = future.result()
                _upsert_pairwise_result(payload)
                pair_results.append(
                    {
                        "prompt_hash": payload["prompt_hash"],
                        "use_case": payload["use_case"],
                        "complexity": payload["complexity"],
                        "model_a": payload["model_a"],
                        "model_b": payload["model_b"],
                        "winner_model": payload["winner_model"],
                        "judge_model": payload["judge_model"],
                    }
                )
            except Exception as exc:
                failures += 1
                prompt_hash, model_a, model_b = futures[future]
                print(f"[PAIRWISE ERROR] prompt={prompt_hash[:8]} {model_a} vs {model_b}: {exc}")

    return {
        "prompt_groups": len(prompt_groups),
        "pairs_total": len(futures),
        "pairs_completed": completed - failures,
        "pairs_failed": failures,
        "pair_results": pair_results,
    }


def refresh_model_win_rates_for_use_cases(use_cases: list[str]) -> int:
    page_size = 1000
    start = 0
    rows: list[dict[str, Any]] = []
    target_use_cases = {use_case.strip() for use_case in use_cases if use_case and use_case.strip()}

    while True:
        batch = (
            supabase.table("pairwise_results")
            .select("use_case,complexity,model_a,model_b,winner,winner_model,judge_model")
            .range(start, start + page_size - 1)
            .execute()
            .data
            or []
        )
        if not batch:
            break
        if target_use_cases:
            batch = [row for row in batch if str(row.get("use_case", "")).strip() in target_use_cases]
        rows.extend(batch)
        start += page_size

    aggregate: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_matches": 0,
            "total_participations": 0,
            "judges": set(),
        }
    )

    for row in rows:
        use_case = str(row.get("use_case", "")).strip()
        complexity = str(row.get("complexity", "")).strip().lower() or "all"
        winner = str(row.get("winner", "")).strip().upper()
        winner_model = str(row.get("winner_model", "")).strip()
        model_a = str(row.get("model_a", "")).strip()
        model_b = str(row.get("model_b", "")).strip()
        judge_model = str(row.get("judge_model", "")).strip()

        for model_id in (model_a, model_b):
            if not model_id or not use_case:
                continue
            bucket = aggregate[(model_id, use_case, complexity)]
            _update_bucket(bucket, model_id, winner, winner_model, judge_model)
            all_bucket = aggregate[(model_id, use_case, "all")]
            _update_bucket(all_bucket, model_id, winner, winner_model, judge_model)

    upserts = []
    for (model_id, use_case, complexity), bucket in aggregate.items():
        decisive_matches = bucket["wins"] + bucket["losses"]
        confidence = min(1.0, decisive_matches / FULL_CONFIDENCE_MATCHES) if decisive_matches else 0.0
        tie_rate = (bucket["ties"] / bucket["total_matches"]) if bucket["total_matches"] else 0.0
        win_rate = None
        if decisive_matches >= MIN_DECISIVE_MATCHES:
            win_rate = bucket["wins"] / decisive_matches

        upserts.append(
            {
                "model_id": model_id,
                "use_case": use_case,
                "complexity": complexity,
                "win_rate": None if win_rate is None else round(win_rate, 4),
                "total_matches": bucket["total_matches"],
                "total_participations": bucket["total_participations"],
                "decisive_matches": decisive_matches,
                "wins": bucket["wins"],
                "losses": bucket["losses"],
                "ties": bucket["ties"],
                "tie_rate": round(tie_rate, 4),
                "judge_count": len(bucket["judges"]),
                "confidence": round(confidence, 4),
            }
        )

    if upserts:
        supabase.table("model_win_rates").upsert(
            upserts,
            on_conflict="model_id,use_case,complexity",
        ).execute()
    return len(upserts)
