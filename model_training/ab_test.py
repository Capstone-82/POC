"""
ab_test.py — LLM Router A/B Test Runner
========================================
Group A (control)   → uses model from CSV (or DEFAULT_MODEL)
Group B (treatment) → asks the router for the best model, then uses it

Usage:
    python ab_test.py                        # uses prompts.csv
    python ab_test.py --input my_prompts.csv
    python ab_test.py --concurrency 5

Output:
    experiment_results.csv   (local CSV backup)
    Supabase: ab_test_results table

Create the Supabase table once:
    CREATE TABLE IF NOT EXISTS ab_test_results (
        id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        run_id            TEXT,
        prompt            TEXT,
        use_case          TEXT,
        group_label       TEXT,
        model_used        TEXT,
        recommended_model TEXT,
        eval_llama4_score FLOAT,
        eval_mistral_score FLOAT,
        eval_nova_score   FLOAT,
        avg_accuracy_score FLOAT,
        score_stdev       FLOAT,
        latency_ms        INTEGER,
        cost              FLOAT,
        created_at        TIMESTAMPTZ DEFAULT now()
    );
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv

# Load .env from backend directory
load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama3-3-70b"          # fallback when CSV model column is empty
BASE_URL      = "http://127.0.0.1:8000"

RECOMMENDER_API_URL = f"{BASE_URL}/api/inference/recommend"
INFERENCE_API_URL   = f"{BASE_URL}/api/inference/run"
EVALUATOR_API_URL   = f"{BASE_URL}/api/inference/evaluate"

# All Bedrock evaluators — no Gemini (Vertex billing disabled)
EVALUATOR_MODELS = [
    "llama4-maverick",
    "mistral-large",
    "nova-premier",
]

# Gemini models cannot be called (Vertex billing disabled) — skip if recommended
GEMINI_SHORT_IDS = {
    "gemini-2-0-flash", "gemini-2-0-flash-lite",
    "gemini-2-5-flash", "gemini-2-5-pro",
    "gemini-3-1-pro",   "gemini-3-1-flash-lite",
}

MAX_RETRIES   = 2
RETRY_DELAY_S = 1.5
RANDOM_SEED   = 42

OUTPUT_COLUMNS = [
    "prompt", "use_case", "group", "csv_model", "model_used", "recommended_model",
    "recommendation_mode",
    "eval_llama4_score", "eval_mistral_score", "eval_nova_score",
    "avg_accuracy_score", "score_stdev", "latency_ms", "cost",
]


# ─── Group assignment ─────────────────────────────────────────────────────────

def assign_group(index: int, rng: random.Random) -> str:
    """Return 'A' or 'B' with equal probability (deterministic given seed)."""
    return "A" if rng.random() < 0.5 else "B"


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

async def _post(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    label: str = "",
) -> Optional[dict]:
    """POST with up to MAX_RETRIES retries. Returns parsed JSON or None."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 200:
                    return await resp.json()
                text = await resp.text()
                print(f"  [WARN] {label} HTTP {resp.status}: {text[:120]}")
        except Exception as exc:
            print(f"  [WARN] {label} attempt {attempt + 1} failed: {exc}")

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))

    print(f"  [ERROR] {label} gave up after {MAX_RETRIES + 1} attempts")
    return None



async def run_model(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    use_case: str,
) -> Optional[dict]:
    """
    Call POST /api/inference/run.
    Returns { response, latency_ms, cost } or None on failure.
    """
    payload = {"prompt": prompt, "model": model, "use_case": use_case}
    result  = await _post(session, INFERENCE_API_URL, payload, label=f"run/{model}")
    return result


async def evaluate_with_multiple_judges(
    session: aiohttp.ClientSession,
    prompt: str,
    response: str,
    use_case: str,
) -> dict:
    """
    Call POST /api/inference/evaluate for each evaluator in EVALUATOR_MODELS
    concurrently.  Fails gracefully if one evaluator errors.

    Returns:
        {
            "eval_llama4_score":  float | None,
            "eval_mistral_score": float | None,
            "eval_nova_score":    float | None,
            "avg_accuracy_score": float | None,
            "score_stdev":        float | None,
            "eval_count":         int,
        }
    """
    async def _one_eval(evaluator_model: str) -> Optional[float]:
        payload = {
            "prompt":          prompt,
            "response":        response,
            "use_case":        use_case,
            "evaluator_model": evaluator_model,
        }
        result = await _post(session, EVALUATOR_API_URL, payload, label=f"eval/{evaluator_model}")
        if result is None:
            return None
        if result.get("error"):
            print(f"  [WARN] Evaluator {evaluator_model} error: {result['error']}")
            return None
        score = result.get("score")
        return float(score) if score is not None else None

    scores_raw = await asyncio.gather(*[_one_eval(m) for m in EVALUATOR_MODELS])
    llama4_score, mistral_score, nova_score = scores_raw

    valid = [s for s in scores_raw if s is not None]
    avg   = round(statistics.mean(valid), 2)          if valid else None
    stdev = round(statistics.stdev(valid), 2)         if len(valid) >= 2 else None

    return {
        "eval_llama4_score":  llama4_score,
        "eval_mistral_score": mistral_score,
        "eval_nova_score":    nova_score,
        "avg_accuracy_score": avg,
        "score_stdev":        stdev,
        "eval_count":         len(valid),
    }


# ─── Per-row processor ────────────────────────────────────────────────────────

async def process_row(
    session: aiohttp.ClientSession,
    index: int,
    total: int,
    row: dict,
    group: str,
) -> dict:
    """
    Run the full pipeline for a single CSV row in Group A or B.
    Returns a result dict matching OUTPUT_COLUMNS.
    """
    prompt       = (row.get("prompt") or "").strip()
    use_case     = (row.get("use_case") or "text-generation").strip()
    csv_model    = (row.get("model") or "").strip() or DEFAULT_MODEL

    recommended_model   = None
    recommendation_mode = None
    model_used          = csv_model  # default for group A and on failure

    # ── Group B: get recommendation ───────────────────────────────────────────
    if group == "B":
        rec_result = await _post(
            session, RECOMMENDER_API_URL,
            {"prompt": prompt, "use_case": use_case, "current_model": csv_model},
            label="recommend",
        )
        if rec_result:
            rec = rec_result.get("recommended_model") or rec_result.get("final_suggestion_model")
            recommendation_mode = rec_result.get("recommendation_mode")

            # If the top recommendation is Gemini, walk top_candidates for a Bedrock one
            if rec and rec in GEMINI_SHORT_IDS:
                candidates = rec_result.get("top_candidates", [])
                for candidate in candidates:
                    alt = candidate.get("model_id", "")
                    if alt and alt not in GEMINI_SHORT_IDS:
                        print(f"  [INFO] Gemini '{rec}' skipped → using '{alt}'")
                        rec = alt
                        break
                else:
                    print(f"  [WARN] All KNN candidates are Gemini — falling back to csv_model")
                    rec = None

            if rec:
                recommended_model = rec
                model_used        = rec
            else:
                print(f"  [WARN] Row {index}: recommendation failed, using {csv_model}")
        else:
            print(f"  [WARN] Row {index}: recommend API failed, using {csv_model}")

    # ── Run inference ─────────────────────────────────────────────────────────
    run_result  = await run_model(session, prompt, model_used, use_case)
    response    = ""
    latency_ms  = None
    cost        = None

    if run_result:
        response   = run_result.get("response", "")
        latency_ms = run_result.get("latency_ms")
        cost       = run_result.get("cost")
    else:
        print(f"  [WARN] Row {index}: inference failed for model '{model_used}'")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_result = {"eval_llama4_score": None, "eval_mistral_score": None,
                   "eval_nova_score": None, "avg_accuracy_score": None,
                   "score_stdev": None, "eval_count": 0}

    if response:
        eval_result = await evaluate_with_multiple_judges(
            session, prompt, response, use_case
        )
    else:
        print(f"  [WARN] Row {index}: no response to evaluate")

    print(f"  [{group}] {index}/{total} | model={model_used} | "
          f"avg_acc={eval_result.get('avg_accuracy_score')} | "
          f"lat={latency_ms}ms | cost=${cost}")

    return {
        "prompt":              prompt,
        "use_case":            use_case,
        "group":               group,
        "csv_model":           csv_model,
        "model_used":          model_used,
        "recommended_model":   recommended_model or "",
        "recommendation_mode": recommendation_mode or "",
        "eval_llama4_score":   eval_result.get("eval_llama4_score"),
        "eval_mistral_score":  eval_result.get("eval_mistral_score"),
        "eval_nova_score":     eval_result.get("eval_nova_score"),
        "avg_accuracy_score":  eval_result.get("avg_accuracy_score"),
        "score_stdev":         eval_result.get("score_stdev"),
        "latency_ms":          latency_ms,
        "cost":                cost,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(input_path: Path, concurrency: int) -> None:
    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: '{input_path}' not found.")
        sys.exit(1)

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("ERROR: CSV is empty.")
        sys.exit(1)

    total = len(rows)
    print(f"\n{'='*60}")
    print(f"  LLM Router A/B Test")
    print(f"  Input:       {input_path}")
    print(f"  Rows:        {total}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Seed:        {RANDOM_SEED}")
    print(f"{'='*60}\n")

    # ── Assign groups ─────────────────────────────────────────────────────────
    rng    = random.Random(RANDOM_SEED)
    groups = [assign_group(i, rng) for i in range(total)]
    a_count = groups.count("A")
    b_count = groups.count("B")
    print(f"Group A (control):   {a_count} rows")
    print(f"Group B (treatment): {b_count} rows\n")

    # ── Run experiment ────────────────────────────────────────────────────────
    results   = []
    semaphore = asyncio.Semaphore(concurrency)
    start_ts  = time.time()

    async with aiohttp.ClientSession() as session:
        async def bounded_process(idx: int, row: dict, group: str) -> dict:
            async with semaphore:
                return await process_row(session, idx + 1, total, row, group)

        tasks = [
            bounded_process(i, row, groups[i])
            for i, row in enumerate(rows)
        ]
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_ts
    print(f"\nFinished {total} rows in {elapsed:.1f}s ({elapsed/total:.1f}s/row avg)\n")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    output_path = input_path.parent / "experiment_results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ CSV saved: {output_path}")

    # ── Save to Supabase ──────────────────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]
    _save_to_supabase(results, run_id)

    # ── Summary stats ─────────────────────────────────────────────────────────
    _print_summary(results)


def _save_to_supabase(results: list[dict], run_id: str) -> None:
    """
    Save experiment results to Supabase ab_test_results table.
    Non-fatal: prints a warning and skips if the table doesn't exist yet
    or if SUPABASE_URL / SUPABASE_KEY are not set.

    Create the table once in Supabase SQL Editor (printed in module docstring).
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        print("[SUPABASE] SUPABASE_URL/SUPABASE_KEY not set — skipping DB save.")
        return

    try:
        from supabase import create_client
        sb = create_client(url, key)

        records = [
            {
                "run_id":             run_id,
                "prompt":             r.get("prompt", "")[:2000],
                "use_case":           r.get("use_case"),
                "group_label":        r.get("group"),
                "model_used":         r.get("model_used"),
                "recommended_model":  r.get("recommended_model") or None,
                "eval_llama4_score":  r.get("eval_llama4_score"),
                "eval_mistral_score": r.get("eval_mistral_score"),
                "eval_nova_score":    r.get("eval_nova_score"),
                "avg_accuracy_score": r.get("avg_accuracy_score"),
                "score_stdev":        r.get("score_stdev"),
                "latency_ms":         r.get("latency_ms"),
                "cost":               r.get("cost"),
            }
            for r in results
        ]

        # Insert in batches of 50 to stay within Supabase payload limits
        batch_size = 50
        saved = 0
        for i in range(0, len(records), batch_size):
            sb.table("ab_test_results").insert(records[i : i + batch_size]).execute()
            saved += min(batch_size, len(records) - i)

        print(f"✓ Supabase: saved {saved} rows to ab_test_results (run_id={run_id})")

    except Exception as exc:
        err = str(exc)
        if "PGRST205" in err or "ab_test_results" in err:
            print(
                f"[SUPABASE] Table 'ab_test_results' not found.\n"
                f"  → Create it in Supabase SQL Editor (SQL is in the script docstring).\n"
                f"  → Results are saved locally in experiment_results.csv"
            )
        else:
            print(f"[SUPABASE ERROR] {exc}")


def _print_summary(results: list[dict]) -> None:
    """Print a quick A vs B comparison table."""
    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 3) if vals else None

    a_rows = [r for r in results if r["group"] == "A"]
    b_rows = [r for r in results if r["group"] == "B"]

    a_acc  = safe_mean([r["avg_accuracy_score"] for r in a_rows])
    b_acc  = safe_mean([r["avg_accuracy_score"] for r in b_rows])
    a_lat  = safe_mean([r["latency_ms"] for r in a_rows])
    b_lat  = safe_mean([r["latency_ms"] for r in b_rows])
    a_cost = safe_mean([r["cost"] for r in a_rows])
    b_cost = safe_mean([r["cost"] for r in b_rows])

    # Switch rate = fraction of B rows where the router picked a DIFFERENT model than the CSV baseline
    b_switched = sum(
        1 for r in b_rows
        if r.get("recommended_model") and r["recommended_model"] != r.get("csv_model", "")
    )

    print("=" * 60)
    print(f"  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Group A (control)':>18} {'Group B (KNN)':>15}")
    print(f"  {'-'*58}")
    print(f"  {'Rows':<25} {len(a_rows):>18} {len(b_rows):>15}")
    print(f"  {'Avg accuracy':<25} {str(a_acc):>18} {str(b_acc):>15}")
    print(f"  {'Avg latency (ms)':<25} {str(a_lat):>18} {str(b_lat):>15}")
    print(f"  {'Avg cost ($)':<25} {str(a_cost):>18} {str(b_cost):>15}")

    if a_acc is not None and b_acc is not None:
        delta = round(b_acc - a_acc, 3)
        print(f"\n  Accuracy delta (B - A): {'+' if delta >= 0 else ''}{delta}")

    if b_rows:
        switch_pct = round(b_switched / len(b_rows) * 100, 1)
        print(f"  B switched from CSV model:   {b_switched}/{len(b_rows)} ({switch_pct}%)")

    # Group B: recommendation modes
    mode_dist = {}
    for r in b_rows:
        m = r.get("recommendation_mode") or "unknown"
        mode_dist[m] = mode_dist.get(m, 0) + 1
    if mode_dist:
        print(f"\n  Group B recommendation modes:")
        for m, cnt in sorted(mode_dist.items(), key=lambda x: -x[1]):
            print(f"    {m:<30} {cnt} rows ({cnt/max(len(b_rows),1)*100:.1f}%)")

    model_dist_b = {}
    for r in b_rows:
        m = r["model_used"]
        model_dist_b[m] = model_dist_b.get(m, 0) + 1

    print(f"\n  Group B model distribution:")
    for m, cnt in sorted(model_dist_b.items(), key=lambda x: -x[1]):
        print(f"    {m:<30} {cnt} rows ({cnt/max(len(b_rows),1)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Router A/B Test")
    parser.add_argument("--input",       type=Path, default=Path("prompts.csv"),
                        help="Path to input CSV (default: prompts.csv)")
    parser.add_argument("--concurrency", type=int,  default=3,
                        help="Max concurrent rows to process (default: 3)")
    args = parser.parse_args()

    asyncio.run(main(input_path=args.input, concurrency=args.concurrency))
