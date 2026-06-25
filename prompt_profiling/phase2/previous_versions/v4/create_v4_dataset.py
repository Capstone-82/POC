from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PHASE2_DIR = THIS_DIR.parent
BASE_PATH = PHASE2_DIR / "dataset_for_team" / "prompt_classifier_phase1_phase2_merged_cleaned.csv"
OUTPUT_PATH = THIS_DIR / "prompt_classifier_phase2_v4_dataset.csv"
AUG_ONLY_PATH = THIS_DIR / "prompt_classifier_phase2_v4_augmented_rows_only.csv"
AUDIT_PATH = THIS_DIR / "V4_DATASET_AUDIT.md"

SCORE_COLUMNS = ["d1", "d2", "d3", "d4", "d5"]


def complexity_score(d1: float, d2: float, d3: float, d4: float, d5: float) -> float:
    return round(d1 * 0.35 + d2 * 0.20 + d3 * 0.20 + d4 * 0.15 + d5 * 0.10, 4)


def tier_from_score(score: float) -> str:
    if score < 0.40:
        return "T1"
    if score < 0.70:
        return "T2"
    return "T3"


def complexity_from_tier(tier: str) -> str:
    return {"T1": "low", "T2": "medium", "T3": "high"}[tier]


def signal_cell(signals: list[str]) -> str:
    return json.dumps(signals, ensure_ascii=True)


def normalize_research_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    normalized = []
    changed = 0
    for _, row in df.iterrows():
        try:
            signals = json.loads(row["research_signals"])
        except Exception:
            signals = []

        if row["d4"] == 0 and signals:
            signals = []
            changed += 1
        if row["d4"] > 0 and not signals:
            signals = ["external_research"]
            changed += 1
        normalized.append(signal_cell(signals))
    df["research_signals"] = normalized
    return df, changed


def add_audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    expected_intent = df["d1"].map(
        lambda x: "FACTUAL"
        if x <= 0.25
        else ("ANALYTICAL" if x == 0.50 else ("SYNTHETIC" if x == 0.75 else "STRATEGIC"))
    )
    df["intent_d1_mismatch_flag"] = df["intent"] != expected_intent
    df["boundary_t12_flag"] = df["complexity_score"].between(0.35, 0.45, inclusive="both")
    df["boundary_t23_flag"] = df["complexity_score"].between(0.65, 0.75, inclusive="left")
    df["original_row_flag"] = df["source"].isin(["phase1", "phase2"])
    return df


def make_row(
    row_id: str,
    prompt: str,
    intent: str,
    task_type: str,
    reasoning_chain_detected: bool,
    d1: float,
    d2: float,
    d3: float,
    d4: float,
    d5: float,
    research_signals: list[str],
    prompt_type: str,
    generation_group: str,
) -> dict:
    score = complexity_score(d1, d2, d3, d4, d5)
    tier = tier_from_score(score)
    if d4 == 0:
        research_signals = []
    elif not research_signals:
        research_signals = ["external_research"]

    return {
        "id": row_id,
        "prompt": prompt,
        "intent": intent,
        "task_type": task_type,
        "reasoning_chain_detected": reasoning_chain_detected,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "d5": d5,
        "complexity_score": score,
        "tier": tier,
        "complexity": complexity_from_tier(tier),
        "original_complexity": pd.NA,
        "research_signals": signal_cell(research_signals),
        "confidence": pd.NA,
        "low_confidence_flag": False,
        "task_description": generation_group,
        "expected_answer": pd.NA,
        "prompting_techniques": pd.NA,
        "prompt_type": prompt_type,
        "phrasing_style": pd.NA,
        "domain": pd.NA,
        "source": "aug_v4",
        "augmentation_group": generation_group,
    }


def build_boundary_rows() -> list[dict]:
    topics = [
        ("cloud waste triage", "AWS, idle databases, owner tags, budget alerts", ["finops", "cloud_infrastructure"]),
        ("vendor exception review", "SOC 2 gaps, compensating controls, renewal timing", ["security", "vendor_analysis"]),
        ("pipeline quality incident", "late events, dbt failures, dashboard freshness", ["data_engineering"]),
        ("HR model governance", "candidate scoring, bias checks, appeal process", ["hr_tech", "ai_governance"]),
        ("release reliability", "rollback failures, SRE escalation, change windows", ["devops"]),
        ("market launch decision", "competitor pricing, channel risk, customer segment evidence", ["market_research", "competitive_analysis"]),
        ("audit workflow backlog", "SOX evidence, control owners, stale exceptions", ["regulatory_compliance"]),
        ("customer GenAI controls", "retention settings, prompt logging, hallucination review", ["ai_governance", "security"]),
        ("supply delay planning", "supplier concentration, inventory policy, logistics options", ["supply_chain"]),
        ("API migration readiness", "MuleSoft mappings, SAP updates, retry handling", ["system_integration"]),
    ]
    rows = []
    idx = 1
    for topic, details, signals in topics:
        t2_prompt = (
            f"We have an internal context packet about {topic}: {details}. "
            "Analyze the situation using only the provided packet. Return a concise recommendation, assumptions, risks, and next steps."
        )
        rows.append(
            make_row(
                f"v4_boundary_t2_{idx:03d}",
                t2_prompt,
                "SYNTHETIC",
                "reasoning",
                True,
                0.75,
                0.75,
                0.75,
                0.25,
                0.50,
                [],
                "V4_BOUNDARY_T2",
                "boundary_t2",
            )
        )
        t3_prompt = (
            f"We have an internal context packet about {topic}: {details}. "
            "Analyze the situation, compare it with current external vendor guidance or market benchmarks, "
            "include regulatory implications where relevant, and produce a cross-functional rollout recommendation."
        )
        rows.append(
            make_row(
                f"v4_boundary_t3_{idx:03d}",
                t3_prompt,
                "SYNTHETIC",
                "reasoning",
                True,
                0.75,
                0.75,
                0.75,
                0.75,
                0.50,
                signals,
                "V4_BOUNDARY_T3",
                "boundary_t3",
            )
        )
        idx += 1
    return rows


def build_phase2_style_t3_rows() -> list[dict]:
    scenarios = [
        ("hospital GenAI triage", "HIPAA, clinical safety, audit trails, vendor retention", ["regulatory_compliance", "security", "ai_governance"]),
        ("global FinOps operating model", "AWS and Azure unit economics, forecasting, chargeback, executive KPIs", ["finops", "cloud_infrastructure"]),
        ("AI hiring governance", "EU AI Act, Workday predictions, bias testing, appeals, regional rollout", ["hr_tech", "ai_governance", "regulatory_compliance"]),
        ("data mesh migration", "Snowflake, Databricks, lineage, ownership, dashboard SLAs, executive adoption", ["data_engineering", "cloud_infrastructure"]),
        ("zero-trust board roadmap", "identity sprawl, audit findings, vendor risk, phased funding", ["security", "vendor_analysis"]),
        ("supply-chain risk platform", "supplier concentration, geopolitical exposure, inventory policy, control tower design", ["supply_chain", "market_research"]),
        ("GenAI vendor RFP", "model quality, legal terms, data residency, retention, evaluation scoring", ["vendor_analysis", "ai_governance", "security"]),
        ("multi-cloud resilience", "regional failover, compliance evidence, cost tradeoffs, operating ownership", ["cloud_infrastructure", "regulatory_compliance", "finops"]),
    ]
    rows = []
    score_patterns = [
        (1.0, 1.0, 1.0, 1.0, 0.75, "STRATEGIC"),
        (1.0, 0.75, 1.0, 0.75, 1.0, "STRATEGIC"),
        (0.75, 1.0, 1.0, 1.0, 0.75, "SYNTHETIC"),
        (1.0, 1.0, 0.75, 1.0, 1.0, "STRATEGIC"),
    ]
    idx = 1
    for scenario, details, signals in scenarios:
        for style in ["decision memo", "implementation package", "risk review", "executive briefing", "architecture recommendation"]:
            d1, d2, d3, d4, d5, intent = score_patterns[idx % len(score_patterns)]
            prompt = (
                f"Prepare a {style} for {scenario}. It must synthesize {details}. "
                "Include alternatives, quantified tradeoffs, stakeholder impacts, governance controls, rollout phases, and a final recommendation."
            )
            task_type = "generation" if style in ["implementation package", "executive briefing"] else "reasoning"
            rows.append(
                make_row(
                    f"v4_t3_phase2_style_{idx:03d}",
                    prompt,
                    intent,
                    task_type,
                    True,
                    d1,
                    d2,
                    d3,
                    d4,
                    d5,
                    signals,
                    "V4_PHASE2_STYLE_T3",
                    "phase2_style_t3",
                )
            )
            idx += 1
    return rows


def build_task_rows() -> list[dict]:
    rows = []
    contexts = ["cloud", "security", "finance", "customer support", "data governance", "HR", "DevOps", "vendor management"]
    specs = [
        ("classification", "Classify each item into predefined labels and provide a one-line rationale.", "FACTUAL", False, (0.25, 0.50, 0.25, 0.0, 0.25), "V4_TASK_CLASSIFICATION"),
        ("coding", "Write Python code with tests to validate the payload and return structured errors.", "ANALYTICAL", True, (0.50, 0.75, 0.75, 0.0, 0.50), "V4_TASK_CODING"),
        ("summarisation", "Summarize the packet into executive bullets, root causes, impact, and actions.", "ANALYTICAL", True, (0.50, 0.50, 0.75, 0.0, 0.75), "V4_TASK_SUMMARISATION"),
        ("formatting", "Convert messy notes into strict YAML with owners, dates, commands, rollback, and validation checks.", "ANALYTICAL", True, (0.50, 0.50, 0.75, 0.0, 0.75), "V4_TASK_FORMATTING"),
        ("sparql_generation", "Generate SPARQL queries for an ontology of systems, owners, controls, risks, and evidence.", "ANALYTICAL", True, (0.50, 1.0, 0.75, 0.0, 0.50), "V4_TASK_SPARQL"),
    ]
    idx = 1
    for task_type, instruction, intent, reasoning, scores, prompt_type in specs:
        for context in contexts:
            prompt = f"For a {context} workflow, {instruction} Keep the output compact but complete."
            rows.append(
                make_row(
                    f"v4_task_{task_type}_{idx:03d}",
                    prompt,
                    intent,
                    task_type,
                    reasoning,
                    *scores,
                    research_signals=[],
                    prompt_type=prompt_type,
                    generation_group=f"task_{task_type}",
                )
            )
            idx += 1
    return rows


def build_audit(base: pd.DataFrame, normalized: pd.DataFrame, aug: pd.DataFrame, merged: pd.DataFrame, signal_changes: int) -> str:
    lines = [
        "# V4 Dataset Audit",
        "",
        "## Scope",
        "",
        "This is a separate v4 working dataset. The original team handoff dataset is not modified.",
        "",
        "## Counts",
        "",
        f"- Base rows: `{len(base)}`",
        f"- Added rows: `{len(aug)}`",
        f"- Final rows: `{len(merged)}`",
        f"- Existing research signal cells normalized: `{signal_changes}`",
        "",
        "## Source Counts",
        "```text",
        merged["source"].value_counts().to_string(),
        "```",
        "",
        "## Tier Counts",
        "```text",
        merged["tier"].value_counts().sort_index().to_string(),
        "```",
        "",
        "## Task Type Counts",
        "```text",
        merged["task_type"].value_counts().to_string(),
        "```",
        "",
        "## Intent Counts",
        "```text",
        merged["intent"].value_counts().to_string(),
        "```",
        "",
        "## Validation",
        "",
        f"- Duplicate prompts: `{int(merged['prompt'].duplicated().sum())}`",
        f"- Tier formula mismatches: `{int((merged['complexity_score'].apply(tier_from_score) != merged['tier']).sum())}`",
        f"- Original rows retained: `{int(merged['original_row_flag'].sum())}`",
        f"- Augmentation share: `{len(aug) / len(merged):.2%}`",
        f"- T2/T3 boundary rows: `{int(merged['boundary_t23_flag'].sum())}`",
        "",
        "## V4 Evaluation Requirement",
        "",
        "Report metrics on all rows, original rows only, phase2 rows only, phase1 rows only, and aug_v4 rows separately.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_PATH)
    normalized, signal_changes = normalize_research_signals(base)

    aug_rows = build_boundary_rows() + build_phase2_style_t3_rows() + build_task_rows()
    aug = pd.DataFrame(aug_rows)

    # Align columns while preserving v4 metadata columns.
    normalized["augmentation_group"] = pd.NA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        merged = pd.concat([normalized, aug], ignore_index=True)
    merged = merged.drop_duplicates(subset=["prompt"], keep="first")

    merged["complexity_score"] = merged.apply(
        lambda r: complexity_score(r["d1"], r["d2"], r["d3"], r["d4"], r["d5"]), axis=1
    )
    merged["tier"] = merged["complexity_score"].apply(tier_from_score)
    merged["complexity"] = merged["tier"].apply(complexity_from_tier)
    merged = add_audit_flags(merged)

    aug = add_audit_flags(aug)

    merged.to_csv(OUTPUT_PATH, index=False)
    aug.to_csv(AUG_ONLY_PATH, index=False)
    AUDIT_PATH.write_text(build_audit(base, normalized, aug, merged, signal_changes), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {AUG_ONLY_PATH}")
    print(f"Wrote {AUDIT_PATH}")
    print(f"Base rows: {len(base)}")
    print(f"Added rows: {len(aug)}")
    print(f"Final rows: {len(merged)}")


if __name__ == "__main__":
    main()
