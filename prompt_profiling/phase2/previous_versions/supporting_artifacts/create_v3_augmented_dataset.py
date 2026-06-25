from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "dataset_for_team" / "prompt_classifier_phase1_phase2_merged_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "v3_working"
OUTPUT_PATH = OUTPUT_DIR / "prompt_classifier_phase2_v3_augmented.csv"
AUG_ONLY_PATH = OUTPUT_DIR / "prompt_classifier_phase2_v3_augmented_rows_only.csv"
AUDIT_PATH = OUTPUT_DIR / "V3_AUGMENTATION_AUDIT.md"

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
    task_description: str,
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
        "task_description": task_description[:120],
        "expected_answer": pd.NA,
        "prompting_techniques": pd.NA,
        "prompt_type": prompt_type,
        "phrasing_style": pd.NA,
        "domain": pd.NA,
        "source": "aug_v3",
    }


T3_DOMAINS = [
    ("healthcare AI", "HIPAA, model risk, and clinical safety", ["regulatory_compliance", "security", "ai_governance"]),
    ("multi-cloud FinOps", "AWS, Azure, unit economics, and chargeback policy", ["finops", "cloud_infrastructure", "vendor_analysis"]),
    ("supply chain resilience", "supplier risk, inventory buffers, and geopolitical exposure", ["supply_chain", "market_research"]),
    ("HR analytics governance", "EU AI Act, Workday, bias testing, and employee appeals", ["hr_tech", "regulatory_compliance", "ai_governance"]),
    ("data platform modernization", "Snowflake, Databricks, lineage, quality SLAs, and migration risk", ["data_engineering", "cloud_infrastructure"]),
    ("cybersecurity transformation", "zero trust, identity governance, audit findings, and board risk", ["security", "regulatory_compliance"]),
    ("competitive product strategy", "market benchmarks, pricing, customer segments, and roadmap tradeoffs", ["competitive_analysis", "market_research"]),
    ("GenAI vendor selection", "RFP scoring, legal review, data retention, and deployment architecture", ["vendor_analysis", "ai_governance", "security"]),
    ("IoT factory optimization", "edge telemetry, PLC reliability, maintenance windows, and safety controls", ["cloud_infrastructure", "system_integration"]),
    ("regulatory reporting automation", "SOX controls, evidence collection, workflow approvals, and audit trails", ["regulatory_compliance", "system_integration"]),
]

TASK_TEMPLATES = {
    "reasoning": "Design a strategic recommendation for {domain}. The board needs a decision memo that synthesizes {details}. Include options, tradeoffs, phased risks, success metrics, and a final recommendation.",
    "generation": "Create a complete implementation package for {domain}. It must include an executive summary, target-state architecture, operating model, roadmap, RACI, risk register, and migration plan covering {details}.",
    "summarisation": "Summarize the attached executive packet for {domain}, then turn it into a board-ready brief. The packet covers {details}. Include key decisions, unresolved risks, evidence gaps, and recommended next actions.",
    "coding": "Design Python pseudocode and SQL validation checks for {domain}. The solution must process governance evidence, score risk, generate audit-ready reports, and handle edge cases involving {details}.",
    "classification": "Classify the following portfolio items for {domain} into risk tiers and investment priorities. Explain the criteria and apply them to examples involving {details}.",
    "formatting": "Reformat the raw policy notes for {domain} into a strict JSON schema plus a markdown executive summary. Preserve all controls, exceptions, owners, dates, and evidence fields related to {details}.",
    "sparql_generation": "Generate SPARQL queries for a knowledge graph about {domain}. The ontology includes vendors, controls, risks, systems, owners, and evidence related to {details}. Include prefixes and explain each query.",
}


def build_phase2_style_t3_rows() -> list[dict]:
    rows: list[dict] = []
    task_cycle = ["reasoning", "generation", "summarisation", "coding", "classification", "formatting", "sparql_generation"]
    score_patterns = [
        (1.0, 1.0, 1.0, 1.0, 1.0, "STRATEGIC"),
        (1.0, 1.0, 0.75, 1.0, 0.75, "STRATEGIC"),
        (1.0, 0.75, 1.0, 0.75, 1.0, "STRATEGIC"),
        (0.75, 1.0, 1.0, 1.0, 0.75, "SYNTHETIC"),
        (1.0, 0.75, 0.75, 1.0, 1.0, "STRATEGIC"),
    ]
    idx = 1
    for domain, details, signals in T3_DOMAINS:
        for variant in range(10):
            task_type = task_cycle[(idx + variant) % len(task_cycle)]
            d1, d2, d3, d4, d5, intent = score_patterns[(idx + variant) % len(score_patterns)]
            prompt = TASK_TEMPLATES[task_type].format(domain=domain, details=details)
            prompt += (
                " Make the output detailed enough for executive, technical, legal, and operating stakeholders. "
                f"Scenario variant {variant + 1}: emphasize "
                f"{['operating model', 'risk quantification', 'vendor tradeoffs', 'timeline sequencing', 'governance controls'][variant % 5]}."
            )
            rows.append(
                make_row(
                    f"aug_v3_t3_{idx:03d}",
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
                    "AUGMENTED_T3",
                    f"Phase2-style T3 prompt for {domain}",
                )
            )
            idx += 1
    return rows


BOUNDARY_CASES = [
    ("cloud cost anomaly review", "AWS spend, unused EBS volumes, and simple owner follow-up", ["finops", "cloud_infrastructure"]),
    ("vendor security exception", "SOC 2 gap, compensating controls, and renewal decision", ["security", "vendor_analysis"]),
    ("data pipeline reliability", "late-arriving events, dashboard SLAs, and owner escalation", ["data_engineering"]),
    ("HR AI screening workflow", "bias monitoring, appeal routing, and regional policy", ["hr_tech", "ai_governance"]),
    ("supply chain dashboard", "supplier delays, logistics alerts, and inventory thresholds", ["supply_chain"]),
    ("DevOps release governance", "failed deployments, rollback policy, and SRE escalation", ["devops"]),
    ("customer support GenAI", "retention policy, hallucination reports, and quality review", ["ai_governance", "security"]),
    ("API integration migration", "MuleSoft mappings, SAP records, and retry handling", ["system_integration"]),
    ("market entry comparison", "pricing signals, competitors, and channel assumptions", ["market_research", "competitive_analysis"]),
    ("audit evidence workflow", "SOX evidence, control owners, and exception aging", ["regulatory_compliance"]),
]


def build_boundary_rows() -> list[dict]:
    rows: list[dict] = []
    idx = 1
    for topic, details, signals in BOUNDARY_CASES:
        for pair in range(3):
            base = (
                f"Analyze {topic}. Use the provided context about {details}. "
                f"Give a structured recommendation with assumptions, risks, and next steps. Case variant {pair + 1}."
            )
            rows.append(
                make_row(
                    f"aug_v3_boundary_t2_{idx:03d}",
                    base + " Keep the scope limited to the current team and avoid external market research.",
                    "SYNTHETIC",
                    "reasoning" if pair % 2 else "generation",
                    True,
                    0.75,
                    0.75,
                    0.75,
                    0.25,
                    0.50,
                    [],
                    "AUGMENTED_BOUNDARY_T2",
                    f"T2 boundary case for {topic}",
                )
            )
            rows.append(
                make_row(
                    f"aug_v3_boundary_t3_{idx:03d}",
                    base + " Include current external benchmarks, vendor guidance, regulatory implications, and a cross-functional rollout plan.",
                    "SYNTHETIC",
                    "reasoning" if pair % 2 else "generation",
                    True,
                    0.75,
                    0.75,
                    0.75,
                    0.75,
                    0.50,
                    signals,
                    "AUGMENTED_BOUNDARY_T3",
                    f"T3 boundary case for {topic}",
                )
            )
            idx += 1
    return rows


def build_task_coverage_rows() -> list[dict]:
    rows: list[dict] = []
    categories = [
        ("classification", "Classify each support ticket as billing, technical, security, or account access. Return the label and one-sentence rationale for every item.", "FACTUAL", False, (0.25, 0.50, 0.25, 0.0, 0.25), [], "AUGMENTED_CLASSIFICATION"),
        ("coding", "Write a Python function that validates a JSON payload, checks required fields, normalizes dates, and returns structured validation errors with sample tests.", "ANALYTICAL", True, (0.50, 0.75, 0.75, 0.0, 0.50), [], "AUGMENTED_CODING"),
        ("summarisation", "Summarize the attached incident timeline into an executive summary, root-cause bullets, customer impact, and follow-up actions.", "ANALYTICAL", True, (0.50, 0.50, 0.75, 0.0, 0.75), [], "AUGMENTED_SUMMARISATION"),
        ("formatting", "Convert these messy implementation notes into a strict YAML runbook with owners, prerequisites, commands, rollback steps, and validation checks.", "ANALYTICAL", True, (0.50, 0.50, 0.75, 0.0, 0.75), [], "AUGMENTED_FORMATTING"),
        ("sparql_generation", "Generate SPARQL queries to retrieve applications by owner, linked controls, open risks, and related evidence documents from an enterprise governance ontology.", "ANALYTICAL", True, (0.50, 1.0, 0.75, 0.0, 0.50), [], "AUGMENTED_SPARQL"),
    ]
    contexts = [
        "cloud infrastructure",
        "customer support",
        "security operations",
        "finance operations",
        "HR workflow",
        "data governance",
        "DevOps release",
        "vendor management",
    ]
    idx = 1
    for task_type, template, intent, reasoning, scores, signals, prompt_type in categories:
        for context in contexts:
            for variant in range(1, 6):
                prompt = f"{template} Context: {context}, variant {variant}. Keep the answer precise and use the requested output structure."
                rows.append(
                    make_row(
                        f"aug_v3_task_{task_type}_{idx:03d}",
                        prompt,
                        intent,
                        task_type,
                        reasoning,
                        *scores,
                        research_signals=signals,
                        prompt_type=prompt_type,
                        task_description=f"{task_type} coverage example for {context}",
                    )
                )
                idx += 1
    return rows


def normalize_existing_research_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed: list[str] = []
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
        parsed.append(signal_cell(signals))
    df["research_signals"] = parsed
    return df, changed


def build_audit(original: pd.DataFrame, normalized: pd.DataFrame, aug: pd.DataFrame, merged: pd.DataFrame, signal_changes: int) -> str:
    lines = [
        "# V3 Augmentation Audit",
        "",
        "## Files",
        "",
        f"- Augmented dataset: `{OUTPUT_PATH.name}`",
        f"- Augmented rows only: `{AUG_ONLY_PATH.name}`",
        "",
        "## Scope",
        "",
        "The original team handoff dataset was not modified. This is a v3 working dataset.",
        "",
        "## Counts",
        "",
        f"- Original rows: `{len(original)}`",
        f"- Existing rows after research-signal normalization: `{len(normalized)}`",
        f"- Added rows: `{len(aug)}`",
        f"- Final rows: `{len(merged)}`",
        f"- Existing research_signal cells normalized: `{signal_changes}`",
        "",
        "## Source Counts",
        "",
        "```text",
        merged["source"].value_counts().to_string(),
        "```",
        "",
        "## Tier Counts",
        "",
        "```text",
        merged["tier"].value_counts().sort_index().to_string(),
        "```",
        "",
        "## Task Type Counts",
        "",
        "```text",
        merged["task_type"].value_counts().to_string(),
        "```",
        "",
        "## Intent Counts",
        "",
        "```text",
        merged["intent"].value_counts().to_string(),
        "```",
        "",
        "## D-Score Coverage",
        "",
    ]
    for col in SCORE_COLUMNS:
        lines.extend(["```text", f"{col}: {merged[col].value_counts().sort_index().to_dict()}", "```", ""])
    lines.extend(
        [
            "## Validation",
            "",
            f"- Duplicate prompts: `{int(merged['prompt'].duplicated().sum())}`",
            f"- Tier formula mismatches: `{int((merged['complexity_score'].apply(tier_from_score) != merged['tier']).sum())}`",
            f"- D4=0 with non-empty research_signals: `{count_d4_zero_nonempty(merged)}`",
            f"- D4>0 with empty research_signals: `{count_d4_positive_empty(merged)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def count_d4_zero_nonempty(df: pd.DataFrame) -> int:
    total = 0
    for _, row in df.iterrows():
        signals = json.loads(row["research_signals"])
        total += int(row["d4"] == 0 and bool(signals))
    return total


def count_d4_positive_empty(df: pd.DataFrame) -> int:
    total = 0
    for _, row in df.iterrows():
        signals = json.loads(row["research_signals"])
        total += int(row["d4"] > 0 and not signals)
    return total


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(INPUT_PATH)
    normalized, signal_changes = normalize_existing_research_signals(original)

    added_rows = build_phase2_style_t3_rows() + build_boundary_rows() + build_task_coverage_rows()
    aug = pd.DataFrame(added_rows).reindex(columns=original.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        merged = pd.concat([normalized, aug], ignore_index=True)
    merged = merged.drop_duplicates(subset=["prompt"], keep="first")

    # Recompute derived fields defensively.
    merged["complexity_score"] = merged.apply(lambda r: complexity_score(r["d1"], r["d2"], r["d3"], r["d4"], r["d5"]), axis=1)
    merged["tier"] = merged["complexity_score"].apply(tier_from_score)
    merged["complexity"] = merged["tier"].apply(complexity_from_tier)

    aug.to_csv(AUG_ONLY_PATH, index=False)
    merged.to_csv(OUTPUT_PATH, index=False)
    AUDIT_PATH.write_text(build_audit(original, normalized, aug, merged, signal_changes), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {AUG_ONLY_PATH}")
    print(f"Wrote {AUDIT_PATH}")
    print(f"Original rows: {len(original)}")
    print(f"Added rows: {len(aug)}")
    print(f"Final rows: {len(merged)}")


if __name__ == "__main__":
    main()
