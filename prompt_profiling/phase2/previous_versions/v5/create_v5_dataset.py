from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PHASE2_DIR = THIS_DIR.parent
BASE_PATH = PHASE2_DIR / "v4" / "prompt_classifier_phase2_v4_dataset.csv"
OUTPUT_PATH = THIS_DIR / "prompt_classifier_phase2_v5_dataset.csv"
AUG_ONLY_PATH = THIS_DIR / "prompt_classifier_phase2_v5_augmented_rows_only.csv"
AUDIT_PATH = THIS_DIR / "V5_DATASET_AUDIT.md"


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
    signals: list[str],
    prompt_type: str,
    augmentation_group: str,
) -> dict:
    score = complexity_score(d1, d2, d3, d4, d5)
    tier = tier_from_score(score)
    if d4 == 0:
        signals = []
    elif not signals:
        signals = ["external_research"]
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
        "research_signals": signal_cell(signals),
        "confidence": pd.NA,
        "low_confidence_flag": False,
        "task_description": augmentation_group,
        "expected_answer": pd.NA,
        "prompting_techniques": pd.NA,
        "prompt_type": prompt_type,
        "phrasing_style": pd.NA,
        "domain": pd.NA,
        "source": "aug_v5",
        "augmentation_group": augmentation_group,
    }


def build_d1_low_rows() -> list[dict]:
    # D1=0.25 examples: simple but not one-word factual. Phase2 originally lacks this level.
    topics = [
        "what a webhook is",
        "how a password reset token works",
        "what a cloud region means",
        "how a billing alert is triggered",
        "what a database index does",
        "how a support ticket priority is assigned",
        "what SOC 2 means at a high level",
        "how a CSV import validates columns",
        "what an API rate limit is",
        "how a feature flag is turned on",
        "what a DNS record does",
        "how a meeting summary should be structured",
    ]
    rows = []
    for i, topic in enumerate(topics, 1):
        prompt = f"Explain {topic} in two short paragraphs and give one simple example. Do not compare alternatives or design a solution."
        rows.append(
            make_row(
                f"v5_d1_025_{i:03d}",
                prompt,
                "FACTUAL",
                "reasoning",
                False,
                0.25,
                0.50,
                0.25,
                0.0,
                0.0,
                [],
                "V5_D1_LOW",
                "d1_025_phase2_style",
            )
        )
    return rows


def build_d1_high_rows() -> list[dict]:
    scenarios = [
        ("enterprise AI incident response", "security, legal, customer communications, model rollback, and executive reporting", ["security", "ai_governance", "regulatory_compliance"]),
        ("multi-cloud cost governance", "AWS, Azure, forecasting, unit economics, procurement, and engineering ownership", ["finops", "cloud_infrastructure", "vendor_analysis"]),
        ("global HR analytics rollout", "EU AI Act, bias testing, Workday workflows, regional approvals, and employee appeals", ["hr_tech", "ai_governance", "regulatory_compliance"]),
        ("data platform operating model", "Snowflake, Databricks, data contracts, lineage, quality SLAs, and executive dashboards", ["data_engineering", "cloud_infrastructure"]),
        ("supply chain risk control tower", "supplier concentration, inventory buffers, geopolitical risk, and logistics options", ["supply_chain", "market_research"]),
        ("zero trust transformation", "identity sprawl, audit findings, vendor access, board funding, and phased adoption", ["security", "vendor_analysis"]),
        ("GenAI procurement strategy", "vendor scoring, retention terms, evaluation datasets, legal review, and deployment architecture", ["vendor_analysis", "ai_governance"]),
        ("regulatory evidence automation", "SOX controls, audit trails, owners, exceptions, and workflow approvals", ["regulatory_compliance", "system_integration"]),
    ]
    rows = []
    for i, (scenario, details, signals) in enumerate(scenarios, 1):
        prompt = (
            f"Design a strategic plan for {scenario}. It must synthesize {details}. "
            "Include alternative architectures, operating model changes, risk tradeoffs, phased rollout, governance metrics, and a final executive recommendation."
        )
        rows.append(
            make_row(
                f"v5_d1_100_{i:03d}",
                prompt,
                "STRATEGIC",
                "reasoning",
                True,
                1.0,
                1.0,
                1.0,
                0.75,
                0.75,
                signals,
                "V5_D1_HIGH",
                "d1_100_phase2_style",
            )
        )
    return rows


def build_d2_contrast_rows() -> list[dict]:
    pairs = [
        ("explain caching", "Explain caching in simple terms with one everyday example.", "Explain Redis cache eviction, TTL strategy, and cache stampede prevention for a Kubernetes-hosted API."),
        ("write a validation function", "Write a simple function that checks whether a number is positive.", "Write a Python validator for HL7 FHIR Patient resources, including identifier systems, date normalization, and required field errors."),
        ("summarize an update", "Summarize this short team update in three bullets.", "Summarize this SOC 2 evidence packet into control IDs, exception owners, audit impact, and remediation due dates."),
        ("compare tools", "Compare email and chat for quick communication.", "Compare AWS Control Tower, Azure Policy, and Terraform Cloud for multi-account governance and drift remediation."),
        ("classify items", "Classify these fruits as citrus or non-citrus.", "Classify Kubernetes admission controller violations by CIS benchmark category, severity, and remediation owner."),
        ("create a checklist", "Create a checklist for packing for a short trip.", "Create a PCI-DSS evidence checklist for payment data flows across Snowflake, S3, IAM, and vendor support access."),
        ("explain monitoring", "Explain what monitoring means for a website.", "Explain OpenTelemetry trace sampling, Prometheus cardinality risk, and SLO burn-rate alerting for a microservices platform."),
        ("draft a plan", "Draft a simple plan for organizing a team lunch.", "Draft a NIST-aligned incident response tabletop plan for ransomware recovery across Active Directory, backups, and cloud workloads."),
    ]
    rows = []
    for i, (label, generic_prompt, domain_prompt) in enumerate(pairs, 1):
        rows.append(
            make_row(
                f"v5_d2_low_{i:03d}",
                generic_prompt,
                "FACTUAL" if i in [1, 7] else "ANALYTICAL",
                "reasoning" if i not in [2, 5] else ("coding" if i == 2 else "classification"),
                i not in [1, 5, 7],
                0.25 if i in [1, 7] else 0.50,
                0.0,
                0.25 if i in [1, 5, 7] else 0.50,
                0.0,
                0.0,
                [],
                "V5_D2_LOW",
                "d2_contrast_low",
            )
        )
        rows.append(
            make_row(
                f"v5_d2_high_{i:03d}",
                domain_prompt,
                "ANALYTICAL" if i not in [4, 8] else "SYNTHETIC",
                "coding" if i == 2 else ("classification" if i == 5 else "reasoning"),
                True,
                0.50 if i in [1, 2, 3, 5, 7] else 0.75,
                1.0,
                0.75,
                0.25 if i in [3, 6, 8] else 0.0,
                0.50,
                ["regulatory_compliance"] if i in [3, 6, 8] else [],
                "V5_D2_HIGH",
                "d2_contrast_high",
            )
        )
    return rows


def build_boundary_d1d2_rows() -> list[dict]:
    rows = []
    topics = [
        ("cloud cost allocation", "AWS linked accounts, shared NAT gateways, showback, owner tags", ["finops", "cloud_infrastructure"]),
        ("security exception triage", "SOC 2 control gap, compensating controls, vendor renewal", ["security", "vendor_analysis"]),
        ("data quality rollout", "dbt tests, dashboard SLAs, lineage, executive adoption", ["data_engineering"]),
        ("AI governance review", "model cards, retention, bias tests, legal approval", ["ai_governance", "regulatory_compliance"]),
        ("DevOps reliability plan", "failed deployments, rollback, SRE escalation, error budgets", ["devops"]),
        ("HR workflow automation", "Workday approvals, employee appeals, regional policies", ["hr_tech"]),
    ]
    for i, (topic, details, signals) in enumerate(topics, 1):
        prompt_t2 = (
            f"Using the provided internal notes about {topic}, analyze {details}. "
            "Produce a practical recommendation for the current team, with assumptions and next steps."
        )
        rows.append(
            make_row(
                f"v5_boundary_d1d2_t2_{i:03d}",
                prompt_t2,
                "SYNTHETIC",
                "reasoning",
                True,
                0.75,
                0.75,
                0.75,
                0.25,
                0.50,
                [],
                "V5_BOUNDARY_D1D2_T2",
                "boundary_d1d2_t2",
            )
        )
        prompt_t3 = (
            f"Using the provided internal notes about {topic}, analyze {details}. "
            "Compare current vendor guidance, regulatory expectations, and external benchmarks, then create an executive rollout recommendation."
        )
        rows.append(
            make_row(
                f"v5_boundary_d1d2_t3_{i:03d}",
                prompt_t3,
                "SYNTHETIC",
                "reasoning",
                True,
                0.75,
                1.0,
                0.75,
                0.75,
                0.50,
                signals,
                "V5_BOUNDARY_D1D2_T3",
                "boundary_d1d2_t3",
            )
        )
    return rows


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


def build_audit(base: pd.DataFrame, aug: pd.DataFrame, merged: pd.DataFrame) -> str:
    lines = [
        "# V5 Dataset Audit",
        "",
        "## Scope",
        "",
        "Final-iteration working dataset focused on D1/D2 and tier confidence. Original datasets are untouched.",
        "",
        "## Counts",
        "",
        f"- Base v4 rows: `{len(base)}`",
        f"- Added v5 rows: `{len(aug)}`",
        f"- Final rows: `{len(merged)}`",
        f"- Augmentation share: `{len(aug) / len(merged):.2%}`",
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
        "## D1 Counts",
        "```text",
        merged["d1"].value_counts().sort_index().to_string(),
        "```",
        "",
        "## D2 Counts",
        "```text",
        merged["d2"].value_counts().sort_index().to_string(),
        "```",
        "",
        "## Validation",
        "",
        f"- Duplicate prompts: `{int(merged['prompt'].duplicated().sum())}`",
        f"- Tier formula mismatches: `{int((merged['complexity_score'].apply(tier_from_score) != merged['tier']).sum())}`",
        f"- Original rows retained: `{int(merged['original_row_flag'].sum())}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_PATH)
    aug_rows = build_d1_low_rows() + build_d1_high_rows() + build_d2_contrast_rows() + build_boundary_d1d2_rows()
    aug = pd.DataFrame(aug_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        merged = pd.concat([base, aug], ignore_index=True)
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
    AUDIT_PATH.write_text(build_audit(base, aug, merged), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {AUG_ONLY_PATH}")
    print(f"Wrote {AUDIT_PATH}")
    print(f"Base rows: {len(base)}")
    print(f"Added rows: {len(aug)}")
    print(f"Final rows: {len(merged)}")


if __name__ == "__main__":
    main()
