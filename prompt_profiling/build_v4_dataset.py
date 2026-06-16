"""
Build the v4 prompt profiling dataset.

Inputs:
  - dataset_prompt_profiling_v2.csv

Outputs:
  - dataset_prompt_profiling_v4.csv
  - dataset_v4_audit.md

The expansion is intentionally targeted rather than random:
  - D2=0.00 simple/general prompts
  - D4=0.50 prompts with multiple provided artifacts but no live retrieval
  - D5=1.00 prompts with very large/multi-document context requirements
  - boundary T2/T3 examples where tier accuracy usually breaks
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "dataset_prompt_profiling_v2.csv"
OUTPUT_CSV = BASE_DIR / "dataset_prompt_profiling_v4.csv"
AUDIT_MD = BASE_DIR / "dataset_v4_audit.md"

WEIGHTS = {"d1": 0.35, "d2": 0.20, "d3": 0.20, "d4": 0.15, "d5": 0.10}
SCORE_VALUES = {0.0, 0.25, 0.5, 0.75, 1.0}
COLUMNS = ["id", "prompt", "phrasing_style", "domain", "d1", "d2", "d3", "d4", "d5"]


DOMAINS = [
    "Cloud Infrastructure",
    "DevOps",
    "Data Engineering",
    "Security",
    "FinOps",
    "AI Governance",
    "Marketing Tech",
    "Supply Chain",
    "HR Tech",
    "IoT & Smart Factory",
    "System Integration",
    "Regulatory Compliance",
    "General Enterprise",
]

SIMPLE_TOPICS = [
    ("what is a cloud region", "General Enterprise"),
    ("what is a service account", "Security"),
    ("what does high availability mean", "Cloud Infrastructure"),
    ("what is a deployment pipeline", "DevOps"),
    ("what is a data warehouse", "Data Engineering"),
    ("what does mfa stand for", "Security"),
    ("what is a webhook", "System Integration"),
    ("what is a purchase order", "Supply Chain"),
    ("what is employee onboarding", "HR Tech"),
    ("what is a marketing qualified lead", "Marketing Tech"),
    ("what is cost allocation", "FinOps"),
    ("what is model drift", "AI Governance"),
    ("what is modbus", "IoT & Smart Factory"),
    ("what is a retention policy", "Regulatory Compliance"),
    ("what is an api gateway", "System Integration"),
    ("what is object storage", "Cloud Infrastructure"),
    ("what is a container image", "DevOps"),
    ("what is schema validation", "Data Engineering"),
    ("what is least privilege access", "Security"),
    ("what is demand forecasting", "Supply Chain"),
]

ARTIFACT_SETS = [
    ("architecture diagram", "incident log", "service inventory"),
    ("billing export", "tagging report", "team allocation sheet"),
    ("network topology", "firewall policy", "asset inventory"),
    ("etl run history", "data catalog", "schema change log"),
    ("on-call report", "alert catalog", "runbook draft"),
    ("campaign export", "crm field map", "attribution report"),
    ("warehouse report", "supplier scorecard", "shipment exception log"),
    ("employee survey", "org chart", "attrition dashboard"),
    ("factory sensor map", "plc error log", "maintenance schedule"),
    ("api contract", "queue configuration", "integration error log"),
    ("access review export", "policy draft", "audit evidence folder"),
    ("model card", "risk register", "evaluation report"),
]

BIG_CONTEXT_ARTIFACTS = [
    "three years of cloud billing exports, resource tags, architecture diagrams, and vendor contracts",
    "two years of incident tickets, postmortems, monitoring exports, and runbook history",
    "full data catalog, dbt project, lineage export, warehouse query history, and access logs",
    "global HRIS export, job architecture, compensation bands, attrition history, and policy documents",
    "complete supplier contracts, logistics data, demand forecast history, and warehouse telemetry",
    "source code inventory, CI/CD configuration, security scan history, and deployment audit logs",
    "customer journey analytics, campaign exports, CRM opportunity history, and consent records",
    "regulatory control library, audit findings, data processing inventory, and vendor risk files",
    "factory telemetry archive, device inventory, maintenance logs, and network packet captures",
    "model registry exports, prompt logs, evaluation datasets, risk assessments, and approval records",
]


def complexity_score(row: pd.Series | dict) -> float:
    return sum(float(row[k]) * w for k, w in WEIGHTS.items())


def get_tier(score: float) -> str:
    if score < 0.40:
        return "T1"
    if score < 0.70:
        return "T2"
    return "T3"


def add_row(rows, row_id, prompt, phrasing_style, domain, d1, d2, d3, d4, d5):
    rows.append((row_id, prompt, phrasing_style, domain, d1, d2, d3, d4, d5))


def build_new_rows() -> pd.DataFrame:
    rows = []

    # 77 rows: D2=0.00 simple/general examples. A few are near the T1/T2 edge.
    for i in range(77):
        topic, domain = SIMPLE_TOPICS[i % len(SIMPLE_TOPICS)]
        style = ["explicit", "vague", "implicit"][i % 3]
        if i % 10 in {0, 1, 2}:
            prompt = f"{topic} and give a short example"
            d1, d3 = 0.25, 0.25
        else:
            prompt = topic
            d1, d3 = 0.0, 0.0
        add_row(rows, f"v4_d2zero_{i+1:03d}", prompt, style, domain, d1, 0.0, d3, 0.0, 0.0)

    # 72 rows: D4=0.50, multiple provided artifacts, no live retrieval.
    # These sit deliberately near the T2/T3 boundary at score 0.6875.
    for i in range(72):
        domain = DOMAINS[i % len(DOMAINS)]
        a1, a2, a3 = ARTIFACT_SETS[i % len(ARTIFACT_SETS)]
        style = ["explicit", "implicit", "vague"][i % 3]
        if style == "vague":
            prompt = (
                f"the {domain.lower()} situation is messy; using the attached {a1}, {a2}, and {a3}, "
                "figure out the root causes and write a structured remediation plan with owners and risks"
            )
        elif style == "implicit":
            prompt = (
                f"Our {domain.lower()} team has attached the {a1}, {a2}, and {a3}. "
                "Analyze the patterns across these documents and produce a formal improvement plan, "
                "including findings, recommendations, dependencies, and success metrics."
            )
        else:
            prompt = (
                f"Using the attached {a1}, {a2}, and {a3}, create a formal {domain.lower()} assessment "
                "with a gap analysis, prioritized remediation roadmap, decision log, and implementation checklist."
            )
        add_row(rows, f"v4_d4mid_{i+1:03d}", prompt, style, domain, 0.75, 0.75, 0.75, 0.50, 0.25)

    # 80 rows: D5=1.00 long-context T3. First 24 are boundary-ish T3, the rest are high-confidence T3.
    for i in range(80):
        domain = DOMAINS[i % len(DOMAINS)]
        artifacts = BIG_CONTEXT_ARTIFACTS[i % len(BIG_CONTEXT_ARTIFACTS)]
        style = ["explicit", "implicit", "vague", "explicit"][i % 4]
        if i < 24:
            d1, d2, d3, d4 = 0.75, 0.75, 0.75, 0.50
            prompt = (
                f"Using the full uploaded archive of {artifacts}, build a formal {domain.lower()} findings report "
                "with prioritized recommendations, an implementation roadmap, and evidence references for each decision."
            )
        else:
            d1, d2, d3 = 1.0, 1.0, 1.0
            d4 = [0.75, 1.0, 0.75, 1.0][i % 4]
            if d4 == 1.0:
                research_clause = (
                    "and incorporate current market pricing, recent vendor documentation, and relevant regulatory updates"
                )
            else:
                research_clause = (
                    "and compare the uploaded evidence against current provider documentation and industry best practices"
                )
            prompt = (
                f"Using the complete uploaded corpus of {artifacts}, create an enterprise-grade {domain.lower()} package "
                f"with executive summary, detailed technical specification, risk register, operating model, phased roadmap, "
                f"cost model, governance controls, and appendix-level evidence mapping, {research_clause}."
            )
        add_row(rows, f"v4_d5max_{i+1:03d}", prompt, style, domain, d1, d2, d3, d4, 1.0)

    # 45 rows: D5=0.75 plus D4=1.00. These are intentionally medium-tier
    # live-research tasks with sizable context, so the classifier learns that
    # research dependency alone does not always mean T3.
    for i in range(45):
        domain = DOMAINS[i % len(DOMAINS)]
        style = ["explicit", "implicit", "vague"][i % 3]
        if style == "vague":
            prompt = (
                f"we have a long internal packet about {domain.lower()} and need the latest external facts checked; "
                "compare current vendor or regulatory information against it and give a focused recommendation"
            )
        elif style == "implicit":
            prompt = (
                f"Our team uploaded a large {domain.lower()} context bundle. Check current vendor documentation, "
                "recent pricing, or latest regulatory guidance as relevant, then summarize what changes our plan needs."
            )
        else:
            prompt = (
                f"Using the attached {domain.lower()} context bundle, research the latest vendor documentation, "
                "current pricing, or recent regulatory guidance and produce a concise structured recommendation."
            )
        add_row(rows, f"v4_d5high_d4max_{i+1:03d}", prompt, style, domain, 0.50, 0.50, 0.50, 1.00, 0.75)

    df_new = pd.DataFrame(rows, columns=COLUMNS)
    return df_new


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    bad_scores = {}
    for col in ["d1", "d2", "d3", "d4", "d5"]:
        values = set(df[col].dropna().astype(float).unique())
        bad = sorted(values - SCORE_VALUES)
        if bad:
            bad_scores[col] = bad
    if bad_scores:
        raise ValueError(f"Invalid score values: {bad_scores}")

    if df[COLUMNS].isna().sum().sum() > 0:
        raise ValueError("Null values found in required columns")
    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"Duplicate ids found: {dupes}")

    audited = df[COLUMNS].copy()
    audited["complexity_score"] = audited.apply(complexity_score, axis=1)
    audited["tier"] = audited["complexity_score"].apply(get_tier)
    return audited


def markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    table.insert(0, "value", table.index)
    headers = [str(c) for c in table.columns]
    rows = [[str(v) for v in row] for row in table.to_numpy()]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_audit(before: pd.DataFrame, after: pd.DataFrame, new_rows: pd.DataFrame) -> None:
    before_a = validate_dataset(before)
    after_a = validate_dataset(after)
    new_a = validate_dataset(new_rows)

    audit_lines = [
        "# Dataset v4 Audit",
        "",
        "## Summary",
        "",
        f"- Base dataset: {len(before_a)} rows",
        f"- Added targeted rows: {len(new_a)} rows",
        f"- Final dataset: {len(after_a)} rows",
        "- Score values validated against {0.0, 0.25, 0.50, 0.75, 1.0}",
        "- No duplicate ids",
        "- No nulls in required columns",
        "",
        "## Tier Distribution",
        "",
        markdown_table(pd.DataFrame({
            "v2": before_a["tier"].value_counts().sort_index(),
            "v4": after_a["tier"].value_counts().sort_index(),
        }).fillna(0).astype(int)),
        "",
        "## Dimension Distributions",
        "",
    ]

    for col in ["d1", "d2", "d3", "d4", "d5"]:
        dist = pd.DataFrame({
            "v2": before_a[col].value_counts().sort_index(),
            "added": new_a[col].value_counts().sort_index(),
            "v4": after_a[col].value_counts().sort_index(),
        }).fillna(0).astype(int)
        audit_lines.extend([f"### {col.upper()}", "", markdown_table(dist), ""])

    boundary_040 = after_a[(after_a["complexity_score"] >= 0.35) & (after_a["complexity_score"] < 0.45)]
    boundary_070 = after_a[(after_a["complexity_score"] >= 0.65) & (after_a["complexity_score"] < 0.75)]
    audit_lines.extend([
        "## Boundary Coverage",
        "",
        f"- T1/T2 boundary window [0.35, 0.45): {len(boundary_040)} rows",
        f"- T2/T3 boundary window [0.65, 0.75): {len(boundary_070)} rows",
        "",
        "## Notes",
        "",
        "- D3 is treated as Output Formality, matching the source rubric.",
        "- D5 is treated as Context Requirement, matching the source rubric.",
        "- D4=0.50 rows are written as multi-document/provided-artifact analysis, not live web retrieval.",
        "- D5=1.00 rows explicitly describe very large uploaded corpora or multi-document injection.",
        "- Existing v2 rows were preserved; v4 adds targeted rubric-aligned coverage rather than rewriting old labels.",
        "",
    ])
    AUDIT_MD.write_text("\n".join(audit_lines), encoding="utf-8")


def main() -> None:
    df_base = pd.read_csv(INPUT_CSV)
    df_new = build_new_rows()
    df_v4 = pd.concat([df_base[COLUMNS], df_new], ignore_index=True)

    validate_dataset(df_v4)
    df_v4.to_csv(OUTPUT_CSV, index=False)
    write_audit(df_base, df_v4, df_new)

    audited = validate_dataset(df_v4)
    print(f"Wrote {OUTPUT_CSV.name}: {len(df_v4)} rows")
    print(f"Wrote {AUDIT_MD.name}")
    print("Tier distribution:")
    print(audited["tier"].value_counts().sort_index())
    print("Minimum class counts by dimension:")
    for col in ["d1", "d2", "d3", "d4", "d5"]:
        print(f"  {col}: {audited[col].value_counts().min()}")


if __name__ == "__main__":
    main()
