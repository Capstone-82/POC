from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PHASE1_PATH = BASE_DIR.parent / "phase1" / "dataset_prompt_profiling_v4.csv"
PHASE2_PATH = BASE_DIR / "prompt_example_classifier_bedrock_output.csv"

CLEAN_PHASE2_PATH = BASE_DIR / "prompt_example_classifier_phase2_cleaned.csv"
MERGED_PATH = BASE_DIR / "prompt_classifier_phase1_phase2_merged_cleaned.csv"
AUDIT_PATH = BASE_DIR / "merged_dataset_audit_summary.txt"

VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}
SCORE_COLUMNS = ["d1", "d2", "d3", "d4", "d5"]
DROP_PHASE2_COLUMNS = ["good_prompt", "bad_prompt", "error", "notes"]

INTENT_REMAP = {
    "GENERATION": "ANALYTICAL",
    "generation": "ANALYTICAL",
    "CLASSIFICATION": "FACTUAL",
    "coding": "ANALYTICAL",
    "FACTUAL|ANALYTICAL": "ANALYTICAL",
}

TASK_REMAP = {
    "translation": "generation",
    "explanation": "reasoning",
    "generation|reasoning": "reasoning",
    "classification|generation": "classification",
}

DOMAIN_TO_PROMPT_TYPE = {
    "FinOps": "INFORMATIONAL",
    "DevOps": "INSTRUCTIONAL",
    "Cloud Infrastructure": "INFORMATIONAL",
    "Security": "ANALYSIS_CRITIQUE",
    "Data Engineering": "DATA_EXTRACTION",
    "General Enterprise": "INFORMATIONAL",
    "AI Governance": "ANALYSIS_CRITIQUE",
    "System Integration": "INSTRUCTIONAL",
    "Supply Chain": "ANALYSIS_CRITIQUE",
    "Marketing Tech": "ANALYSIS_CRITIQUE",
    "IoT & Smart Factory": "INFORMATIONAL",
    "HR Tech": "INFORMATIONAL",
    "Regulatory Compliance": "ANALYSIS_CRITIQUE",
    "AI/LLM": "INFORMATIONAL",
    "Competitive Intelligence": "COMPARISON",
}

RESEARCH_SIGNAL_KEYWORDS = {
    "market_research": ["market", "industry", "trend", "tam", "sam", "som"],
    "competitive_analysis": ["competitor", "competitive", "benchmark", "rival"],
    "regulatory_compliance": ["regulation", "regulatory", "compliance", "gdpr", "hipaa", "sox"],
    "security": ["security", "vulnerability", "threat", "risk", "iam", "zero trust"],
    "cloud_infrastructure": ["aws", "azure", "gcp", "cloud", "kubernetes", "terraform"],
    "finops": ["finops", "cost", "spend", "budget", "showback", "chargeback"],
    "devops": ["ci/cd", "pipeline", "deployment", "sre", "devops", "observability"],
    "data_engineering": ["data pipeline", "etl", "warehouse", "lakehouse", "spark"],
    "ai_governance": ["ai governance", "llm", "model risk", "genai", "guardrail"],
    "system_integration": ["integration", "api", "webhook", "middleware"],
    "supply_chain": ["supply chain", "inventory", "procurement", "logistics"],
    "hr_tech": ["hr", "employee", "workforce", "talent", "recruiting"],
    "vendor_analysis": ["vendor", "rfi", "rfp", "procurement"],
}


def complexity_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["d1"] * 0.35
        + df["d2"] * 0.20
        + df["d3"] * 0.20
        + df["d4"] * 0.15
        + df["d5"] * 0.10
    )


def score_to_tier(score: float) -> str:
    if score < 0.40:
        return "T1"
    if score < 0.70:
        return "T2"
    return "T3"


def tier_to_complexity(tier: str) -> str:
    return {"T1": "low", "T2": "medium", "T3": "high"}[tier]


def normalize_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def parse_signal_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def signals_to_csv_cell(signals: list[str]) -> str:
    return json.dumps(signals, ensure_ascii=True)


def derive_intent_from_d1(d1: float) -> str:
    if d1 <= 0.25:
        return "FACTUAL"
    if d1 == 0.50:
        return "ANALYTICAL"
    if d1 == 0.75:
        return "SYNTHETIC"
    return "STRATEGIC"


def has_any_term(text: str, terms: list[str]) -> bool:
    for term in terms:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text):
            return True
    return False


def derive_task_type(prompt: str) -> str:
    text = prompt.lower()
    if has_any_term(
        text,
        [
            "python",
            "sql query",
            "source code",
            "code",
            "function",
            "script",
            "debug",
            "yaml",
            "json",
            "syntax error",
            "stack trace",
            "kubernetes manifest",
        ],
    ):
        return "coding"
    if has_any_term(text, ["summarize", "summary", "tl;dr", "condense"]):
        return "summarisation"
    if has_any_term(text, ["classify", "label", "categorize"]):
        return "classification"
    if has_any_term(text, ["sparql", "rdf", "ontology", "knowledge graph"]):
        return "sparql_generation"
    if has_any_term(text, ["format", "rewrite", "rephrase", "tone", "style"]):
        return "formatting"
    if has_any_term(text, ["create", "draft", "write", "generate", "compose", "build"]):
        return "generation"
    if has_any_term(text, ["compare", "analyze", "evaluate", "assess", "why", "how"]):
        return "reasoning"
    return "reasoning"


def derive_research_signals(prompt: str, d4: float) -> list[str]:
    if d4 <= 0:
        return []

    text = prompt.lower()
    signals: list[str] = []
    for signal, keywords in RESEARCH_SIGNAL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            signals.append(signal)

    if not signals:
        signals.append("external_research")
    return signals


def add_derived_score_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["complexity_score"] = complexity_score(df).round(4)
    df["tier"] = df["complexity_score"].apply(score_to_tier)
    df["complexity"] = df["tier"].apply(tier_to_complexity)
    return df


def clean_phase2() -> pd.DataFrame:
    df = pd.read_csv(PHASE2_PATH)

    required_label_columns = [
        "intent",
        "task_type",
        "reasoning_chain_detected",
        *SCORE_COLUMNS,
        "confidence",
    ]
    df = df.dropna(subset=required_label_columns, how="all")
    df = df.drop_duplicates(subset=["prompt"], keep="first")
    df = df.drop(columns=[col for col in DROP_PHASE2_COLUMNS if col in df.columns])

    for col in SCORE_COLUMNS:
        df[col] = df[col].astype(float)

    df["intent"] = df["intent"].replace(INTENT_REMAP)
    df["task_type"] = df["task_type"].replace(TASK_REMAP)
    df["reasoning_chain_detected"] = df["reasoning_chain_detected"].apply(normalize_bool)
    df["research_signals"] = df["research_signals"].apply(parse_signal_list).apply(signals_to_csv_cell)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["low_confidence_flag"] = df["confidence"] <= 0.5

    df["original_complexity"] = df["complexity"]
    df = add_derived_score_fields(df)

    df["id"] = None
    df["phrasing_style"] = None
    df["domain"] = None
    df["source"] = "phase2"
    return df


def expand_phase1() -> pd.DataFrame:
    df = pd.read_csv(PHASE1_PATH)
    df = df.drop_duplicates(subset=["prompt"], keep="first")

    for col in SCORE_COLUMNS:
        df[col] = df[col].astype(float)

    df["intent"] = df["d1"].apply(derive_intent_from_d1)
    df["task_type"] = df["prompt"].apply(derive_task_type)
    df["reasoning_chain_detected"] = df["d1"] >= 0.50
    df["research_signals"] = df.apply(
        lambda row: signals_to_csv_cell(derive_research_signals(row["prompt"], row["d4"])),
        axis=1,
    )
    df["confidence"] = pd.NA
    df["low_confidence_flag"] = False
    df["task_description"] = df["prompt"].str.slice(0, 80)
    df["expected_answer"] = pd.NA
    df["prompting_techniques"] = pd.NA
    df["prompt_type"] = df["domain"].map(DOMAIN_TO_PROMPT_TYPE).fillna("INFORMATIONAL")
    df["original_complexity"] = pd.NA
    df = add_derived_score_fields(df)
    df["source"] = "phase1"
    return df


def build_audit_summary(cleaned_phase2: pd.DataFrame, merged: pd.DataFrame) -> str:
    lines = [
        "Clean and merge audit summary",
        "=" * 32,
        f"Phase 2 cleaned output: {CLEAN_PHASE2_PATH.name}",
        f"Merged output: {MERGED_PATH.name}",
        "",
        f"Cleaned Phase 2 shape: {cleaned_phase2.shape[0]} rows x {cleaned_phase2.shape[1]} columns",
        f"Merged shape: {merged.shape[0]} rows x {merged.shape[1]} columns",
        "",
        "Source counts:",
        merged["source"].value_counts(dropna=False).to_string(),
        "",
        "Tier counts:",
        merged["tier"].value_counts(dropna=False).sort_index().to_string(),
        "",
        "Intent counts:",
        merged["intent"].value_counts(dropna=False).to_string(),
        "",
        "Task type counts:",
        merged["task_type"].value_counts(dropna=False).to_string(),
        "",
        "D-score coverage:",
    ]

    for col in SCORE_COLUMNS:
        counts = merged[col].value_counts(dropna=False).sort_index()
        missing = sorted(VALID_SCORES - set(merged[col].dropna().unique()))
        lines.append(f"{col}: {counts.to_dict()} | missing={missing}")

    lines.extend(
        [
            "",
            "Low-confidence Phase 2 rows retained but flagged:",
            str(int(cleaned_phase2["low_confidence_flag"].sum())),
            "",
            "Original Phase 2 complexity mismatches preserved in original_complexity:",
            str(int((cleaned_phase2["original_complexity"] != cleaned_phase2["complexity"]).sum())),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    cleaned_phase2 = clean_phase2()
    expanded_phase1 = expand_phase1()

    output_columns = [
        "id",
        "prompt",
        "intent",
        "task_type",
        "reasoning_chain_detected",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "complexity_score",
        "tier",
        "complexity",
        "original_complexity",
        "research_signals",
        "confidence",
        "low_confidence_flag",
        "task_description",
        "expected_answer",
        "prompting_techniques",
        "prompt_type",
        "phrasing_style",
        "domain",
        "source",
    ]

    cleaned_phase2 = cleaned_phase2.reindex(columns=output_columns)
    expanded_phase1 = expanded_phase1.reindex(columns=output_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        merged = pd.concat([cleaned_phase2, expanded_phase1], ignore_index=True)

    cleaned_phase2.to_csv(CLEAN_PHASE2_PATH, index=False)
    merged.to_csv(MERGED_PATH, index=False)
    AUDIT_PATH.write_text(build_audit_summary(cleaned_phase2, merged), encoding="utf-8")

    print(f"Wrote {CLEAN_PHASE2_PATH}")
    print(f"Wrote {MERGED_PATH}")
    print(f"Wrote {AUDIT_PATH}")


if __name__ == "__main__":
    main()
