# Phase 2 Prompt Profiling Dataset Handoff

## Dataset File

Use this CSV for model training and architecture experiments:

```text
prompt_classifier_phase1_phase2_merged_cleaned.csv
```

Shape:

```text
2273 rows x 24 columns
```

Source composition:

```text
phase2: 1447 rows
phase1: 826 rows
```

Duplicate prompts:

```text
0
```

## Expected Inference Output

The trained model should return JSON in this shape:

```json
{
  "d1": 0.75,
  "d1_label": "Semantic Complexity",
  "d2": 0.50,
  "d2_label": "Domain Specificity",
  "d3": 0.75,
  "d3_label": "Output Formality",
  "d4": 0.25,
  "d4_label": "Research Dependency",
  "d5": 0.50,
  "d5_label": "Context Requirement",
  "complexity_score": 0.5625,
  "tier": "T2",
  "intent": "ANALYTICAL",
  "task_type": "reasoning",
  "reasoning_chain_detected": true,
  "research_signals": ["market_research", "competitive_analysis"],
  "confidence": 0.85
}
```

## Core Prediction Targets

Trainable targets:

- `d1`
- `d2`
- `d3`
- `d4`
- `d5`
- `tier`
- `intent`
- `task_type`
- `reasoning_chain_detected`
- `research_signals`

Do not train `confidence` as a normal target. It should be derived at inference time from model probabilities or calibration.

## Dimension Labels

Use these fixed labels in inference output:

| Field | Label |
|---|---|
| `d1` | `Semantic Complexity` |
| `d2` | `Domain Specificity` |
| `d3` | `Output Formality` |
| `d4` | `Research Dependency` |
| `d5` | `Context Requirement` |

## Valid D-Score Values

Each dimension uses the same ordinal score vocabulary:

```text
0.0, 0.25, 0.5, 0.75, 1.0
```

All five score levels are present for all five dimensions in the dataset.

## Complexity Score and Tier Formula

Derive `complexity_score` from the five D-scores:

```text
complexity_score =
  d1 * 0.35 +
  d2 * 0.20 +
  d3 * 0.20 +
  d4 * 0.15 +
  d5 * 0.10
```

Tier thresholds:

```text
T1: complexity_score < 0.40
T2: 0.40 <= complexity_score < 0.70
T3: complexity_score >= 0.70
```

The CSV already includes `complexity_score` and `tier`, but teams should recompute them during validation to check consistency.

## Valid Intent Values

```text
FACTUAL
ANALYTICAL
SYNTHETIC
STRATEGIC
```

Intent meaning:

| Intent | Meaning |
|---|---|
| `FACTUAL` | Single known answer or factual recall |
| `ANALYTICAL` | Multi-step reasoning within one domain |
| `SYNTHETIC` | Cross-domain synthesis of multiple concepts |
| `STRATEGIC` | Strategic analysis requiring novel synthesis or external research |

## Valid Task Type Values

```text
classification
generation
reasoning
coding
summarisation
sparql_generation
formatting
```

Important note:

```text
sparql_generation has 0 samples in this dataset.
```

If teams train a classifier directly on this CSV, they should either exclude `sparql_generation` from the trainable class list or add labeled samples before expecting the model to predict it.

## Research Signals

`research_signals` is stored as a JSON-list string in the CSV.

Examples:

```json
[]
```

```json
["security", "regulatory_compliance"]
```

Recommended handling:

- Parse the field as JSON.
- Treat it as a multi-label target if training a learned research signal model.
- A rule-based extractor is also acceptable for a first architecture.

## Confidence

`confidence` is metadata, not a standard supervised target.

Phase 2 rows may contain source confidence values from the original Bedrock labeling process.

Phase 1 rows have null confidence because Phase 1 did not include this label.

Recommended inference behavior:

- derive confidence from model probability outputs;
- use calibrated probabilities if available;
- for multi-head architectures, combine head confidences conservatively, for example using minimum or weighted average of max probabilities.

## Important Columns

| Column | Use |
|---|---|
| `prompt` | Main model input |
| `d1`-`d5` | Ordinal dimension targets |
| `complexity_score` | Derived score, useful for validation |
| `tier` | Final routing tier target |
| `intent` | 4-class intent target |
| `task_type` | Task category target |
| `reasoning_chain_detected` | Boolean target |
| `research_signals` | Multi-label/list target |
| `source` | Provenance: `phase1` or `phase2` |
| `low_confidence_flag` | Phase 2 label-quality flag |
| `prompt_type` | Useful metadata or feature |
| `domain` | Phase 1 domain metadata |
| `phrasing_style` | Phase 1 prompt style metadata |

## Columns to Treat Carefully

| Column | Guidance |
|---|---|
| `confidence` | Do not train as a normal target |
| `original_complexity` | Raw Phase 2 complexity label, kept only for audit |
| `complexity` | Human-readable derived tier label: `low`, `medium`, `high` |
| `task_description` | Metadata; often derived or approximate |
| `expected_answer` | Available mainly for Phase 2 rows |
| `prompting_techniques` | Available mainly for Phase 2 rows |

## Evaluation Notes

Recommended evaluation:

- Use stratified cross-validation for `tier`.
- Report confusion matrix for `T1`, `T2`, and `T3`.
- Pay special attention to `T2 <-> T3` boundary confusion.
- Report per-dimension metrics for `d1`-`d5`.
- Report class-wise precision/recall/F1 for `intent` and `task_type`.
- Evaluate `research_signals` with multi-label precision, recall, F1, and exact-match only as a stricter secondary metric.

Suggested handling:

- Use class weighting or sample weighting for imbalanced targets.
- Do not evaluate on low-confidence Phase 2 rows if the goal is clean benchmark reporting.
- Keep `source` available for split analysis to check whether Phase 1-derived rows and Phase 2 rows behave differently.

## Current Dataset Profile

Tier counts:

```text
T1: 942
T2: 1018
T3: 313
```

Intent counts:

```text
ANALYTICAL: 1306
FACTUAL: 413
SYNTHETIC: 380
STRATEGIC: 174
```

Task type counts:

```text
reasoning: 1226
generation: 766
summarisation: 112
classification: 90
coding: 77
formatting: 2
sparql_generation: 0
```

Known limitations:

- `formatting` has only 2 samples.
- `sparql_generation` has 0 samples.
- `D3` and `D4` remain imbalanced.
- `reasoning_chain_detected` is majority `True`.
- Phase 1 rows contain derived Phase 2 fields, not independently labeled Phase 2 fields.

## Recommended Baseline Architecture

A practical first architecture is a shared-feature multi-head classifier:

```text
Prompt
  -> text embedding
  -> hand-crafted prompt features
  -> shared feature matrix
  -> separate heads for:
       d1-d5
       tier
       intent
       task_type
       reasoning_chain_detected
       research_signals
```

Recommended starting point:

- Use the Phase 1 v4 style backbone if available.
- Train separate heads first for easier debugging.
- Derive `confidence` from prediction probabilities.
- Consider rule-based `research_signals` first, then move to multi-label learning if needed.

## Handoff Status

This dataset is ready for team audit and architecture experiments.

No model training has been performed as part of this handoff.
