# Phase 2 Dataset Cleaning and Phase 1 Merge Notes

## Scope

This step only cleans and merges datasets for audit. No model training was performed.

The original source CSV files were not modified:

- `prompt_profiling/phase1/dataset_prompt_profiling_v4.csv`
- `prompt_profiling/phase2/prompt_example_classifier_bedrock_output.csv`

## Files Created

The following new files were created under `prompt_profiling/phase2`:

- `clean_and_merge_datasets.py`
  - Reproducible script for cleaning Phase 2 and merging Phase 1 + Phase 2.

- `prompt_example_classifier_phase2_cleaned.csv`
  - Cleaned Phase 2-only dataset.
  - Shape after cleaning: `1447 rows x 24 columns`.

- `prompt_classifier_phase1_phase2_merged_cleaned.csv`
  - Final merged audit dataset containing cleaned Phase 2 rows plus expanded Phase 1 rows.
  - Shape after merge: `2273 rows x 24 columns`.

- `merged_dataset_audit_summary.txt`
  - Text summary of row counts, tier counts, intent counts, task type counts, D-score coverage, and flagged low-confidence rows.

## Phase 2 Cleaning Performed

For the Phase 2 source CSV:

- Dropped unusable columns:
  - `good_prompt`
  - `bad_prompt`
  - `error`
  - `notes`

- Dropped rows with all required label fields missing.

- Dropped duplicate Phase 2 prompts, keeping the first occurrence.

- Remapped invalid `intent` values:
  - `GENERATION` -> `ANALYTICAL`
  - `generation` -> `ANALYTICAL`
  - `CLASSIFICATION` -> `FACTUAL`
  - `coding` -> `ANALYTICAL`
  - `FACTUAL|ANALYTICAL` -> `ANALYTICAL`

- Remapped non-standard `task_type` values:
  - `translation` -> `generation`
  - `explanation` -> `reasoning`
  - `generation|reasoning` -> `reasoning`
  - `classification|generation` -> `classification`

- Converted `reasoning_chain_detected` from string-like values to boolean values.

- Parsed `research_signals` into normalized JSON-list strings.

- Added `low_confidence_flag`:
  - `True` when `confidence <= 0.5`
  - These rows were retained but flagged for audit/evaluation exclusion.

- Preserved the original Phase 2 `complexity` label in `original_complexity`.

- Re-derived clean `complexity_score`, `tier`, and `complexity` from `d1`-`d5`.

## Verdict Fixes Applied

After the first audit verdict, two additional fixes were applied:

- Deduplicated Phase 1 prompts before merge.
  - Removed `63` duplicate Phase 1 rows.
  - Phase 1 contribution changed from `889` rows to `826` rows.
  - This prevents duplicate-prompt leakage during future cross-validation.

- Removed the question-ending task type rule.
  - Prompts ending in `?` are no longer automatically assigned to `classification`.
  - They now fall through to keyword matching and default to `reasoning`.
  - Classification is reserved for prompts that explicitly ask to classify, label, or categorize.

The task type heuristic was also tightened to use term-aware matching instead of broad substring matching. This avoids accidental matches such as `outage` containing `tag` or `transcript` containing `script`.

## Tier and Complexity Derivation

The merged dataset trusts `d1`-`d5` as ground truth and derives tier from the standard Phase 1 formula:

```text
complexity_score =
  d1 * 0.35 +
  d2 * 0.20 +
  d3 * 0.20 +
  d4 * 0.15 +
  d5 * 0.10
```

Tier thresholds:

- `T1`: `complexity_score < 0.40`
- `T2`: `0.40 <= complexity_score < 0.70`
- `T3`: `complexity_score >= 0.70`

Complexity labels:

- `T1` -> `low`
- `T2` -> `medium`
- `T3` -> `high`

## Phase 1 Expansion for Merge

The Phase 1 final CSV only had:

- `id`
- `prompt`
- `phrasing_style`
- `domain`
- `d1` through `d5`

To merge it with the Phase 2 schema, new fields were derived for Phase 1 rows only in the merged output file. The original Phase 1 CSV was not changed.

### `intent`

Derived from `d1`, because Phase 2 requirements define intent as primarily correlated with Semantic Complexity:

- `d1 <= 0.25` -> `FACTUAL`
- `d1 == 0.50` -> `ANALYTICAL`
- `d1 == 0.75` -> `SYNTHETIC`
- `d1 == 1.00` -> `STRATEGIC`

### `task_type`

Derived heuristically from prompt text:

- code keywords such as `python`, `sql query`, `source code`, `code`, `function`, `script`, `debug`, `yaml`, `json`, `syntax error`, `stack trace`, `kubernetes manifest` -> `coding`
- summary keywords such as `summarize`, `summary`, `tl;dr`, `condense` -> `summarisation`
- classification keywords such as `classify`, `label`, `categorize` -> `classification`
- SPARQL/RDF/ontology/knowledge graph keywords -> `sparql_generation`
- formatting keywords such as `format`, `rewrite`, `rephrase`, `tone`, `style` -> `formatting`
- generation keywords such as `create`, `draft`, `write`, `generate`, `compose`, `build` -> `generation`
- reasoning keywords such as `compare`, `analyze`, `evaluate`, `assess`, `why`, `how` -> `reasoning`
- fallback -> `reasoning`

### `reasoning_chain_detected`

Derived from `d1`:

- `d1 >= 0.50` -> `True`
- `d1 < 0.50` -> `False`

### `research_signals`

Derived from `d4` plus keyword matching:

- `d4 == 0` -> `[]`
- `d4 > 0` -> scan the prompt for research/domain keywords.

Keyword groups include:

- `market_research`
- `competitive_analysis`
- `regulatory_compliance`
- `security`
- `cloud_infrastructure`
- `finops`
- `devops`
- `data_engineering`
- `ai_governance`
- `system_integration`
- `supply_chain`
- `hr_tech`
- `vendor_analysis`

If `d4 > 0` and no specific keyword matched, the fallback signal is:

```json
["external_research"]
```

### `confidence`

Set to blank/null for Phase 1 rows.

Reason: Phase 1 has no confidence label, and Phase 2 requirements say inference confidence should later be derived from model probability outputs rather than treated as a training label.

### `prompt_type`

Mapped approximately from Phase 1 `domain`.

Examples:

- `FinOps` -> `INFORMATIONAL`
- `DevOps` -> `INSTRUCTIONAL`
- `Security` -> `ANALYSIS_CRITIQUE`
- `Data Engineering` -> `DATA_EXTRACTION`
- `Competitive Intelligence` -> `COMPARISON`

### Other Phase 1 Derived Fields

- `task_description`: first 80 characters of the prompt.
- `expected_answer`: blank/null.
- `prompting_techniques`: blank/null.
- `original_complexity`: blank/null.
- `source`: `phase1`.

## Final Validation Summary

Merged dataset:

- Rows: `2273`
- Columns: `24`
- Phase 2 rows: `1447`
- Phase 1 rows: `826`

Tier counts:

- `T1`: `942`
- `T2`: `1018`
- `T3`: `313`

Intent counts:

- `ANALYTICAL`: `1306`
- `FACTUAL`: `413`
- `SYNTHETIC`: `380`
- `STRATEGIC`: `174`

Task type counts:

- `reasoning`: `1226`
- `generation`: `766`
- `summarisation`: `112`
- `classification`: `90`
- `coding`: `77`
- `formatting`: `2`
- `sparql_generation`: `0`

D-score coverage:

- `d1`: all valid score levels present
- `d2`: all valid score levels present
- `d3`: all valid score levels present
- `d4`: all valid score levels present
- `d5`: all valid score levels present

Other validation:

- Invalid intents: `0`
- Invalid task types: `0`
- Phase 2 duplicate prompts after cleaning: `0`
- Phase 1 duplicate prompts after verdict fix: `0`
- Merged duplicate prompts after verdict fix: `0`
- Phase 1 question-ending prompts assigned to `classification`: `0`
- Low-confidence Phase 2 rows retained and flagged: `36`

Known training note: `sparql_generation` has `0` samples in the merged dataset, so a model cannot learn that class unless samples are added later or the class is excluded from the trainable label set.

## Status

Dataset cleaning and merging is complete.

Next step is manual audit of the generated CSVs before any model training or architecture implementation.
