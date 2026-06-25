# V4 Dataset Audit

## Scope

This is a separate v4 working dataset. The original team handoff dataset is not modified.

## Counts

- Base rows: `2273`
- Added rows: `100`
- Final rows: `2373`
- Existing research signal cells normalized: `111`

## Source Counts
```text
source
phase2    1447
phase1     826
aug_v4     100
```

## Tier Counts
```text
tier
T1     950
T2    1060
T3     363
```

## Task Type Counts
```text
task_type
reasoning            1270
generation            782
summarisation         120
classification         98
coding                 85
formatting             10
sparql_generation       8
```

## Intent Counts
```text
intent
ANALYTICAL    1338
FACTUAL        421
SYNTHETIC      410
STRATEGIC      204
```

## Validation

- Duplicate prompts: `0`
- Tier formula mismatches: `0`
- Original rows retained: `2273`
- Augmentation share: `4.21%`
- T2/T3 boundary rows: `267`

## V4 Evaluation Requirement

Report metrics on all rows, original rows only, phase2 rows only, phase1 rows only, and aug_v4 rows separately.
