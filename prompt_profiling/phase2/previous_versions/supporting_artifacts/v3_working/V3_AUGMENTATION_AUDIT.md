# V3 Augmentation Audit

## Files

- Augmented dataset: `prompt_classifier_phase2_v3_augmented.csv`
- Augmented rows only: `prompt_classifier_phase2_v3_augmented_rows_only.csv`

## Scope

The original team handoff dataset was not modified. This is a v3 working dataset.

## Counts

- Original rows: `2273`
- Existing rows after research-signal normalization: `2273`
- Added rows: `360`
- Final rows: `2633`
- Existing research_signal cells normalized: `111`

## Source Counts

```text
source
phase2    1447
phase1     826
aug_v3     360
```

## Tier Counts

```text
tier
T1     982
T2    1208
T3     443
```

## Task Type Counts

```text
task_type
reasoning            1260
generation            821
summarisation         166
classification        145
coding                131
formatting             56
sparql_generation      54
```

## Intent Counts

```text
intent
ANALYTICAL    1466
SYNTHETIC      460
FACTUAL        453
STRATEGIC      254
```

## D-Score Coverage

```text
d1: {0.0: 468, 0.25: 127, 0.5: 908, 0.75: 891, 1.0: 239}
```

```text
d2: {0.0: 120, 0.25: 140, 0.5: 1461, 0.75: 664, 1.0: 248}
```

```text
d3: {0.0: 298, 0.25: 143, 0.5: 1314, 0.75: 670, 1.0: 208}
```

```text
d4: {0.0: 1728, 0.25: 145, 0.5: 291, 0.75: 262, 1.0: 207}
```

```text
d5: {0.0: 1051, 0.25: 329, 0.5: 849, 0.75: 244, 1.0: 160}
```

## Validation

- Duplicate prompts: `0`
- Tier formula mismatches: `0`
- D4=0 with non-empty research_signals: `0`
- D4>0 with empty research_signals: `0`
