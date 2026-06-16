# Dataset v4 Audit

## Summary

- Base dataset: 615 rows
- Added targeted rows: 274 rows
- Final dataset: 889 rows
- Score values validated against {0.0, 0.25, 0.50, 0.75, 1.0}
- No duplicate ids
- No nulls in required columns

## Tier Distribution

| value | v2 | v4 |
| --- | --- | --- |
| T1 | 208 | 285 |
| T2 | 216 | 333 |
| T3 | 191 | 271 |

## Dimension Distributions

### D1

| value | v2 | added | v4 |
| --- | --- | --- | --- |
| 0.0 | 110.0 | 53.0 | 163.0 |
| 0.25 | 81.0 | 24.0 | 105.0 |
| 0.5 | 110.0 | 45.0 | 155.0 |
| 0.75 | 212.0 | 96.0 | 308.0 |
| 1.0 | 102.0 | 56.0 | 158.0 |

### D2

| value | v2 | added | v4 |
| --- | --- | --- | --- |
| 0.0 | 23.0 | 77.0 | 100.0 |
| 0.25 | 130.0 | 0.0 | 130.0 |
| 0.5 | 204.0 | 45.0 | 249.0 |
| 0.75 | 166.0 | 96.0 | 262.0 |
| 1.0 | 92.0 | 56.0 | 148.0 |

### D3

| value | v2 | added | v4 |
| --- | --- | --- | --- |
| 0.0 | 106.0 | 53.0 | 159.0 |
| 0.25 | 94.0 | 24.0 | 118.0 |
| 0.5 | 132.0 | 45.0 | 177.0 |
| 0.75 | 191.0 | 96.0 | 287.0 |
| 1.0 | 92.0 | 56.0 | 148.0 |

### D4

| value | v2 | added | v4 |
| --- | --- | --- | --- |
| 0.0 | 305.0 | 77.0 | 382.0 |
| 0.25 | 115.0 | 0.0 | 115.0 |
| 0.5 | 28.0 | 96.0 | 124.0 |
| 0.75 | 107.0 | 28.0 | 135.0 |
| 1.0 | 60.0 | 73.0 | 133.0 |

### D5

| value | v2 | added | v4 |
| --- | --- | --- | --- |
| 0.0 | 184.0 | 77.0 | 261.0 |
| 0.25 | 217.0 | 72.0 | 289.0 |
| 0.5 | 139.0 | 0.0 | 139.0 |
| 0.75 | 55.0 | 45.0 | 100.0 |
| 1.0 | 20.0 | 80.0 | 100.0 |

## Boundary Coverage

- T1/T2 boundary window [0.35, 0.45): 87 rows
- T2/T3 boundary window [0.65, 0.75): 196 rows

## Notes

- D3 is treated as Output Formality, matching the source rubric.
- D5 is treated as Context Requirement, matching the source rubric.
- D4=0.50 rows are written as multi-document/provided-artifact analysis, not live web retrieval.
- D5=1.00 rows explicitly describe very large uploaded corpora or multi-document injection.
- Existing v2 rows were preserved; v4 adds targeted rubric-aligned coverage rather than rewriting old labels.
