# Architecture 3 v4 - Verdict

## Overall Assessment: Target Reached

v4 is a strong improvement over v3 for the goal that matters most: tier accuracy.

| Metric | v3 | v4 | Change |
|---|---:|---:|---:|
| Tier Accuracy | 86.2% +/- 2.0% | 89.88% +/- 1.28% | +3.68 pp |
| T1 Recall | 90% | 93% | +3 pp |
| T2 Recall | 82% | 86% | +4 pp |
| T3 Recall | 86% | 90% | +4 pp |
| Dataset Size | 615 | 889 | +274 |
| Feature-to-sample ratio | 1:11.4 | 1:13.3 | Better |

## v4 Results

| Dimension | Rubric Label | MAE | RMSE | R2 | Accuracy |
|---|---|---:|---:|---:|---:|
| D1 | Semantic Complexity | 0.0582 | 0.1328 | 0.8455 | 79.19% |
| D2 | Domain Specificity | 0.0740 | 0.1512 | 0.7524 | 74.01% |
| D3 | Output Formality | 0.0675 | 0.1455 | 0.8110 | 76.15% |
| D4 | Research Dependency | 0.0689 | 0.1903 | 0.7455 | 82.68% |
| D5 | Context Requirement | 0.0554 | 0.1268 | 0.8490 | 79.64% |

## Global Tier Confusion Matrix

```text
     T1   T2   T3
T1  266   13    6
T2   12  288   33
T3    1   25  245
```

Classification report:

| Tier | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T1 | 0.95 | 0.93 | 0.94 | 285 |
| T2 | 0.88 | 0.86 | 0.87 | 333 |
| T3 | 0.86 | 0.90 | 0.88 | 271 |

Overall accuracy: 0.90 on 889 cross-validated predictions.

## What Worked

1. The targeted dataset expansion worked. The previous minority-class gaps were substantially reduced:
   - D4=0.50 increased from 28 to 124.
   - D5=1.00 increased from 20 to 100.
   - D5=0.75 increased from 55 to 100.
   - D2=0.00 increased from 23 to 100.

2. Rubric-aligned D3/D5 features worked. The strongest hand-crafted features are now highly aligned to the document:
   - `has_report_package`
   - `structured_section_count`
   - `large_context_signal`
   - `multi_document_signal`
   - `provided_artifact_count`

3. T3 recall improved to 90%, which is important because under-routing complex prompts is the most damaging production error.

4. D2 improved materially. In v3, D2 R2 was around 0.57; in v4, D2 R2 is 0.7524.

5. D5 is now strong. Context requirement was previously conceptually muddled with output complexity; v4 gives it direct features and reaches R2=0.8490.

## Remaining Weaknesses

1. T2 -> T3 remains the largest error bucket.

There are 33 T2 prompts classified as T3. This is not disastrous because over-routing T2 to premium is safer than under-routing T3 to a weaker model, but it will increase cost.

2. T3 -> T2 still exists.

There are 25 T3 prompts classified as T2. This is the main remaining production risk.

3. Boundary-aware tier policy did not improve over Stage 1 in the current run.

Stage 1 tier accuracy and final tier accuracy are both 89.88%. That means the direct tier classifier is already dominating correctly, or the dimension-derived tier is not adding enough independent signal.

4. `has_cost_comparison` has zero importance.

It can be removed or rewritten. The signal is probably already captured by `has_market_terms`, `external_data_score`, and `has_time_reference`.

## Recommendation

Use v4 as the new baseline. It is now a credible 90%-class tiering model for this synthetic POC.

Next highest-ROI step:

1. Add a small error-analysis cell that prints all misclassified prompts by fold, especially T3 -> T2 and T2 -> T3.
2. Manually audit 50-60 boundary examples around complexity scores 0.65-0.75.
3. Either tune the final tier threshold/policy for T3 recall or add 50 more hard T3 boundary examples.

Do not expand the dataset broadly anymore. The next improvement should be driven by actual misclassified examples.
