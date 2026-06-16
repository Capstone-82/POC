# Architecture 3 v4 — Final Verdict

## The Numbers: Excellent Progression 🟢🟢🟢

| Metric | v1 | v2 | v3 | **v4** |
|--------|-----|-----|-----|--------|
| Tier Accuracy | ~65% | 79.8% | 86.2% ±2.0% | **89.9% ±1.3%** |
| T1 Recall | — | 95% | 90% | **93%** |
| T2 Recall | — | 82% | 82% | **86%** |
| T3 Recall | — | 57% | 86% | **90%** |
| D1 R² | ~0.45 | 0.81 | 0.77 | **0.85** |
| D2 R² | ~0.35 | 0.50 | 0.57 | **0.75** |
| D3 R² | ~0.40 | 0.74 | 0.69 | **0.81** |
| D4 | rule 0.28 MAE | rule 0.28 | 0.53 R² | **0.75 R²** |
| D5 R² | ~0.35 | 0.60 | 0.65 | **0.85** |
| Dataset | 470 | 470 | 615 | **889** |

Every single dimension R² is now **>0.74**. That's a genuinely strong result. The rubric alignment (fixing D3/D5 names and features) clearly paid off — D5 went from 0.65 → 0.85 R².

---

## Can You Reach 95% Tier Accuracy?

### Short answer: **No, not with this architecture and dataset design.**

Here's the mathematical reality:

### The Boundary Problem Is Structural, Not a Bug

```
T2/T3 boundary [0.65, 0.75): 196 prompts
  → 100 are T2 (cs = 0.65–0.6999)
  → 96 are T3  (cs = 0.70–0.75)

T2 with cs ≥ 0.60: 159 prompts (vulnerable to T3 misclassification)
T3 with cs < 0.80: 112 prompts (vulnerable to T2 misclassification)
```

**196 prompts** sit in the T2/T3 boundary zone — that's **22% of your entire dataset**. These prompts have complexity scores separated by as little as **0.01**. A single dimension error of 0.25 on D1 shifts the composite score by `0.25 × 0.35 = 0.0875` — enough to flip the tier.

### The error math:

Current confusion matrix:
```
         T1   T2   T3
    T1 [266   13    6]    ← 19 errors
    T2 [ 12  288   33]    ← 45 errors
    T3 [  1   25  245]    ← 26 errors
                           Total: 90 errors / 889 = 10.1%
```

To reach 95%, you need ≤ **44 errors** out of 889. That means cutting errors in half.

- **T1 errors (19):** Already good. Maybe save 5.
- **T2→T3 errors (33):** These are T2 prompts over-routed as T3. Hard to fix without hurting T3 recall.
- **T3→T2 errors (25):** These are the dangerous ones. To fix these, you'd need D4/D5 to be near-perfect for boundary prompts.
- **T2→T1 errors (12):** Hard to eliminate without expanding T1 boundary.

Even if you **perfectly** fix T1 (save 5) + fix half of T3→T2 (save 12) + fix half of T2→T3 (save 16), you get:
`90 - 5 - 12 - 16 = 57 errors → 93.6%`

**93-94% is the realistic ceiling.** 95% would require near-perfect dimension prediction at the boundary, which is beyond what text features + small embeddings can reliably deliver.

### Why 95% is fundamentally hard:

1. **The formula is adversarial at boundaries.** A T2 prompt with scores (0.75, 0.75, 0.75, 0.25, 0.50) has cs=0.65 (T2). If the model predicts D4=0.50 instead of 0.25, cs jumps to 0.6875 — still T2 but barely. If D1 is also wrong by 0.25, cs=0.775 → T3. **Two quarter-point errors = wrong tier.**

2. **The dataset is AI-generated.** Even with careful rubric alignment, AI-generated labels introduce label noise at boundaries. A human labeler might disagree on whether a prompt is D3=0.50 vs D3=0.75 for output formality. This noise is **irreducible** — no model can learn it perfectly.

3. **MiniLM-L6-v2 embeddings are general-purpose.** They capture semantic similarity well but don't encode "output formality" or "context requirement" directly. Your hand-crafted features compensate, but 32 features can only encode so many rubric patterns.

---

## Can You Avoid the Boundary Confusion?

### Partially — here's what actually works:

#### What's already done right in v4:
- **Cost-sensitive T3 weighting (1.35x)** — good for production safety
- **Boundary sample weighting (1.15x)** — helps the model focus on hard cases
- **Direct tier classifier** — bypasses the dimension→score→tier pipeline entirely

#### What would NOT help much:
- **More boundary data** — you already have 196 prompts in the T2/T3 zone. Adding more doesn't help if the signal isn't distinguishable.
- **More features** — `has_cost_comparison` is already at zero importance. The useful signals are exhausted.
- **Deeper trees** — would overfit the training set without generalizing.

#### What COULD help (but with diminishing returns):

1. **Widen the buffer zone** — Instead of a hard cutoff at 0.70, use a soft boundary:
   ```
   if 0.65 ≤ cs ≤ 0.75: tier = "T2-T3" (uncertain, route conservatively to T3)
   ```
   This trades precision for safety. In production, you'd rather over-route than under-route.

2. **Confidence-based routing** — Use the tier classifier's probability outputs:
   ```python
   probs = tier_clf.predict_proba(X)  # [P(T1), P(T2), P(T3)]
   if max(probs) < 0.6:  # low confidence
       tier = "T3"  # default to premium (safe)
   ```

3. **A dedicated boundary binary classifier** — Train a separate model ONLY on prompts with cs ∈ [0.55, 0.85] to classify "T2 vs T3". This narrows the problem but needs careful training.

---

## What I'd Actually Do

### For this capstone: **Stop at v4.**

You have a **compelling four-iteration story**:

| Version | Accuracy | Key Insight |
|---------|---------|-------------|
| v1 | ~65% | Overfitting — 1:1 feature-to-sample ratio |
| v2 | 80% | PCA + hand-crafted features + ordinal classification |
| v3 | 86% | D4 via ML + dataset augmentation + two-stage prediction |
| v4 | **90%** | Rubric alignment + targeted class balancing + regularization |

Each version solved a **specific, diagnosed problem** with a **principled engineering fix**. That's a better story than "we kept tweaking until the number went up."

### The metrics tell a complete story:

- **Every dimension R² > 0.74** — the model genuinely learned all 5 rubric dimensions
- **90% tier accuracy with 5-fold CV** — robust, not overfit
- **All 4 test prompts classified correctly** with reasonable D-scores
- **Feature importance makes sense** — `has_report_package`, `structured_section_count`, `large_context_signal` are the top features, exactly what the rubric says matters

### Bottom line:

> **90% with honest 5-fold CV on a 889-sample synthetic dataset using a non-LLM pipeline is a very strong capstone result. 95% is not achievable with this architecture — the boundary math makes it structurally impossible without changing the tier formula or using a fundamentally different approach (like an LLM doing zero-shot classification). Stop here, document the journey, and present v4 as your final model.**
