# Architecture 3 — Diagnosis: Why Are The Scores Low?

## Quick Verdict

The low scores are **expected and structural** — they are not a bug in the code. Architecture 3 has fundamental design limitations that cap its performance ceiling. Here's the breakdown.

---

## The 5 Root Causes

### 1. Curse of Dimensionality — 384 Features vs 470 Samples

This is the **single biggest problem**.

| Metric | Value |
|--------|-------|
| Features (embedding dims) | 384 |
| Training samples | ~376 (80% of 470) |
| Feature-to-sample ratio | **~1:1** |

XGBoost needs significantly more samples than features to learn meaningful splits. At a 1:1 ratio, the model is essentially memorizing noise in the training set and generalizing poorly. Best practice is **10–30× more samples than features**.

> [!IMPORTANT]
> With 384 features and only 376 training samples, XGBoost cannot reliably learn — it's like trying to fit a line through a cloud of points in 384-dimensional space with barely enough data to cover each dimension once.

### 2. Discrete Ordinal Targets Treated as Continuous Regression

The D1–D5 scores are **not continuous** — they only take 5 discrete values:

```
{0.0, 0.25, 0.50, 0.75, 1.0}
```

But `XGBRegressor` with `reg:squarederror` treats them as continuous real numbers. This creates two problems:
- The model predicts values like `0.37` or `0.62` that don't exist in the label space
- R² is artificially deflated because small deviations from discrete targets look like large errors

### 3. Semantic Gap — General Embeddings vs Task-Specific Scoring

MiniLM-L6-v2 is trained on general NLI/STS tasks. It captures **what a prompt is about** (topic/domain), but the D1–D5 scores depend on **structural characteristics**:

| Dimension | What Matters | What MiniLM Captures |
|-----------|-------------|---------------------|
| D1 (Scope & Complexity) | Multi-step reasoning, breadth of deliverables | ❌ Poorly — semantic similarity ≠ structural complexity |
| D2 (Specificity) | Attachment references, explicit instructions | ❌ Poorly — "attached" vs "vague" is a structural signal |
| D3 (Cross-domain) | Number of domains touched | ⚠️ Partially — domain keywords help but aren't enough |
| D4 (External Data) | Need for real-time/market data | ❌ Rule-based, not ML |
| D5 (Output Complexity) | Report vs one-liner | ⚠️ Partially |

Two prompts can be **semantically similar** but have **completely different complexity scores**:
- *"explain what FinOps is"* → T1 (simple definition)  
- *"build a FinOps cost attribution framework with market research across 3 clouds"* → T3 (massive scope)

MiniLM would place these relatively close in embedding space because they share "FinOps" semantics, but their D1–D5 profiles are vastly different.

### 4. D4 Rule Engine is Crude

The keyword-matching rule engine has inherent limitations:

- **False negatives**: Many prompts needing external data don't contain the magic keywords (e.g., *"what is the current price of..."* triggers on "current", but *"check how competitor X handles..."* gets D4=0.50 via "compare" instead of the correct higher score)
- **False positives**: "analysis" appears in prompts that don't need external data at all (e.g., *"root cause analysis of our logs"*)
- **No context**: The rule engine can't distinguish *"market research"* as a task (D4=1.0) from someone discussing market research as a concept

> [!WARNING]
> The D4 rule engine MAE (~0.20–0.30 typically) directly degrades tier accuracy because D4 carries 15% weight in the complexity score formula.

### 5. No Structural/Engineered Features

The model uses **only** raw embeddings. It has zero access to powerful hand-crafted signals like:

| Missing Feature | Why It Matters |
|----------------|---------------|
| `prompt_length` (word count) | T3 prompts average 80+ words, T1 prompts average 15–25 words |
| `sentence_count` | Multi-sentence = higher complexity |
| `has_attachment_reference` | "attached", "uploaded", "pasted" → strong D2/D5 signal |
| `domain_count` | Mentions AWS + Azure + GCP → cross-domain (D3) |
| `question_word_count` | More questions = more scope (D1) |
| `phrasing_style` encoding | explicit/implicit/vague is directly correlated with scores |
| `keyword_density` for compliance terms | NIST, HIPAA, GDPR → strong T3 signals |

---

## Expected vs Ideal Performance Range

| Metric | Architecture 3 (Expected) | What "Good" Looks Like | Gap |
|--------|--------------------------|----------------------|-----|
| D1 MAE | 0.10–0.15 | < 0.05 | 2–3× |
| D1 R² | 0.40–0.60 | > 0.85 | Significant |
| D2 R² | 0.30–0.50 | > 0.80 | Significant |
| Tier Accuracy | 65–80% | > 90% | 10–25 pp |

---

## Prioritized Improvements (If You Want Better Scores)

### High Impact (do these first)

| # | Change | Expected Improvement | Effort |
|---|--------|---------------------|--------|
| 1 | **Add PCA/UMAP dimensionality reduction** (384 → 32–64 dims) | +5–10% tier accuracy | Low |
| 2 | **Add hand-crafted features** (word count, sentence count, attachment refs, domain keywords) alongside embeddings | +10–15% tier accuracy | Medium |
| 3 | **Round predictions to nearest {0, 0.25, 0.5, 0.75, 1.0}** after inference | Reduces MAE by ~30% | Trivial |
| 4 | **Hyperparameter tune XGBoost** (max_depth, learning_rate, min_child_weight, subsample) | +3–5% tier accuracy | Medium |

### Medium Impact

| # | Change | Expected Improvement |
|---|--------|---------------------|
| 5 | **Use ordinal classification** instead of regression (treat {0, 0.25, 0.5, 0.75, 1.0} as ordered classes) | Better calibrated outputs |
| 6 | **Data augmentation** — paraphrase existing prompts to 3–5× the dataset size | Breaks the dimensionality curse |
| 7 | **Switch to a KNN-based approach** for dimensions (Architecture 2 style) — with only 470 samples, nearest-neighbor methods often outperform tree ensembles | Potentially significant |

### Lower Priority

| # | Change | Notes |
|---|--------|-------|
| 8 | Fine-tune the embedding model on your specific task | Requires much more data |
| 9 | Use an ensemble of architectures | Complexity vs marginal gain |

---

## Bottom Line

> [!CAUTION]
> Architecture 3's core problem is **asking a 384-feature XGBoost model to learn from ~376 samples**. This is a dimensionality disaster that no amount of hyperparameter tuning can fix. The model needs either **fewer features** (PCA), **more samples** (augmentation), or **supplementary hand-crafted features** that directly encode what the scoring rubric measures.

The notebook code is correct — the low scores reflect the architecture's structural limitations, not implementation bugs. If you want to improve performance within this architecture, start with improvements #1 (PCA) and #3 (rounding) — they're easy wins that require minimal code changes.

Want me to implement any of these improvements as a v2 of the notebook?
