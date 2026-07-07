# Phase 3 — Plan to Reach 90%+ Tier Accuracy

## Current State

| Metric | Value |
|--------|-------|
| **Blended Tier Accuracy** | **81.29% ± 1.59%** |
| Dataset | 2,421 prompts |
| Architecture | MiniLM → PCA(40) → 50 handcrafted → 9 XGBoost heads |
| PCA experiment | PCA 40→50 gave 0% improvement — confirmed saturated |

## Why You're Stuck at 81%

The boundary analysis reveals the **structural ceiling**:

| Zone | Samples | % | What Happens |
|------|---------|---|-------------|
| **T1/T2 boundary** (cs 0.35–0.44) | **589** | **24.3%** | A single D1 or D2 error flips the tier |
| **T2/T3 boundary** (cs 0.65–0.74) | **276** | **11.4%** | Same — one D-score step = tier flip |
| **Total boundary zone** | **865** | **35.7%** | More than 1/3 of your data sits where errors are most costly |
| Safe core (unambiguous) | 1,530 | 63.2% | Model gets ~95% of these right |

**The model is ~95% accurate on safe-zone prompts and ~55% accurate on boundary prompts.** That averages to ~81%. To hit 90%, you need boundary accuracy to reach ~75–80%.

### Three Root Causes

1. **Embedding weakness** — MiniLM-L6 is a 22M parameter model with 384d embeddings. It cannot distinguish "compare AWS vs Azure pricing" (D1=0.50, ANALYTICAL) from "design a multi-cloud cost governance strategy comparing AWS, Azure, and GCP" (D1=0.75, SYNTHETIC). They're too similar in 384d space.

2. **Label noise** — 574 prompts (23.7%) have D1 ↔ intent misalignment. Example: D1=0.75 but intent=ANALYTICAL (should be SYNTHETIC). These contradictory labels train the model to be confused at boundaries.

3. **Boundary data scarcity** — Phase 2 raw data only contributed **42 T3 samples**. The T2/T3 boundary zone has 276 prompts, but many are Phase 1 prompts (enterprise-specific). The model lacks diversity of boundary examples.

---

## The 5 Changes (In Priority Order)

---

### Change 1: Upgrade Embeddings — BGE-base-en-v1.5

**Expected gain: +3–5pp tier accuracy**

MiniLM-L6 (22M params, 384d) is a lightweight model designed for speed, not accuracy. For your use case (2,421 prompts, offline training, batch inference), you can afford a larger embedding model.

| Model | Params | Dim | MTEB Score | Inference Speed |
|-------|--------|-----|------------|-----------------|
| MiniLM-L6-v2 (current) | 22M | 384 | 56.3 | 14k sent/s |
| **BAAI/bge-base-en-v1.5** | **109M** | **768** | **63.6** | 4k sent/s |
| intfloat/e5-large-v2 | 335M | 1024 | 64.0 | 1.5k sent/s |
| BAAI/bge-large-en-v1.5 | 335M | 1024 | 64.1 | 1.5k sent/s |

**My recommendation: `BAAI/bge-base-en-v1.5`** — best accuracy/speed tradeoff. The 768d embeddings capture finer semantic distinctions (critical for D1 boundary discrimination). It's 5x slower than MiniLM but still fast enough for your use case (~600 prompts/second on a T4 GPU).

> [!IMPORTANT]
> BGE models require a query prefix for best performance. Add `"Represent this sentence: "` before each prompt during encoding.

```python
from sentence_transformers import SentenceTransformer

# v2: MiniLM (384d)
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# v3: BGE-base (768d) — significantly stronger semantic embeddings
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# BGE models benefit from a query prefix
prompts_for_encoding = ["Represent this sentence: " + p for p in prompts]
embeddings = embedding_model.encode(
    prompts_for_encoding,
    batch_size=32,  # smaller batch for larger model
    show_progress_bar=True,
    normalize_embeddings=True,
)
print('Embedding shape:', embeddings.shape)  # (2421, 768)
```

**PCA adjustment:** With 768d input, increase PCA to **60 components** (was 40 for 384d). This captures the same variance fraction from a richer embedding space.

```python
pca = PCA(n_components=60, random_state=RANDOM_STATE)
```

**Why this works:** The boundary problem is fundamentally a semantic discrimination problem. "Compare AWS vs Azure" and "Design a multi-cloud governance architecture comparing AWS, Azure, and GCP" need different D1 scores (0.50 vs 0.75). MiniLM maps them to nearly identical vectors. BGE-base will place them further apart in 768d space, giving XGBoost more room to draw the boundary.

---

### Change 2: Fix D1 ↔ Intent Label Misalignment (574 rows)

**Expected gain: +2–4pp tier accuracy**

23.7% of your data has contradictory labels. The D1→intent mapping should be:

| D1 | Expected Intent |
|----|----------------|
| 0.00 | FACTUAL |
| 0.25 | FACTUAL |
| 0.50 | ANALYTICAL |
| 0.75 | SYNTHETIC |
| 1.00 | STRATEGIC |

But 574 rows violate this. Examples of misalignment:

- D1=0.75 + intent=ANALYTICAL (should be SYNTHETIC) — 194 rows
- D1=0.50 + intent=FACTUAL (should be ANALYTICAL) — ~80 rows
- D1=0.00 + intent=ANALYTICAL (should be FACTUAL) — ~50 rows

**These contradictions directly hurt tier accuracy** because the intent head and D1 head learn conflicting signals from the same embedding. When the model is uncertain at a boundary, contradictory labels push it to flip randomly.

**Fix approach:** For each misaligned row, decide which is correct — the D1 score or the intent — based on the prompt text. Create a review script:

```python
# Step 1: Identify misaligned rows
intent_to_d1 = {
    'FACTUAL': [0.0, 0.25],
    'ANALYTICAL': [0.5],
    'SYNTHETIC': [0.75],
    'STRATEGIC': [1.0],
}

df['d1_intent_aligned'] = df.apply(
    lambda r: r['d1'] in intent_to_d1.get(r['intent'], []), axis=1
)
misaligned = df[~df['d1_intent_aligned']]
print(f"Misaligned: {len(misaligned)}")

# Step 2: For each, decide whether to fix D1 or fix intent
# RULE: Trust D1 (it's the rubric ground truth) and fix intent to match
df.loc[df['d1'].isin([0.0, 0.25]) & ~df['d1_intent_aligned'], 'intent'] = 'FACTUAL'
df.loc[(df['d1'] == 0.5) & ~df['d1_intent_aligned'], 'intent'] = 'ANALYTICAL'
df.loc[(df['d1'] == 0.75) & ~df['d1_intent_aligned'], 'intent'] = 'SYNTHETIC'
df.loc[(df['d1'] == 1.0) & ~df['d1_intent_aligned'], 'intent'] = 'STRATEGIC'
```

> [!WARNING]
> This is a bulk fix. Some of those 574 rows may have correct intent but wrong D1. For the highest quality, you should manually review at least the boundary-zone misaligned rows (~200 rows). But the bulk fix is still better than leaving contradictions.

---

### Change 3: Error-Driven Boundary Augmentation (400–600 new prompts)

**Expected gain: +2–3pp tier accuracy**

The boundary zones need more diverse training examples. Current boundary coverage:

| Zone | Current | Needed | Gap |
|------|---------|--------|-----|
| T1/T2 boundary (cs 0.375) | 354 prompts at cs=0.375 (all identical score) | 200 diverse | Need variety, not volume |
| T2/T3 boundary (cs 0.65–0.74) | 276 prompts | 400+ | Need +150 diverse T3-leaning prompts |
| D1=0.50 vs D1=0.75 | This is the single hardest discrimination | +100 contrastive pairs | Need matched pairs |
| Phase 2-style T3 prompts | Only 42 from Phase 2 raw | 150+ | Phase 2 T3 is severely underrepresented |

**Augmentation strategy:** Generate prompts in **contrastive pairs** — one prompt that's clearly D1=0.50 and a similar prompt that's clearly D1=0.75. This teaches the model what the boundary looks like.

Example pair:
```
D1=0.50 (ANALYTICAL): "Compare the cost of AWS S3 vs Azure Blob Storage for storing 10TB of data"
D1=0.75 (SYNTHETIC): "Design a multi-cloud storage architecture that optimizes cost across AWS S3, 
                       Azure Blob, and GCP Cloud Storage, factoring in egress fees, 
                       data residency requirements, and disaster recovery"
```

**Target augmentation:**
- 100 contrastive D1=0.50/0.75 pairs (200 prompts)
- 100 T2/T3 boundary prompts (cs 0.65–0.74)
- 100 Phase 2-style T3 prompts (general domain, not just enterprise)
- 100 D2 boundary prompts (D2=0.50 vs D2=0.75)

Total: ~500 new prompts → dataset grows to ~2,900

Use the same LLM-generation approach from Phase 1 — specify target D-scores and verify.

---

### Change 4: Retrieval-Augmented Features (FAISS Nearest Neighbor)

**Expected gain: +1–3pp tier accuracy**

Add a new feature: **"what did similar prompts score?"** For each prompt, find the K nearest neighbors in the training set and use their labels as additional features.

```python
import faiss

# Build FAISS index on training embeddings
def build_faiss_index(train_embeddings):
    d = train_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (embeddings are normalized)
    index.add(train_embeddings.astype(np.float32))
    return index

# For each prompt, retrieve K=5 nearest neighbors and compute:
# - mean tier of neighbors (0=T1, 1=T2, 2=T3)
# - mean D1 of neighbors
# - tier agreement ratio (what fraction of neighbors share the same tier)
def retrieval_features(embeddings, index, train_labels, K=5):
    distances, indices = index.search(embeddings.astype(np.float32), K + 1)
    # Skip self (index 0) if present in training
    features = []
    for i in range(len(embeddings)):
        neighbor_idx = indices[i][1:K+1]  # skip self
        neighbor_tiers = train_labels['tier'].iloc[neighbor_idx].values
        neighbor_d1 = train_labels['d1'].iloc[neighbor_idx].values

        tier_numeric = np.array([{'T1': 0, 'T2': 1, 'T3': 2}[t] for t in neighbor_tiers])
        features.append({
            'nn_tier_mean': tier_numeric.mean(),
            'nn_tier_std': tier_numeric.std(),
            'nn_d1_mean': neighbor_d1.mean(),
            'nn_d1_std': neighbor_d1.std(),
            'nn_tier_agreement': (tier_numeric == tier_numeric[0]).mean(),
            'nn_max_dist': distances[i][-1],
        })
    return pd.DataFrame(features)
```

> [!IMPORTANT]
> FAISS features must be computed inside each CV fold using only training data — never look up validation samples in the index. During inference, the index contains all training samples.

**Why this works:** When a prompt sits at a boundary, the classifier is uncertain. But if 4 out of 5 nearest neighbors are T2, that's strong evidence for T2. This is especially powerful for boundary prompts where the embedding captures semantic similarity that the classifier alone can't use.

---

### Change 5: Ordinal-Aware D-Score Prediction

**Expected gain: +1–2pp tier accuracy**

D-scores are ordinal (0.0 < 0.25 < 0.50 < 0.75 < 1.0), but XGBoost treats them as unrelated classes. A prediction of D1=0.25 when the truth is D1=0.50 is penalized the same as predicting D1=1.0 — but in reality, 0.25 is a much smaller error.

**Approach:** Use ordinal binary decomposition. Instead of one 5-class classifier, train 4 binary classifiers:

```
D1 >= 0.25?  (yes/no)
D1 >= 0.50?  (yes/no)
D1 >= 0.75?  (yes/no)
D1 >= 1.00?  (yes/no)
```

The final prediction is determined by how many thresholds are crossed:
- All no → D1 = 0.0
- First yes, rest no → D1 = 0.25
- First two yes → D1 = 0.50
- etc.

```python
def ordinal_predict(models, X):
    """Predict ordinal class from 4 binary threshold models."""
    predictions = np.zeros(len(X), dtype=int)
    for threshold_idx, model in enumerate(models):
        proba = model.predict_proba(X)[:, 1]  # probability of >= threshold
        predictions += (proba >= 0.5).astype(int)
    return predictions  # 0-4, maps to {0.0, 0.25, 0.50, 0.75, 1.0}
```

**Why this works:** When the true D1 is 0.50, the ordinal model gets "credit" for predicting 0.25 (correct on 1 out of 4 thresholds) instead of being fully wrong. This produces smoother predictions near boundaries and reduces the tier-flipping errors by 20–30%.

---

## Expected Cumulative Impact

| Change | Tier Accuracy Gain | Cumulative |
|--------|-------------------|------------|
| Baseline | — | **81.3%** |
| 1. BGE-base embeddings | +3–5pp | **84–86%** |
| 2. Fix D1↔intent labels | +2–4pp | **86–90%** |
| 3. Boundary augmentation | +2–3pp | **88–93%** |
| 4. FAISS retrieval features | +1–3pp | **89–94%** |
| 5. Ordinal D-scores | +1–2pp | **90–95%** |

> [!IMPORTANT]
> Gains are NOT strictly additive — they overlap. The realistic target after all 5 changes is **88–92% tier accuracy**. To guarantee 90+%, you likely need Changes 1–3 to all land well. Changes 4–5 provide the final push.

---

## Implementation Order

| Step | What | Effort | Run Alone First? |
|------|------|--------|-----------------|
| **Step 1** | Swap MiniLM → BGE-base, PCA=60 | 30 min | **Yes** — run 5-fold CV to measure the embedding gain in isolation |
| **Step 2** | Fix 574 misaligned labels | 1–2 hours | **Yes** — run on fixed dataset with BGE to see combined gain |
| **Step 3** | Generate 500 boundary prompts | 3–4 hours | **Yes** — run on augmented + fixed dataset |
| **Step 4** | Add FAISS retrieval features | 2–3 hours | Run after Steps 1–3 are settled |
| **Step 5** | Ordinal D-score prediction | 2–3 hours | Run after Steps 1–3 are settled |

**My recommendation:** Do Steps 1–3 first and measure. If you're at 88%+ after Step 3, Steps 4–5 should push you past 90%. If you're still at 85%, there may be deeper label quality issues that need manual review.

---

## What NOT To Do

| Don't | Why |
|-------|-----|
| Increase PCA beyond 60 | Already proven no ROI from PCA tuning |
| Add more handcrafted features | 50 is already saturated for 2.4K samples |
| Switch to neural network | Not enough data; XGBoost is still optimal at this scale |
| Stack XGB + CatBoost + LGBM | Marginal gain (~0.5pp) for 3x complexity |
| Fine-tune the embedding model | 2.4K samples will overfit a 109M param model |
