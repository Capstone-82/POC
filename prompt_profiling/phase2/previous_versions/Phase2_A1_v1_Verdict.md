# Phase 2 Architecture A v1 — Verdict

## Summary

This is a solid v1 baseline. The architecture is correctly structured as a shared-backbone multi-head XGBoost system, matching the plan. All 9 heads train and produce valid outputs. The inference function returns the correct JSON schema with all required fields. The dataset fixes (deduplication → 2,273 rows) were applied.

---

## Results at a Glance

| Head | Accuracy | Macro F1 | Comment |
|------|----------|----------|---------|
| **Tier** | **79.8%** | 79.9% | Baseline — Phase 1 v4 hit 89.9% |
| **Intent** | **83.1%** | 77.7% | Good for v1 — STRATEGIC recall 65% is the weakest |
| **Task Type** | **77.8%** | 65.3% | Classification recall = 25%, formatting = 0 samples in val |
| **Reasoning Chain** | **90.6%** | 84.9% | Strong — but 77% True prevalence inflates accuracy |
| D1 | 68.6% | 66.9% | Weakest dimension |
| D2 | 74.1% | 58.2% | D2=0.0 recall = 29%, D2=0.25 recall = 33% |
| D3 | 75.0% | 64.3% | D3=0.50 dominates at 94% recall; others weak |
| D4 | 78.5% | 61.3% | D4=0.0 recall = 96%; D4=0.75 recall = 27% |
| D5 | 73.4% | 72.8% | D5=0.75 recall = 39%, D5=1.0 recall = 100% |

---

## What's Done Right ✅

1. **Architecture matches the plan** — shared MiniLM backbone → PCA(35) → hand-crafted features → StandardScaler → 9 independent XGBoost heads
2. **Dataset properly validated** — tier formula mismatches = 0, max score deviation ≈ 0
3. **Deduplication applied** — 2,273 rows (the 63 Phase 1 duplicates were removed)
4. **Hand-crafted features are well designed** — 21 features including role detection, step requests, code blocks, research terms, and output format signals
5. **Inference function complete** — returns all required fields: D1–D5, tier, intent, task_type, reasoning_chain, research_signals, confidence
6. **Research signals rule engine** — properly gated on D4 > 0 with 13 signal categories
7. **Both direct tier AND formula-derived tier compared** — good diagnostic practice
8. **Tier derivation formula verified** against dataset at load time

---

## 🔴 Issues to Fix

### Issue 1: Single Train/Val Split Instead of 5-Fold CV

**What:** The notebook uses a single 80/20 stratified split (1,818 train / 455 val). This means all metrics are measured on a single random split and could vary ±3–5% on a different seed.

**Why it matters:** Phase 1 v4 used 5-fold stratified CV, which gave robust mean ± std metrics. A single split is fine for a quick baseline but should not be the final evaluation.

**Fix for v2:** Switch to 5-fold stratified CV. Report mean ± std for all metrics.

---

### Issue 2: No Sample Weighting (Critical for Imbalanced Classes)

**What:** All 9 heads are trained with uniform sample weights. But the data has extreme imbalances:

| Target | Majority | Minority | Ratio |
|--------|----------|----------|-------|
| D4 | 0.0 (1,585) | 0.25 (115) | 13.8:1 |
| D3 | 0.50 (1,320) | 0.25 (121) | 10.9:1 |
| D2 | 0.50 (1,347) | 0.25 (140) | 9.6:1 |
| task_type | reasoning (1,226) | formatting (2) | 613:1 |
| intent | ANALYTICAL (1,306) | STRATEGIC (174) | 7.5:1 |

**Impact visible in results:**
- D4=0.75 recall = **27%** (the model barely tries to predict this class)
- D2=0.0 recall = **29%**
- D2=0.25 recall = **33%**
- classification task_type recall = **25%**
- STRATEGIC intent recall = **65%**

**Fix for v2:** Add `sample_weight=compute_sample_weight('balanced', y_train)` for all heads. This is the single highest-impact change you can make.

---

### Issue 3: No Two-Stage Tier-Augmented Features

**What:** Phase 1 v4 used a two-stage pipeline where the Stage 1 tier prediction was appended as an extra feature for the D1–D5 dimension classifiers. This tier-as-feature conditioning improved D-score accuracy significantly.

**Current v1:** All heads receive the same 56 features. The dimension heads don't benefit from knowing the predicted tier.

**Fix for v2:** After training the tier head, predict tier on the training data and append it as feature #57 for the D1–D5 heads:

```python
tier_pred_train = heads['tier'].predict(X_train)
X_train_aug = np.hstack([X_train, tier_pred_train.reshape(-1, 1)])
```

---

### Issue 4: Fewer Hand-Crafted Features Than Phase 1

**What:** v1 has 21 hand-crafted features. Phase 1 v4 had 32 features. Missing features that were high-importance in Phase 1:

| Phase 1 Feature | Importance Rank | Missing in v1 |
|-----------------|----------------|---------------|
| `has_report_package` | #1 | ❌ Yes |
| `structured_section_count` | #2 | ❌ Yes |
| `has_compliance` | #3 | ❌ Yes |
| `has_attachment` | #4 | ❌ Yes |
| `large_context_signal` | #7 | ❌ Yes |
| `cloud_providers_mentioned` | — | ❌ Yes |
| `domain_framework_count` | — | ❌ Yes |
| `phrasing_style` (one-hot) | — | ❌ Yes |

v1 has simpler term-counting features (`research_term_count`, `code_term_count`, etc.) which are less discriminative than the targeted keyword detectors from Phase 1.

**Fix for v2:** Port the Phase 1 hand-crafted feature extractor and merge it with the new Phase 2 features. Keep the new features (e.g., `has_role_prompt`, `has_step_request`, `has_code_block`) and add the missing Phase 1 features.

---

### Issue 5: Confidence = min(all head probas) Produces Very Low Values

**What:** The sample T3 prompt ("Design a multi-cloud GenAI governance architecture...") gets confidence = **0.4248**. This is because `min()` across 9 heads takes the worst-case probability. With 5 ordinal D-score heads (each predicting 5 classes), it's common for at least one head to have max probability ≈ 0.40.

**Impact:** In production, almost every prompt will have confidence < 0.6. The confidence signal becomes uninformative.

**Fix for v2:** Use a weighted average instead of min, or exclude D-score heads from confidence and only use the categorical heads:

```python
# Option A: weighted average
confidence = np.mean(confidences)

# Option B: only use tier + intent + task_type confidence
key_confidences = [tier_conf, intent_conf, task_type_conf]
confidence = np.mean(key_confidences)
```

---

## 🟡 Minor Observations

### `formatting` task_type has 2 samples

Only 2 rows in the entire dataset have `task_type=formatting`. The model cannot learn this class. In the validation set, 0 formatting samples appeared, so it was never evaluated. Either add more formatting samples or remove it from the valid types.

### `classification` task_type recall = 25%

Only 12 validation samples; 3 correct. This is partly due to the `? → classification` heuristic issue identified in the dataset verdict. The Phase 1 question prompts were wrongly labeled as `classification` — those are actually `reasoning`.

### D1 accuracy dropped vs Phase 1

Phase 1 v4 achieved D1 R² = 0.845, accuracy = 79.2%. v1 gets D1 accuracy = 68.6%. This is likely due to: (a) more diverse data making D1 harder, (b) missing Phase 1 features (`has_scope_words`, `action_verb_count`, `multi_stage_signal`), and (c) no two-stage conditioning.

### No `min_child_weight` in XGB config

Phase 1 v4 used `min_child_weight=5` to prevent overfitting on small classes. v1 omits this, defaulting to 1. This makes the model more prone to memorizing minority class patterns.

---

## Phase 1 v4 → Phase 2 A1 v1 Comparison

| Metric | Phase 1 v4 | Phase 2 A1 v1 | Delta | Explanation |
|--------|-----------|---------------|-------|-------------|
| Tier accuracy | 89.9% | 79.8% | **−10.1pp** | No sample weighting, no two-stage, fewer features |
| D1 accuracy | 79.2% | 68.6% | −10.6pp | Missing `has_scope_words`, `action_verb_count` |
| D2 accuracy | 74.0% | 74.1% | +0.1pp | Comparable |
| D3 accuracy | 76.2% | 75.0% | −1.2pp | Comparable |
| D4 accuracy | 82.7% | 78.5% | −4.2pp | Missing `has_market_terms`, `has_cost_comparison` |
| D5 accuracy | 79.6% | 73.4% | −6.2pp | Missing `has_attachment`, `large_context_signal` |
| Evaluation | 5-fold CV | Single split | — | v1 may be 3–5% overoptimistic |

The 10pp tier accuracy drop is the most concerning metric. Phase 1 achieved 89.9% with focused enterprise data. Phase 2 has broader domain diversity (harder problem) but also lost key features and training strategies.

---

## v2 Implementation Guide

Everything below should be implemented in `phase2_a1_v2.ipynb`. Changes are grouped by section.

---

### Change 1: Fix Dataset Labels Before Loading (Pre-Training)

Before training, fix the 29 Phase 1 question prompts that were wrongly assigned `task_type='classification'`. Add this right after loading the CSV:

```python
# Fix Phase 1 question prompts wrongly labeled as 'classification'
# The '? → classification' heuristic was too aggressive.
# Questions like "what is X?" and "why does Y?" are reasoning, not classification.
mask = (
    (df['source'] == 'phase1') &
    (df['prompt'].str.strip().str.endswith('?')) &
    (df['task_type'] == 'classification')
)
# Re-derive using keyword priority (code > summary > generation > reasoning)
for idx in df[mask].index:
    prompt_lower = df.loc[idx, 'prompt'].lower()
    if any(kw in prompt_lower for kw in ['python', 'sql', 'code', 'script', 'debug', 'yaml', 'kubernetes']):
        df.loc[idx, 'task_type'] = 'coding'
    elif any(kw in prompt_lower for kw in ['summarize', 'summary']):
        df.loc[idx, 'task_type'] = 'summarisation'
    else:
        df.loc[idx, 'task_type'] = 'reasoning'

print(f'Fixed {mask.sum()} misclassified question prompts')
```

Also handle the `formatting` class — it only has 2 samples:

```python
# Merge 'formatting' into 'generation' — 2 samples cannot be learned
df.loc[df['task_type'] == 'formatting', 'task_type'] = 'generation'
# Update VALID_TASK_TYPES to remove 'formatting'
VALID_TASK_TYPES = ['classification', 'generation', 'reasoning', 'coding', 'summarisation']
```

---

### Change 2: PCA Components → 40

```python
# v1: PCA(n_components=35)
# v2: increase to 40 — dataset grew from 889 → 2,273 samples
pca = PCA(n_components=40, random_state=RANDOM_STATE)
emb_pca = pca.fit_transform(embeddings)
print(f'PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}')
```

---

### Change 3: Hand-Crafted Features — 42 Total (32 Phase 1 + 10 New)

Replace the entire `handcrafted_features` function. The new version has 3 groups:

#### Group A: Phase 1 Features (32 features, port from v4)

These are the proven features. Copy the exact function from Phase 1 v4 notebook:

```python
# ── Text statistics (6) ──
'char_len', 'word_count', 'sentence_count',
'avg_word_len', 'unique_word_ratio', 'line_count'

# ── D5: Context Requirement (4) ──
'has_attachment'          # "uploaded", "attached", "provided file", "document below"
'provided_artifact_count' # count of "csv", "json", "pdf", "log", "yaml", "xlsx"
'large_context_signal'    # "across all", "entire", "all of our", "company-wide"
'multi_document_signal'   # "multiple", "all the", "each of the", "various"

# ── D3: Output Formality (4) ──
'has_formal_deliverable'  # "report", "brief", "proposal", "specification", "whitepaper"
'has_report_package'      # "appendix", "table of contents", "risk register", "executive summary"
'has_long_output_signal'  # "comprehensive", "detailed", "thorough", "in-depth"
'structured_section_count'# count matches of "executive summary", "timeline", "roadmap", etc.

# ── D1: Semantic Complexity (3) ──
'has_scope_words'         # "strategic", "cross-domain", "enterprise-wide", "synthesize"
'action_verb_count'       # "build", "design", "evaluate", "integrate", "optimize", "develop"
'multi_stage_signal'      # "phase", "stage", "step 1", "milestone", "sequentially"

# ── D2: Domain Specificity (4) ──
'has_compliance'          # "NIST", "HIPAA", "SOC2", "GDPR", "ISO 27001", "PCI-DSS"
'cloud_providers_mentioned'# count of "aws", "azure", "gcp", "oci"
'systems_mentioned'       # "Salesforce", "ServiceNow", "Jira", "Workday", "SAP", etc.
'domain_framework_count'  # "ITIL", "FinOps", "TOGAF", "OWASP", "DORA", etc.

# ── D4: Research Dependency (5) ──
'external_data_score'     # "market research", "industry report", "analyst", "third-party"
'has_time_reference'      # "2024", "this quarter", "FY25", "latest", "current"
'vendor_tool_count'       # count of specific vendor/tool names
'has_market_terms'        # "competitor", "market share", "TAM", "benchmark"
'has_cost_comparison'     # "pricing", "cost analysis", "TCO", "ROI", "showback"

# ── Boundary/Risk (3) ──
'has_comparison'          # "compare", "versus", "vs", "difference between"
'stakeholder_mentions'    # "CEO", "CTO", "board", "leadership", "management"
'risk_language'           # "risk", "threat", "vulnerability", "mitigation"

# ── Phrasing Style (3) ──
# One-hot encode from Phase 1 'phrasing_style' column (explicit/implicit/vague)
# For Phase 2 rows (no phrasing_style), set all three to 0
'phrasing_explicit', 'phrasing_implicit', 'phrasing_vague'
```

#### Group B: Phase 2 New Features (10 features)

These target the new heads (intent, task_type, reasoning_chain):

```python
# ── Intent / Reasoning Chain targeting ──
'has_role_prompt'         # "you are", "act as", "assume the role"
'has_step_request'        # "step-by-step", "first...then", "sequentially"
'has_chain_of_thought'    # "think through", "reason about", "let's think",
                          # "chain of thought", "walk me through"
'question_complexity'     # 0 = no question
                          # 1 = simple factual "what is", "define"
                          # 2 = analytical "why", "how", "compare"
                          # 3 = strategic "what should we", "recommend", "design a"
'multi_domain_count'      # count of distinct domain buckets mentioned
                          # buckets: cloud, finops, security, devops, data, ai, hr, supply_chain
                          # SYNTHETIC/STRATEGIC prompts mention 2+ domains

# ── Task Type targeting ──
'has_code_block'          # triple backticks in prompt
'has_output_format'       # "in JSON", "as a table", "format as", "CSV output"
'has_creative_language'   # "imagine", "creative", "story", "write a", "compose"
'has_classification_request' # "classify", "categorize", "tag", "label", "which category"
'enumeration_signal'      # "list", "top 5", "enumerate", "bullet points", "rank"
```

#### Implementation Notes

```python
def handcrafted_features(prompts, phrasing_styles=None):
    """
    Returns DataFrame with 42 features.
    phrasing_styles: Series/list aligned with prompts. None for Phase 2 rows.
    """
    rows = []
    for i, prompt in enumerate(prompts):
        text = str(prompt)
        lower = text.lower()
        words = re.findall(r'\b\w+\b', lower)

        row = {}

        # ── Group A: Phase 1 Features (32) ──
        # ... (copy Phase 1 v4 feature extraction logic here)

        # ── Group B: Phase 2 New Features (10) ──
        row['has_role_prompt'] = int(bool(re.search(r'\byou are\b|\bact as\b|\bassume the role\b', lower)))
        row['has_step_request'] = int(bool(re.search(r'\bstep[- ]by[- ]step\b|\bfirst\b.*\bthen\b|\bsequentially\b', lower)))
        row['has_chain_of_thought'] = int(bool(re.search(
            r'\bthink through\b|\breason about\b|\blet.s think\b|\bchain of thought\b|\bwalk me through\b', lower)))

        # question_complexity: 0/1/2/3
        if '?' not in text:
            row['question_complexity'] = 0
        elif any(kw in lower for kw in ['what should', 'recommend', 'design a', 'propose', 'strategy']):
            row['question_complexity'] = 3
        elif any(kw in lower for kw in ['why', 'how', 'compare', 'analyze', 'evaluate', 'assess']):
            row['question_complexity'] = 2
        else:
            row['question_complexity'] = 1

        # multi_domain_count
        domain_buckets = {
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'kubernetes'],
            'finops': ['finops', 'cost', 'budget', 'chargeback', 'showback'],
            'security': ['security', 'vulnerability', 'iam', 'zero trust', 'soc'],
            'devops': ['devops', 'ci/cd', 'pipeline', 'sre', 'deployment'],
            'data': ['data pipeline', 'etl', 'warehouse', 'lakehouse', 'spark'],
            'ai': ['ai', 'llm', 'genai', 'machine learning', 'model'],
            'hr': ['hr', 'employee', 'talent', 'workforce', 'recruiting'],
            'supply': ['supply chain', 'inventory', 'procurement', 'logistics'],
        }
        row['multi_domain_count'] = sum(
            1 for bucket_kws in domain_buckets.values()
            if any(kw in lower for kw in bucket_kws)
        )

        row['has_code_block'] = int('```' in text)
        row['has_output_format'] = int(bool(re.search(
            r'\bin json\b|\bas a table\b|\bformat as\b|\bcsv output\b|\bin yaml\b|\bas markdown\b', lower)))
        row['has_creative_language'] = int(any(kw in lower for kw in [
            'imagine', 'creative', 'story', 'write a', 'compose', 'fictional']))
        row['has_classification_request'] = int(any(kw in lower for kw in [
            'classify', 'categorize', 'tag', 'label', 'which category', 'sort into']))
        row['enumeration_signal'] = int(bool(re.search(
            r'\blist\b|\btop \d+\b|\benumerate\b|\bbullet point\b|\brank\b', lower)))

        # Phrasing style one-hot (Phase 1 only, Phase 2 = all zeros)
        ps = phrasing_styles[i] if phrasing_styles is not None and pd.notna(phrasing_styles[i]) else None
        row['phrasing_explicit'] = int(ps == 'explicit')
        row['phrasing_implicit'] = int(ps == 'implicit')
        row['phrasing_vague'] = int(ps == 'vague')

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)

# Call with phrasing style from dataset
hand_df = handcrafted_features(df['prompt'], df.get('phrasing_style'))
print(f'Hand-crafted features: {hand_df.shape[1]}')  # Should print 42
```

---

### Change 4: XGBoost Hyperparameters

```python
def make_xgb(num_classes):
    objective = 'binary:logistic' if num_classes == 2 else 'multi:softprob'
    params = dict(
        n_estimators=300,         # v1: 250 → v2: 300 (more data supports more trees)
        max_depth=3,              # keep same — prevents overfitting
        learning_rate=0.05,       # keep same
        subsample=0.85,           # keep same
        colsample_bytree=0.85,    # keep same
        reg_lambda=2.0,           # keep same
        min_child_weight=5,       # v1: 1 (default) → v2: 5 (prevents minority class memorization)
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        objective=objective,
    )
    if num_classes > 2:
        params['num_class'] = num_classes
    return XGBClassifier(**params)
```

---

### Change 5: Sample Weighting for All Heads

```python
from sklearn.utils.class_weight import compute_sample_weight

for name, num_classes in head_classes.items():
    model = make_xgb(num_classes)
    y_train = targets[name][train_idx]

    # Balanced sample weights — minority classes get higher weight
    sw = compute_sample_weight('balanced', y_train)

    model.fit(X_train_for_head, y_train, sample_weight=sw)
    heads[name] = model
```

---

### Change 6: Two-Stage Pipeline (Tier → D1–D5)

Train the tier and categorical heads on the base feature matrix, then augment features for D-score heads:

```python
# Stage 1: Train tier, intent, task_type, reasoning_chain on base features (X_train)
stage1_heads = ['tier', 'intent', 'task_type', 'reasoning_chain_detected']
for name in stage1_heads:
    model = make_xgb(head_classes[name])
    y_train = targets[name][train_idx]
    sw = compute_sample_weight('balanced', y_train)
    model.fit(X_train, y_train, sample_weight=sw)
    heads[name] = model

# Stage 2: Augment features with tier prediction for D1–D5 heads
tier_pred_train = heads['tier'].predict(X_train).reshape(-1, 1)
tier_pred_val = heads['tier'].predict(X_val).reshape(-1, 1)

X_train_aug = np.hstack([X_train, tier_pred_train])
X_val_aug = np.hstack([X_val, tier_pred_val])

for name in SCORE_COLS:
    model = make_xgb(head_classes[name])
    y_train = targets[name][train_idx]
    sw = compute_sample_weight('balanced', y_train)
    model.fit(X_train_aug, y_train, sample_weight=sw)
    heads[name] = model

# IMPORTANT: Update inference function to also augment with tier prediction
```

---

### Change 7: 5-Fold Stratified Cross-Validation

Replace the single train/val split with 5-fold CV:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_results = {name: {'accuracy': [], 'macro_f1': []} for name in head_classes}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, targets['tier'])):
    print(f'\n=== Fold {fold+1}/5 ===')
    X_train, X_val = X[train_idx], X[val_idx]
    heads = {}

    # Stage 1: categorical heads
    for name in ['tier', 'intent', 'task_type', 'reasoning_chain_detected']:
        model = make_xgb(head_classes[name])
        y_tr = targets[name][train_idx]
        sw = compute_sample_weight('balanced', y_tr)
        model.fit(X_train, y_tr, sample_weight=sw)
        heads[name] = model

    # Stage 2: D-score heads with tier augmentation
    tier_pred_tr = heads['tier'].predict(X_train).reshape(-1, 1)
    tier_pred_va = heads['tier'].predict(X_val).reshape(-1, 1)
    X_train_aug = np.hstack([X_train, tier_pred_tr])
    X_val_aug = np.hstack([X_val, tier_pred_va])

    for name in SCORE_COLS:
        model = make_xgb(head_classes[name])
        y_tr = targets[name][train_idx]
        sw = compute_sample_weight('balanced', y_tr)
        model.fit(X_train_aug, y_tr, sample_weight=sw)
        heads[name] = model

    # Evaluate all heads
    for name in head_classes:
        X_eval = X_val_aug if name in SCORE_COLS else X_val
        y_true = targets[name][val_idx]
        y_pred = heads[name].predict(X_eval)
        fold_results[name]['accuracy'].append(accuracy_score(y_true, y_pred))
        fold_results[name]['macro_f1'].append(f1_score(y_true, y_pred, average='macro'))

# Print summary
print('\n' + '='*60)
print('5-Fold CV Results (mean ± std)')
print('='*60)
for name in head_classes:
    acc = fold_results[name]['accuracy']
    f1 = fold_results[name]['macro_f1']
    print(f'{name:30s} Acc: {np.mean(acc):.4f} ± {np.std(acc):.4f}  '
          f'F1: {np.mean(f1):.4f} ± {np.std(f1):.4f}')
```

---

### Change 8: Confidence Calculation

Replace `min()` with weighted average of key heads only:

```python
def predict_prompt(prompt):
    X_one = build_features_for_prompts([prompt])

    # Stage 1 predictions
    direct_tier = label_encoders['tier'].inverse_transform(heads['tier'].predict(X_one))[0]
    intent = label_encoders['intent'].inverse_transform(heads['intent'].predict(X_one))[0]
    task_type = label_encoders['task_type'].inverse_transform(heads['task_type'].predict(X_one))[0]
    reasoning_chain = bool(heads['reasoning_chain_detected'].predict(X_one)[0])

    # Stage 2: augment with tier for D-score prediction
    tier_enc = heads['tier'].predict(X_one).reshape(-1, 1)
    X_one_aug = np.hstack([X_one, tier_enc])

    predicted_dims = {}
    for col in SCORE_COLS:
        pred_class = int(heads[col].predict(X_one_aug)[0])
        predicted_dims[col] = class_to_score[pred_class]

    # Confidence: weighted average of key head max probabilities
    # Exclude D-score heads — they have 5 classes so max_proba is naturally lower
    key_confidences = [
        max_probability(heads['tier'], X_one),
        max_probability(heads['intent'], X_one),
        max_probability(heads['task_type'], X_one),
        max_probability(heads['reasoning_chain_detected'], X_one),
    ]
    confidence = float(np.mean(key_confidences))

    # ... rest of result building
```

---

### v2 Feature Matrix Summary

```
v1:  MiniLM(384d) → PCA(35)  + 21 hand-crafted  = 56 features
v2:  MiniLM(384d) → PCA(40)  + 42 hand-crafted  = 82 features
                                                    + 1 tier (Stage 2 only) = 83 for D-score heads
```

---

### v2 Expected Targets

| Head | v1 Result | v2 Target | Key Fix |
|------|-----------|-----------|---------|
| **Tier** | 79.8% | **86–88%** | Sample weighting + Phase 1 features + two-stage |
| **Intent** | 83.1% | **85–88%** | Sample weighting + `question_complexity` + `multi_domain_count` |
| **Task Type** | 77.8% | **80–84%** | Label fixes + `has_output_format` + `has_classification_request` |
| **Reasoning Chain** | 90.6% | **91–93%** | `has_chain_of_thought` + `has_step_request` |
| **D1** | 68.6% | **76–80%** | `has_scope_words` + `action_verb_count` + tier-augmented |
| **D4** | 78.5% | **82–85%** | `external_data_score` + `has_market_terms` + sample weighting |
| **D5** | 73.4% | **78–82%** | `has_attachment` + `large_context_signal` + tier-augmented |

### Checklist

- [ ] Fix 29 question→classification labels
- [ ] Merge `formatting` into `generation`
- [ ] PCA = 40
- [ ] 42 hand-crafted features (32 Phase 1 + 10 new)
- [ ] `min_child_weight=5`, `n_estimators=300`
- [ ] `compute_sample_weight('balanced')` for all heads
- [ ] Two-stage: tier → D-score augmentation
- [ ] 5-fold stratified CV
- [ ] Confidence = mean of key head probabilities
- [ ] Update inference function for two-stage pipeline
