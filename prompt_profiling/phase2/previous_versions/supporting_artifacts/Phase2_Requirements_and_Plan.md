# Phase 2 — Requirements, Dataset Cleaning, and Architecture Proposals

---

## 1. New Requirements

### 1.1 Extended Inference Output

The engineering team requires 5 additional fields in the inference JSON, extending the Phase 1 output:

```json
{
  "d1": 0.75,  "d1_label": "Semantic Complexity",
  "d2": 0.50,  "d2_label": "Domain Specificity",
  "d3": 0.75,  "d3_label": "Output Formality",
  "d4": 0.25,  "d4_label": "Research Dependency",
  "d5": 0.50,  "d5_label": "Context Requirement",
  "complexity_score": 0.5625,
  "tier": "T2",

  "intent": "ANALYTICAL",
  "task_type": "reasoning",
  "reasoning_chain_detected": true,
  "research_signals": ["market_research", "competitive_analysis"],
  "confidence": 0.85
}
```

### 1.2 Field Definitions

| Field | Type | Valid Values | Description |
|-------|------|-------------|-------------|
| `intent` | string | `FACTUAL`, `ANALYTICAL`, `SYNTHETIC`, `STRATEGIC` | Prompt intent classification (4-class) |
| `task_type` | string | `classification`, `generation`, `reasoning`, `coding`, `summarisation`, `sparql_generation`, `formatting` | Task category (7-class) |
| `reasoning_chain_detected` | boolean | `true`, `false` | Whether the prompt requires chain-of-thought reasoning |
| `research_signals` | string[] | list of detected signals | Research domain signals detected in the prompt |
| `confidence` | float | 0.0 – 1.0 | Model confidence in the overall classification |

### 1.3 Intent Definitions

| Intent | Definition | Example |
|--------|-----------|---------|
| FACTUAL | Single known answer, factual recall only | "What is the idle timeout default for AWS Bedrock?" |
| ANALYTICAL | Multi-step reasoning within one domain | "Compare FinOps maturity levels 1 through 3" |
| SYNTHETIC | Cross-domain synthesis of multiple concepts | "Create a requirements doc for CoreStack–ClickUp integration" |
| STRATEGIC | Strategic analysis requiring novel synthesis or external research | "Design a multi-cloud GenAI governance architecture for a Fortune 500 firm" |

### 1.4 Relationship Between New Fields and Existing Dimensions

| New Field | Primary Correlation | Rationale |
|-----------|-------------------|-----------|
| `intent` | D1 (Semantic Complexity) | FACTUAL→D1≈0, STRATEGIC→D1≈1.0 |
| `task_type` | D3 (Output Formality) | generation/coding→structured output; classification→short output |
| `reasoning_chain_detected` | D1 (Semantic Complexity) | Reasoning chains correlate with multi-step inference |
| `research_signals` | D4 (Research Dependency) | Non-empty signals → D4 > 0 |
| `confidence` | All dimensions | Model's self-assessed certainty |

---

## 2. Dataset Audit

### 2.1 Overview

| Property | Value |
|----------|-------|
| File | `prompt_example_classifier_bedrock_output.csv` |
| Rows | 1,450 |
| Columns | 20 |
| Source | Bedrock LLM output |
| Duplicate prompts | 2 (4 rows total) |
| Null rows | 1 row has nulls across all label columns |
| Fully empty columns | `error` (100% null), `notes` (100% null) |

### 2.2 Column Inventory

| Column | Keep/Drop | Role |
|--------|-----------|------|
| `prompt` | **Keep** | Primary input text |
| `intent` | **Keep** | New target — needs cleaning |
| `task_type` | **Keep** | New target — needs cleaning |
| `reasoning_chain_detected` | **Keep** | New target — needs type conversion |
| `d1`–`d5` | **Keep** | Existing targets — needs vocabulary fix |
| `research_signals` | **Keep** | New target — needs parsing |
| `confidence` | **Keep** | New target — needs review |
| `complexity` | **Keep** | Cross-reference only; tier derived from D-scores (Decision 1) |
| `prompt_type` | **Keep** | 16 categories — useful as feature or stratification |
| `task_description` | **Keep** | May be useful as auxiliary feature |
| `bad_prompt` | **Drop** | Not used (Decision 4) |
| `good_prompt` | **Drop** | 99.4% identical to `prompt` — redundant |
| `expected_answer` | **Keep** | Valuable for output formality analysis |
| `prompting_techniques` | **Keep** | Useful metadata — correlates with reasoning chain |
| `error` | **Drop** | 100% null |
| `notes` | **Drop** | 100% null |

---

## 3. Identified Data Quality Issues (12 Critical Issues)

### 🔴 CRITICAL — Will break the model if unfixed

#### Issue 1: D-Score Vocabulary Gaps

The Phase 1 rubric defines 5 valid scores: {0.0, 0.25, 0.50, 0.75, 1.0}. This dataset is missing entire score levels:

| Dimension | Used Values | Missing Values | Severity |
|-----------|------------|----------------|----------|
| D1 | {0.0, 0.5, 0.75, 1.0} | **0.25** | 🔴 Critical |
| D2 | {0.0, 0.25, 0.5, 0.75} | **1.0** | 🔴 Critical |
| D3 | {0.0, 0.25, 0.5, 0.75} | **1.0** | 🔴 Critical |
| D4 | {0.0, 0.5, 0.75} | **0.25, 1.0** | 🔴 Critical |
| D5 | {0.0, 0.5, 0.75} | **0.25, 1.0** | 🔴 Critical |

**Impact:** A model trained on this data literally cannot predict D1=0.25, D4=0.25, D4=1.0, D5=0.25, or D5=1.0. The ordinal classification approach from Phase 1 will have empty classes.

#### Issue 2: Extreme D-Score Class Imbalance

| Dimension | Majority Class | Count | Minority Class | Count | Ratio |
|-----------|---------------|-------|----------------|-------|-------|
| D1 | 0.50 | 599 | 1.00 | 1 | **599:1** |
| D2 | 0.50 | 1,098 | 0.25 | 10 | **110:1** |
| D3 | 0.50 | 1,144 | 0.25 | 3 | **381:1** |
| D4 | 0.00 | 1,205 | 0.75 | 77 | **16:1** |
| D5 | 0.00 | 849 | 0.75 | 30 | **28:1** |

**D3 is essentially a constant.** 1,144 out of 1,449 labeled rows (79%) have D3=0.50. The model will learn to always predict D3=0.50 and achieve 79% accuracy by doing nothing. This is not a learnable signal.

**D1=1.0 has exactly 1 sample.** This is a single data point, not a class.

#### Issue 3: T3 Tier Collapse

Deriving tiers from the D-scores using the Phase 1 formula:

| Tier | Count | Percentage |
|------|-------|------------|
| T1 | 716 | 49.4% |
| T2 | 691 | 47.7% |
| T3 | **43** | **3.0%** |

**Only 43 prompts derive to T3.** In Phase 1, T3 was 30% of the dataset. Here it's 3%. The model cannot learn T3 classification from 43 samples. This is a critical gap.

#### Issue 4: `complexity` Label ↔ Tier Mismatch

741 out of 1,450 rows (51%) have a `complexity` label that does not match the tier derived from D-scores:

```
derived_tier   T1   T2   T3
complexity
high           69  275   37     ← 69 "high" prompts have T1 scores!
low           297   41    0     ← 41 "low" prompts have T2 scores!
medium        350  375    6     ← 350 "medium" prompts have T1 scores!
```

**Resolution:** D-scores are ground truth; `complexity` is discarded for tier derivation. Tiers are re-derived from D-scores using the standard formula (Decision 1).

---

### 🟠 MODERATE — Needs fixing before training

#### Issue 5: Intent Field Contamination

42 rows have invalid intent values:

| Invalid Value | Count | Likely Fix |
|---------------|-------|-----------|
| `GENERATION` | 33 | → remap to `ANALYTICAL` or `SYNTHETIC` based on prompt |
| `generation` | 6 | → case error, remap |
| `CLASSIFICATION` | 1 | → remap to `FACTUAL` |
| `FACTUAL\|ANALYTICAL` | 1 | → choose primary |
| `coding` | 1 | → remap to `ANALYTICAL` |

#### Issue 6: Task Type Non-Standard Values

39 rows have task types outside the 7 required values:

| Non-standard Value | Count | Action |
|--------------------|-------|--------|
| `translation` | 24 | → remap to `generation` (Decision 2) |
| `explanation` | 11 | → remap to `reasoning` (Decision 2) |
| `generation|reasoning` | 3 | → remap to `reasoning` (primary) |
| `classification|generation` | 1 | → remap to `classification` (primary) |

#### Issue 7: Intent Class Imbalance

| Intent | Count | Percentage |
|--------|-------|------------|
| ANALYTICAL | 1,116 | 77.0% |
| FACTUAL | 203 | 14.0% |
| SYNTHETIC | 72 | 5.0% |
| STRATEGIC | 16 | 1.1% |

ANALYTICAL dominates at 77%. STRATEGIC has only 16 samples — not enough to learn from.

#### Issue 8: reasoning_chain_detected is String, Not Boolean

The column is stored as string `"True"`/`"False"`, not native boolean. This is a simple type conversion but will cause bugs if not handled.

---

### 🟡 MINOR — Should fix, won't break training

#### Issue 9: `research_signals` Format

- 75.7% of rows have `[]` (empty list as string)
- 353 rows have non-empty signal lists with 47 unique signal types
- Format is valid JSON-string (`'["scientific"]'`) but needs parsing to Python lists

#### Issue 10: Confidence Distribution Gaps

| Confidence | Count |
|------------|-------|
| 0.0 | 28 |
| 0.5 | 8 |
| 0.7 | 67 |
| 0.8 | 614 |
| 0.9 | 573 |
| 0.95 | 131 |
| 1.0 | 27 |

28 rows have confidence=0.0. These should be reviewed — a zero-confidence label is likely unreliable.

#### Issue 11: 1 Fully Null Row

1 row has nulls across `intent`, `task_type`, `reasoning_chain_detected`, `d1`–`d5`, and `confidence`. Drop it.

#### Issue 12: 2 Duplicate Prompts

2 prompt texts appear twice. Drop duplicates keeping the first occurrence.

---

## 4. Dataset Cleaning Plan

### Step 1: Drop Unusable Columns and Rows

```
DROP columns: good_prompt, bad_prompt, error, notes
DROP: 1 fully null row
DROP: 2 duplicate rows (keep first)
Result: 1,447 rows × 16 columns
```

### Step 2: Fix Intent Labels

```python
INTENT_REMAP = {
    'GENERATION': 'ANALYTICAL',      # generation is a task type, not an intent
    'generation': 'ANALYTICAL',
    'CLASSIFICATION': 'FACTUAL',     # classification is factual recall
    'coding': 'ANALYTICAL',          # coding requires analytical reasoning
    'FACTUAL|ANALYTICAL': 'ANALYTICAL'  # multi-label → higher complexity wins
}
```

### Step 3: Fix Task Type Labels

```python
TASK_REMAP = {
    'translation': 'generation',           # translation produces output text
    'explanation': 'reasoning',            # explanation requires reasoning
    'generation|reasoning': 'reasoning',   # multi-label → primary
    'classification|generation': 'classification'  # multi-label → primary
}
```

### Step 4: Fix reasoning_chain_detected Type

```python
df['reasoning_chain_detected'] = df['reasoning_chain_detected'].map({'True': True, 'False': False})
```

### Step 5: Parse research_signals

```python
import ast
df['research_signals'] = df['research_signals'].apply(ast.literal_eval)
```

### Step 6: Validate D-Score Vocabulary

The D-scores already use valid values from {0.0, 0.25, 0.50, 0.75, 1.0} — no snapping is needed. The problem is that some score levels have 0 samples. This is addressed by the Phase 1 merge (Step 8) which brings all 5 score levels for every dimension.

### Step 7: Review and Flag Low-Confidence Rows

```python
# Flag rows with confidence <= 0.5 for review
df['low_confidence_flag'] = df['confidence'] <= 0.5  # 36 rows
```

Keep these rows in training but do NOT use them for evaluation. They are likely noisy labels.

### Step 8: Merge Phase 1 Data (Decision 5)

> [!IMPORTANT]
> **Phase 1 merge is the primary strategy for fixing the T3 collapse and D-score vocabulary gaps.** Phase 1 has 889 enterprise-focused prompts with 271 T3 samples and all 5 D-score levels fully populated. This directly fills the critical gaps in Phase 2.

Phase 1 has 9 columns: `id`, `prompt`, `phrasing_style`, `domain`, `d1`–`d5`. Phase 2 requires additional columns. These must be derived for Phase 1 rows:

#### Column Derivation Strategy for Phase 1 Rows

| Column | Derivation Method | Rationale |
|--------|-------------------|----------|
| `intent` | Map from D1: D1≤0.25→FACTUAL, D1=0.50→ANALYTICAL, D1=0.75→SYNTHETIC, D1=1.0→STRATEGIC | D1 (Semantic Complexity) is the primary correlate of intent. Phase 1 D1 distribution by tier confirms clean mapping: T1 has D1∈{0.0, 0.25}, T2 has D1∈{0.50, 0.75}, T3 has D1∈{0.75, 1.0} |
| `task_type` | Derive heuristically from prompt text: if prompt contains code keywords → `coding`; if prompt asks a question → `classification`; if prompt requests a document/spec → `generation`; if prompt requests analysis/comparison → `reasoning`; if prompt requests summary → `summarisation`; else → `reasoning` | Not perfect but sufficient; the model will learn to override from embeddings |
| `reasoning_chain_detected` | `True` if D1 ≥ 0.50, else `False` | Multi-step reasoning (D1≥0.50) implies reasoning chain |
| `research_signals` | Derive from D4 features: if D4 > 0, extract matching research keywords from prompt text; else `[]` | Phase 1 already has keyword-based D4 features that detect research signals |
| `confidence` | Set to `None` — not applicable for merged rows | Confidence will be derived from prediction probabilities at inference time (Decision 3), not from labels |
| `prompt_type` | Map from Phase 1 `domain` + prompt structure | Approximate mapping: FinOps→INFORMATIONAL, DevOps→INSTRUCTIONAL, etc. |
| `task_description` | Copy from `prompt` (first 80 chars) | Approximate — the model doesn't use this for prediction |
| `complexity` | Derive from D-scores: T1→low, T2→medium, T3→high | Standard formula (Decision 1) |
| `phrasing_style` | Keep from Phase 1 | Phase 2 doesn't have this column but it's useful; Phase 2 rows get `None` |
| `domain` | Keep from Phase 1 | Phase 2 doesn't have this column; Phase 2 rows get `None` |
| `source` | Set to `'phase1'` | Track data provenance; Phase 2 rows get `'phase2'` |

#### What the Merge Fixes

| Problem | Phase 2 Only | After Merge (Phase 1 + Phase 2) |
|---------|-------------|----------------------------------|
| T3 samples | 43 (3%) | **43 + 271 = 314 (13.4%)** |
| D1=0.25 | 0 | **+105 from Phase 1** |
| D1=1.0 | 1 | **+158 from Phase 1** |
| D2=1.0 | 0 | **+148 from Phase 1** |
| D3=1.0 | 0 | **+148 from Phase 1** |
| D4=0.25 | 0 | **+115 from Phase 1** |
| D4=1.0 | 0 | **+133 from Phase 1** |
| D5=0.25 | 0 | **+289 from Phase 1** |
| D5=1.0 | 0 | **+100 from Phase 1** |
| STRATEGIC intent | 16 | **~16 + 158 (D1=1.0) ≈ 174** |
| Total rows | 1,447 | **~2,336** |

> [!NOTE]
> The Phase 1 merge almost completely resolves the D-score vocabulary gaps and T3 collapse. Additional targeted augmentation may still be needed for SYNTHETIC intent and edge cases, but the merge is the single highest-ROI cleaning step.

### Step 9: Optional Targeted Augmentation (Post-Merge)

After the merge, evaluate remaining gaps. If any class still has <50 samples, generate targeted prompts:

| Target | Post-Merge Estimate | Action |
|--------|-------------------|--------|
| SYNTHETIC intent | ~72 + ~195 (D1=0.75 T2) ≈ ~267 | Likely sufficient |
| D3=0.25 | 3 (P2) + 118 (P1) = ~121 | Sufficient |
| D5=0.75 | 30 (P2) + 100 (P1) = ~130 | Sufficient |
| Vague T3 prompts | ~65 from Phase 1 | May need 20–30 more |

### Post-Cleaning Expected State

| Property | Phase 2 Raw | After Clean + Merge |
|----------|-------------|---------------------|
| Total rows | 1,450 | **~2,336** |
| Valid intents | 4 + 5 invalid | **4 valid only** |
| Valid task types | 7 + 4 non-standard | **7 valid only** |
| D-score vocabulary | 3–4 values per dim | **5 values per dim** |
| T3 samples | 43 (3%) | **~314 (13.4%)** |
| T1/T2/T3 balance | 716/691/43 | **~1001/1024/314** |
| STRATEGIC intent | 16 | **~174** |
| Domains | 16 prompt types | **16 prompt types + 15 enterprise domains** |

---

## 5. Architecture Proposals

### Context

Phase 2 extends Phase 1's prediction targets from **5 dimensions + tier** to **5 dimensions + tier + intent + task_type + reasoning_chain + research_signals + confidence**. The system must predict 10 outputs from a single prompt. This changes the architecture requirements:

- **Phase 1:** 5 ordinal classification targets + 1 tier (all numeric/ordinal)
- **Phase 2:** 5 ordinal + 1 tier + 1 categorical (intent, 4-class) + 1 categorical (task_type, 7-class) + 1 binary (reasoning_chain) + 1 structured list (research_signals) + 1 continuous (confidence)

The mix of ordinal, categorical, binary, list, and continuous outputs means a single model cannot handle everything — a multi-head or pipeline approach is needed.

---

### Architecture A: Shared-Backbone Multi-Head XGBoost (Extending v4)

```
Prompt
  │
  ├──→ MiniLM Embedding (384d) ──→ PCA (35d)
  │
  └──→ Hand-Crafted Features (32d)
          │
          ├── Phase 1 features (D1-D5 targeting)
          └── Phase 2 features (intent/task keywords)
                    │
                    ▼
            StandardScaler (67d)
                    │
     ┌──────────────┼──────────────────────┐
     │              │                      │
     ▼              ▼                      ▼
  Tier XGB     D1–D5 MultiOut         Intent XGB
  (3-class)    XGB (5×5-class)        (4-class)
     │              │                      │
     │              ▼                      │
     │         Task Type XGB               │
     │         (7-class)                   │
     │              │                      │
     │         Reasoning Chain XGB         │
     │         (2-class binary)            │
     │              │                      │
     │         Research Signal             │
     │         Rule Engine                 │
     │              │                      │
     └──────────────┼──────────────────────┘
                    │
              Confidence = min(head confidences)
                    │
                    ▼
             Final JSON Output
```

**How it works:**
- Same shared backbone as v4 (MiniLM → PCA → hand-crafted → scale)
- 5 separate XGBoost heads: Tier, D1–D5, Intent, Task Type, Reasoning Chain
- Research signals extracted via rule engine (keyword lists — same approach as Phase 1 D4 features)
- Confidence computed as the minimum of `predict_proba` max values across heads

**Pros:**
- Direct extension of v4 — reuses all Phase 1 infrastructure
- Each head can be trained, tuned, and evaluated independently
- Fastest to implement; easiest to debug
- No new dependencies

**Cons:**
- No shared learning between heads — intent and D1 are correlated but trained independently
- Research signals are rule-based, not learned
- Confidence is heuristic, not calibrated

**Estimated effort:** 1–2 days  
**Expected tier accuracy:** ~88–90% (same as v4 with augmented data)  
**Expected intent accuracy:** ~82–87% (ANALYTICAL dominates; STRATEGIC hard)

---

### Architecture B: Two-Model Pipeline (Classification + Regression Stages)

```
Prompt
  │
  ├──→ MiniLM Embedding (384d) ──→ PCA (35d)
  │
  └──→ Hand-Crafted Features (32d + 10 new Phase 2 features)
                    │
                    ▼
            StandardScaler (77d)
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    MODEL 1: Classifiers    MODEL 2: Regressors
    ┌─────────────────┐     ┌──────────────────┐
    │ Intent (4-class)│     │ D1–D5 Regression │
    │ Task (7-class)  │     │ (MultiOutput     │
    │ Reasoning (2c)  │     │  XGBRegressor)   │
    │ Tier (3-class)  │     │                  │
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             │    ┌──────────────────┤
             │    │                  │
             ▼    ▼                  ▼
        ┌─────────────┐      Score Snapping
        │  Research    │      {0,0.25,0.5,0.75,1}
        │  Signal      │             │
        │  Classifier  │      Complexity Score
        │  (multi-     │      & Tier Derivation
        │   label)     │             │
        └──────┬──────┘             │
               │                    │
               └────────┬───────────┘
                        │
                  Confidence = calibrated
                  probability average
                        │
                        ▼
                  Final JSON Output
```

**How it works:**
- Shared backbone with 10 new Phase 2 features (`has_role_prompt`, `has_chain_of_thought`, `code_block_detected`, `question_type`, `output_length_signal`, etc.)
- Model 1: Multi-task classifier for all categorical/binary outputs
- Model 2: Multi-output regressor for D1–D5 with post-hoc score snapping
- Research signals: trained as a multi-label classifier (not rule-based)
- Confidence: Platt-scaled probability calibration

**Pros:**
- Clean separation between classification and regression tasks
- Research signals are learned, not rule-based
- Score snapping from regression is more robust for rare classes (the regression model can interpolate even if D1=0.25 has few samples)
- Calibrated confidence via Platt scaling

**Cons:**
- Two models to maintain
- Regression + snapping introduces quantization noise
- Multi-label classifier for research signals needs sufficient positive examples (353 non-empty out of 1,450)
- More complex evaluation — need metrics for both stages

**Estimated effort:** 3–4 days  
**Expected tier accuracy:** ~87–90%  
**Expected intent accuracy:** ~84–88%

---

### Architecture C: Hybrid Embedding Fine-Tuning + XGBoost Heads

```
Prompt
  │
  ▼
MiniLM Embedding (384d)
  │
  ├──→ [OPTION] Fine-tune MiniLM with
  │    contrastive loss on (prompt, tier) pairs
  │    OR use as-is (frozen)
  │
  ├──→ PCA (35d)
  │
  └──→ Prompt-level meta-features (42 features)
       ┌───────────────────────────────┐
       │ Phase 1: 32 features          │
       │ Phase 2: 10 new features      │
       │  - has_role_prompt            │
       │  - has_chain_of_thought       │
       │  - code_keyword_count         │
       │  - question_count             │
       │  - list_format_detected       │
       │  - imperative_verb_ratio      │
       │  - prompt_type_encoded (OHE)  │
       │  - ...                        │
       └───────────────────────────────┘
                    │
                    ▼
            StandardScaler (77d)
                    │
          ┌─────────┼──────────┬──────────────┐
          │         │          │              │
          ▼         ▼          ▼              ▼
       Tier      D1–D5     Intent+Task    Research
       XGB       Multi     Joint XGB      Signal
       (3c)      Output    Multi-Output   Multi-Label
                 XGB       (4c + 7c)      XGB
                 (5×5c)                   (47-label)
          │         │          │              │
          │         │          │              │
          ▼         ▼          ▼              ▼
       Reasoning Chain: derived from Intent + D1
       (if Intent ∈ {ANALYTICAL, SYNTHETIC, STRATEGIC} AND D1 ≥ 0.50)
          │
          ▼
       Confidence: weighted average of predict_proba maxima
       across all heads, calibrated via isotonic regression
          │
          ▼
       Final JSON Output
```

**How it works:**
- Optional: fine-tune MiniLM on a contrastive objective (prompts with same tier should have similar embeddings). This could improve the embedding quality for this specific task but adds training complexity.
- Extended feature set (42 features) with Phase 2-specific features
- Intent and Task Type trained jointly (one MultiOutputClassifier) since they are correlated
- Reasoning chain derived deterministically from intent + D1 (avoids training a weak binary classifier on imbalanced data — 80% True)
- Research signals as multi-label XGBoost (one binary classifier per signal type, or top-K signals only)
- Confidence via isotonic regression calibration

**Pros:**
- Fine-tuned embeddings could improve all downstream heads by 2–4%
- Joint intent+task training captures cross-task correlations
- Deterministic reasoning chain avoids training on highly imbalanced data
- Calibrated confidence via isotonic regression (better calibration than Platt for this data size)
- Most scalable — adding new task types only requires dataset update + retraining

**Cons:**
- Fine-tuning MiniLM requires a GPU and ~30 min training
- Multi-label research signal classifier needs careful threshold tuning
- Most complex to implement and debug
- Risk: fine-tuning on 1,950 samples may overfit the embeddings

**Estimated effort:** 5–7 days  
**Expected tier accuracy:** ~89–92%  
**Expected intent accuracy:** ~85–90%

---

## 6. Architecture Comparison

| Property | A: Multi-Head XGB | B: Two-Model Pipeline | C: Hybrid Fine-Tune |
|----------|-------------------|----------------------|---------------------|
| **Backbone** | v4 backbone (frozen) | v4 backbone (frozen) | MiniLM fine-tune (optional) |
| **D1–D5 approach** | Ordinal classification | Regression + snap | Ordinal classification |
| **Intent** | Separate XGB head | Part of classifier model | Joint with task_type |
| **Research signals** | Rule engine | Multi-label classifier | Multi-label classifier |
| **Reasoning chain** | Separate binary XGB | Separate binary XGB | Derived from intent+D1 |
| **Confidence** | min(proba maxima) | Platt calibration | Isotonic calibration |
| **Implementation effort** | 1–2 days | 3–4 days | 5–7 days |
| **Expected tier accuracy** | 88–90% | 87–90% | 89–92% |
| **Expected intent accuracy** | 82–87% | 84–88% | 85–90% |
| **Maintainability** | ★★★★★ Simple | ★★★☆☆ Moderate | ★★☆☆☆ Complex |
| **Risk** | Low | Low | Medium (fine-tune overfit) |

> [!IMPORTANT]
> **My recommendation: Start with Architecture A.** It's the safest extension of the proven v4 pipeline. The new heads (intent, task_type, reasoning_chain) are straightforward XGBoost classifiers on the same feature backbone. If A achieves <85% intent accuracy, escalate to B. Architecture C should only be attempted if both A and B underperform AND more data is available.

---

## 7. Dataset Verdict (Pre-Cleaning)

### What's Good
- **1,450 rows** — larger than Phase 1's 889
- **16 prompt_type categories** — excellent diversity (CREATIVE_WRITING, CODE_EXPLANATION, COMPARISON, etc.)
- **20 columns** — rich metadata including bad_prompt, expected_answer, prompting_techniques
- **All required new fields present** — intent, task_type, reasoning_chain_detected, research_signals, confidence
- **Research signals** have 47 unique signal types with valid JSON-list format

### What's Broken (and must be fixed before ANY training)
1. **T3 collapse: 43 samples (3%)** — cannot learn T3 from this
2. **D3 is a constant: 79% at 0.50** — no learning signal
3. **D-score vocabulary gaps: 10 missing score levels** across 5 dimensions
4. **741 complexity↔tier mismatches (51%)** — labels are inconsistent
5. **STRATEGIC has 16 samples** — not learnable
6. **42 invalid intent values** — data entry errors
7. **39 non-standard task_type values** — needs remapping
8. **D1=1.0 has 1 sample** — a single data point, not a class

### Bottom Line

> [!CAUTION]
> **This dataset is NOT training-ready.** It requires substantial cleaning (Steps 1–7) and mandatory augmentation (Step 8) before any architecture can be applied. The D-score distributions are so imbalanced that a model would achieve "high accuracy" by predicting the majority class for every dimension — which is useless.
>
> The cleaning + augmentation is estimated at ~500 new rows, bringing the total to ~1,950. After augmentation, the dataset should have:
> - Every D-score level represented with ≥60 samples
> - T3 with ~300 samples (15% of total)
> - STRATEGIC intent with ~100 samples
> - All 5 valid D-score values per dimension

---

## 8. Resolved Decisions

All open questions have been resolved:

| # | Question | Decision | Impact |
|---|---------|----------|--------|
| 1 | `complexity` vs D-scores — which is ground truth? | **Trust D-scores, re-derive tier** | 741 mismatched `complexity` labels are overridden; tier comes from formula |
| 2 | Are `translation` and `explanation` valid task types? | **Remap to nearest valid type** | `translation` → `generation`, `explanation` → `reasoning` |
| 3 | Should `confidence` be predicted or calibrated? | **Derive from prediction probabilities** | No separate confidence model; use `predict_proba` max values |
| 4 | Is `bad_prompt` useful? | **No — drop it** | Column removed in Step 1 |
| 5 | Are Phase 1 prompts being merged? | **Yes — merge Phase 1 (889 rows)** | Fixes T3 collapse (43→314), fills all D-score vocabulary gaps |

---

## 9. Next Steps

1. **Execute cleaning Steps 1–7** on Phase 2 dataset
2. **Implement Phase 1 merge (Step 8)** — derive new columns for Phase 1 rows
3. **Validate merged dataset** — verify D-score coverage, tier balance, intent distribution
4. **Evaluate if additional augmentation (Step 9) is needed** based on post-merge class counts
5. **Choose architecture and implement** — start with Architecture A
6. **Train and evaluate** — 5-fold CV on merged dataset
