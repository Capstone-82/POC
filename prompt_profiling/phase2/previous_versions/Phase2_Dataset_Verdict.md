# Phase 2 Dataset — Clean & Merge Verdict

## Overall Assessment: ✅ Well Executed, 2 Fixes Needed

Your cleaning and merge followed the plan accurately. The core operations (column drops, intent remapping, task_type remapping, boolean conversion, research_signals parsing, tier re-derivation, Phase 1 column expansion) are all correct.

**Validated ✅:**
- D-score vocabulary: all 5 levels present for all 5 dimensions ✅
- Intent values: only {FACTUAL, ANALYTICAL, SYNTHETIC, STRATEGIC} ✅
- Task type values: only valid types, no pipe-separated ✅
- `reasoning_chain_detected`: proper boolean dtype ✅
- Tier derivation: 0 mismatches vs formula ✅
- `complexity_score`: 0.000000 max deviation from formula ✅
- D1→Intent mapping: perfectly clean (D1≤0.25→FACTUAL, D1=0.50→ANALYTICAL, D1=0.75→SYNTHETIC, D1=1.0→STRATEGIC) ✅
- No cross-source duplicates (Phase 1 prompt ≠ Phase 2 prompt) ✅
- `research_signals`: 0 parse errors, all valid JSON ✅
- 0 nulls in all critical columns (prompt, intent, task_type, d1–d5, tier) ✅
- Low-confidence rows flagged but retained ✅
- `source` column tracks provenance ✅

---

## 🔴 Mistake 1: 63 Phase 1 Duplicate Prompts (Should Fix)

**What happened:** 26 unique prompt texts appear multiple times in the Phase 1 data (89 total rows), all with identical D-scores. Examples:

```
"what is a cloud region and give a short example" → 4 copies (v4_d2zero_001, _021, _041, _061)
"what is a service account and give a short example" → 4 copies
"what is a deployment pipeline" → 4 copies
```

These are artifacts from the Phase 1 v4 augmentation where D2=0.0 prompts were mass-generated. They have identical text and identical labels — they add no information, only inflate sample counts.

**Impact:** 63 inflated rows (6.7% of Phase 1 data). These will cause the model to memorize these exact prompts during training and artificially boost cross-validation scores. In 5-fold CV, some copies will leak into the test fold while identical copies remain in training — this is **data leakage**.

**Fix:** Deduplicate Phase 1 rows by prompt text, keeping the first occurrence:

```python
# Before merge
p1 = p1.drop_duplicates(subset='prompt', keep='first')
# This removes 63 rows → 826 Phase 1 rows
```

**Post-fix total:** 2,336 − 63 = **2,273 rows**

---

## 🟠 Mistake 2: Question-Ending Prompts → `classification` (Should Fix)

**What happened:** The task_type heuristic maps prompts ending with `?` to `classification`. This caught 29 Phase 1 prompts, but most are wrong:

| Prompt | Intent | Assigned task_type | Correct task_type |
|--------|--------|-------------------|-------------------|
| "hey can someone explain what databricks actually does?" | FACTUAL | classification | **reasoning** |
| "whats the cheapest aws region for s3?" | FACTUAL | classification | **reasoning** |
| "can you check this kubernetes yaml for simple syntax errors?" | FACTUAL | classification | **coding** |
| "the new EU AI Act just dropped and HR is saying our Workday..." | STRATEGIC | classification | **reasoning** |
| "jira alerts are completely spamming our slack channels..." | ANALYTICAL | classification | **reasoning** |

The problem: `classification` in the task_type spec means **"tag/categorize an input into predefined classes"** (e.g., "classify this email as spam or not-spam"). These prompts are questions asking for explanations, comparisons, or analysis — not classification tasks.

**Impact:** 29 wrongly labeled rows. FACTUAL questions like "what is X?" should be `reasoning` (they require explanation). Code-related questions should be `coding`. Only prompts that genuinely ask to categorize/label something should be `classification`.

**Fix:** Change the heuristic priority — `?` ending should NOT override other keyword matches. Remove the `?` → `classification` rule entirely, or move it to lowest priority:

```python
# Current (problematic):
if prompt.endswith('?'):
    return 'classification'  # ← fires before other checks

# Fixed:
# Remove the ? → classification rule entirely
# Let keyword matching decide, fallback to 'reasoning'
```

After the fix, these 29 prompts would get:
- 24 FACTUAL questions → `reasoning` (they ask "what is X?", "how does Y?")
- 4 ANALYTICAL questions → `reasoning`
- 1 with code keywords → `coding`

---

## 🟡 Known Limitations (Acceptable, Not Mistakes)

### Limitation 1: D-Score Imbalance Remains

| Dimension | Majority | % | Comment |
|-----------|----------|---|---------|
| D1 | 0.75 | 34.7% | Acceptable — no single class dominates |
| D2 | 0.50 | 57.7% | Moderate concentration |
| **D3** | **0.50** | **56.5%** | Still heavy — was 79% in Phase 2 alone, merge reduced it |
| **D4** | **0.00** | **67.9%** | Heavy — reflects reality (most prompts need no research) |
| D5 | 0.00 | 47.4% | Acceptable |

D3 and D4 are the most skewed. Phase 1 brought D3 down from 79% to 56.5% — a real improvement, but the model may still over-predict D3=0.50. D4=0.0 at 68% reflects genuine data distribution (most prompts don't need live retrieval), so this is not a labeling error.

**Action:** Use `class_weight='balanced'` or `sample_weight` during training. Not a dataset fix.

### Limitation 2: `reasoning_chain_detected` Is 77% True

76.7% of rows have `reasoning_chain_detected=True`. This makes the binary classifier easy to "cheat" by always predicting True. The Phase 1 derivation (D1 ≥ 0.50 → True) contributes: 622 of 889 Phase 1 rows (70%) have D1 ≥ 0.50.

**Action:** This is an inherent property of the data — most prompts sent to an enterprise LLM routing system do involve reasoning chains. Use balanced sampling or derive it deterministically from intent + D1 (Architecture C approach).

### Limitation 3: 889 Null Confidences

All Phase 1 rows have `confidence=None`. This is correct per your decision (Decision 3: derive confidence from `predict_proba`, not from labels). The confidence column is not a training target — it's an inference output.

**Action:** None needed. The column serves as metadata for Phase 2 rows only.

---

## Final Dataset Profile (After 2 Fixes Applied)

| Property | Value |
|----------|-------|
| **Total rows** | **~2,273** (2,336 − 63 deduped) |
| Phase 1 rows | ~826 |
| Phase 2 rows | 1,447 |
| **Tier balance** | T1: ~975, T2: ~998, T3: ~300 |
| D-score coverage | All 5 levels × all 5 dimensions ✅ |
| Intent balance | ANALYTICAL: ~1,280, FACTUAL: ~450, SYNTHETIC: ~370, STRATEGIC: ~174 |
| Task types | 6 valid types (no `sparql_generation` samples yet) |
| Domains | 16 prompt types + 15 enterprise domains |

## Verdict

> **The dataset is training-ready after the 2 fixes above.** The Phase 1 merge was the right call — it solved the T3 collapse (43 → 313), filled all D-score vocabulary gaps, and brought STRATEGIC from 16 to 174 samples. The cleaning steps (intent remap, task_type remap, boolean conversion, tier re-derivation) were all executed correctly.
>
> Fix the 63 duplicates and the 29 question→classification misassignments, then this dataset is ready for Architecture A implementation.

## Checklist Before Training

- [ ] Deduplicate Phase 1 prompts (drop 63 rows)
- [ ] Fix task_type heuristic: remove `? → classification` rule, re-derive for affected rows
- [ ] Re-export merged CSV
- [ ] Verify post-fix row count (~2,273)
- [ ] Note: `sparql_generation` has 0 samples — model cannot learn this task type. Either add samples or remove from valid list.
