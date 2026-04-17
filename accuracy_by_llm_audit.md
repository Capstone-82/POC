## Root Cause Analysis

**Why scores cluster at 95–99+:**

LLM judges have a well-documented severity bias — they default to generous scores unless the response is obviously wrong. When you ask a model to score another model's output on a 0–100 scale with a vague rubric like "how accurate is this?", you get inflation because:

The judge has no reference answer to compare against. It's scoring in a vacuum, so it evaluates surface qualities — fluency, confidence, structure — rather than correctness. A confidently wrong answer scores higher than a hesitant correct one.

Your evaluators (LLaMA4, Mistral, Nova) are all instruction-tuned models with similar training objectives. They share aesthetic preferences. A response that sounds like a good LLM answer gets a high score from all three, regardless of whether it's actually the best response to that specific prompt. This is why your inter-evaluator agreement is high — they're not independently verifying quality, they're independently agreeing on style.

The 0–100 scale itself is the wrong instrument. LLM judges are not calibrated to use the full range. They treat anything below 70 as "bad" and cluster everything "acceptable" in the 85–100 band. You have effectively a 15-point scale masquerading as a 100-point one.

The fundamental flaw is that you're using the same class of model (instruction-tuned LLM) to evaluate outputs from the same class of model, without ground truth, without a reference answer, and with a scale that has no calibration. The signal you get is "does this response sound like something a good LLM would say" — which is nearly always yes, for any production-grade model.

---

## Better Evaluation Strategy

Stop trying to fix `avg_accuracy_score`. The architecture is wrong. You need signals that don't require a judge to have an opinion.

**The core principle: measure what you can verify, not what an LLM thinks.**

### Signal 1 — Pairwise Preference (Replaces Absolute Scoring)

Instead of asking "score this response 0–100", ask "which of these two responses is better, and why?" Pairwise comparison is dramatically more reliable than absolute scoring because:

- The judge only needs to make a relative judgment, not calibrate to a scale
- You can use Bradley-Terry or Elo to convert pairwise wins into a ranking
- Discrimination power is much higher — models that are similar in absolute score separate clearly in pairwise comparison

Implementation: for each prompt, take the responses from your top 5 candidate models, run all 10 pairwise comparisons, and record win/loss. Aggregate across prompts to get a win rate per model per use case. This is your new primary signal.

The evaluator prompt changes from "score this 0–100" to "Response A vs Response B: which better answers the prompt and why? Answer only A or B." This is harder to inflate.

**This is your highest-ROI change. Implement this first.**

### Signal 2 — Task-Specific Verifiable Metrics (No Judge Needed)

For code-generation specifically, stop using LLM evaluation entirely. Use execution:

- Does the code run without syntax errors? (AST parse check, zero LLM calls)
- Does it pass basic functional tests you generate at prompt creation time?
- Does it handle the specific case mentioned in the prompt?

For reasoning tasks: does the final answer match a checkable form? Math problems have numeric answers. Logic puzzles have correct/incorrect outcomes. These are binary signals with zero evaluator bias.

For text-generation: readability scores (Flesch-Kincaid), coherence metrics (sentence embedding similarity between paragraphs), length-appropriateness relative to the prompt's implied scope.

These signals are cheap, deterministic, and have real discrimination power. A model that produces syntactically broken code scores 0. There is no inflation.

### Signal 3 — Consistency Under Perturbation

Run each model on 3–5 paraphrases of the same prompt. Measure the variance in output quality and content overlap. A reliable model produces consistent outputs across paraphrases. An inconsistent model is unreliable regardless of its peak score.

This catches something absolute scoring completely misses: models that score 95 on one phrasing and 60 on a paraphrase are not good routing targets.

Implementation: cosine similarity between embeddings of outputs across paraphrases. Low variance = high consistency score. This is a pure numerical signal, no judge required.

### Signal 4 — Calibrated Pairwise LLM Scoring (If You Must Use LLM Judges)

If you keep LLM evaluation, change the rubric immediately. Replace the 0–100 scale with:

A structured rubric with 4–5 specific dimensions scored 1–5 each: factual accuracy, completeness, conciseness, format adherence, task-specific quality. Force the judge to justify each dimension score with one sentence. Average the dimensions.

Add a reference answer. For your benchmark prompts, generate a "gold standard" response using your strongest model (or human-written for critical prompts), and ask the judge to compare against it explicitly. "Compared to this reference, how does this response perform on each dimension?" This anchors the scale and reduces inflation dramatically.

Force score distribution. Add to the system prompt: "You must use the full 1–5 range. A score of 5 means this response could not be improved. At least 30% of responses you evaluate should score 3 or below." This is crude but effective at reducing ceiling effects.

---

## Recommendation Impact

The current recommender uses `avg_accuracy_score` as 55% of the fusion score. That weight is being applied to a signal that has near-zero variance (std of ~10 across a 0–100 scale, with everything clustered 91–97). In practice, the 55% accuracy weight is contributing almost nothing to model differentiation — the ranking is being driven by cost and latency, which have real variance, even though they're weighted at 25% and 15%.

**Immediate changes to scoring:**

Replace `avg_accuracy_score` in the fusion formula with a composite of your new signals. Concretely:

For code-generation: 40% execution success rate + 30% pairwise win rate + 20% consistency score + 10% cost-efficiency. Drop latency weight to 0% for async code tasks where users aren't waiting interactively.

For reasoning: 50% pairwise win rate + 30% answer correctness (where verifiable) + 20% consistency score.

For text-generation: 45% pairwise win rate + 25% consistency score + 20% cost-efficiency + 10% latency.

The key shift: pairwise win rate becomes your primary accuracy proxy because it has real discrimination power. A model with a 70% win rate vs one with a 40% win rate is a meaningful difference. A model with 96.0 avg_accuracy vs 94.0 is noise.

**Changes to KNN ranking:** Your sim_weighted_accuracy calculation currently weights by cosine similarity and sums accuracy scores. Since accuracy scores have no variance, this degenerates to "whichever model has more neighbors wins." With pairwise win rates, the same calculation becomes meaningful — a model with high win rates on similar prompts is genuinely the better recommendation.

**Remove accuracy from the switch threshold.** The `min_accuracy_gain` threshold in your switching policy currently requires a minimum accuracy improvement to trigger a switch. With the current inflated scores, this threshold is never meaningfully triggered. Replace it with a minimum win-rate advantage (e.g., only switch if the recommended model has a win rate > 10 percentage points higher than the current model on similar prompts).

---

## Practical Implementation Plan

### Day 1 — Stop the bleeding

**Morning (2–3 hours):** Add a pairwise evaluation endpoint to your existing evaluator API. The prompt template:

```
System: You are evaluating two LLM responses. Be critical and decisive.

Prompt given to both models: {prompt}

Response A ({model_a}): {response_a}
Response B ({model_b}): {response_b}

Which response better addresses the prompt? Consider: accuracy, completeness, 
and conciseness. Output only JSON: {"winner": "A" or "B", "reason": "one sentence"}
```

Run this for your existing benchmark prompts. You don't need new data — re-evaluate your existing 548 prompts with pairwise comparison and store win/loss records in a new `pairwise_results` table.

```sql
CREATE TABLE pairwise_results (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash  TEXT,
    use_case     TEXT,
    complexity   TEXT,
    model_a      TEXT,
    model_b      TEXT,
    winner       TEXT,
    judge_model  TEXT,
    created_at   TIMESTAMPTZ DEFAULT now()
);
```

**Afternoon (2–3 hours):** Write a script that computes win rates per model per use_case from `pairwise_results`. This becomes your new ranking signal. Add a `win_rate` column to your `model_priors` table (when you build it) and expose it from your aggregation query.

### Day 2 — Wire it into the recommender

**Morning:** Replace `avg_accuracy_score` in `score_and_rank_knn_candidates()` with a blended signal: 60% pairwise win rate (pulled from your new table) + 40% existing cost/latency normalization. Don't touch the KNN retrieval logic — just change what you aggregate.

For prompts where you don't yet have pairwise data (new use cases, rare prompts), fall back to the existing `avg_accuracy_score` as a weak prior rather than leaving the field empty.

**Afternoon:** Add a code execution check for code-generation prompts. This is 20 lines of Python — parse the model's output for a code block, run `ast.parse()` on it, record pass/fail. A model that produces syntactically invalid code 20% of the time should never be the top recommendation for code tasks, regardless of its pairwise win rate.

```python
import ast

def check_code_syntax(response: str) -> bool:
    import re
    blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    if not blocks:
        return True  # no code block, not applicable
    try:
        ast.parse(blocks[0])
        return True
    except SyntaxError:
        return False
```

Store this as `syntax_pass` boolean in your benchmark results. For code-generation KNN neighbors, models with syntax_pass rate below 85% get a hard penalty in scoring, not just a soft weight adjustment.

### What Not to Do in These 2 Days

Don't retrain the classifier. Don't rebuild the embedding pipeline. Don't touch the KNN search logic. Don't implement Bradley-Terry Elo (it requires more pairwise data than you currently have). Don't add more evaluator models — you have three, which is enough if the evaluation format is fixed.

The bottleneck is not the number of evaluators or the sophistication of the aggregation. It's that you're asking the wrong question ("score 0–100") and getting a useless answer. Changing the question to "which is better" is the entire fix, and it takes one day to implement on top of your existing infrastructure.