# V2 Router Rollout TODO

## Phase 1: Supabase Schema
- [ ] Run the SQL schema from `model_training/schema_knn.sql` in Supabase.
- [ ] Confirm new tables exist: `pairwise_results`, `model_win_rates`, `routing_log`, `prompt_embeddings`.
- [ ] Confirm new `benchmark_results` columns exist for syntax, correctness, and prompt metadata.

## Phase 2: Existing Dataset Backfill
- [ ] Run `model_training/run_pairwise_eval.py` on the existing benchmark dataset.
- [ ] Run `model_training/compute_win_rates.py` to populate `model_win_rates`.
- [ ] Run `model_training/verify_syntax.py` for existing code-generation rows.
- [ ] Spot-check `pairwise_results` and `model_win_rates` in Supabase.

## Phase 3: KNN Validation
- [ ] Run KNN recommendations using the updated `win_rate`-based scoring.
- [ ] Compare recommended models before vs after the metric change.
- [ ] Evaluate whether switching behavior is now more selective and sensible.
- [ ] Capture a few representative prompts and results for review.

## Phase 4: New Prompt Corpus
- [ ] Generate the new prompt corpus dataset.
- [ ] Benchmark all candidate models on the new prompts.
- [ ] Run pairwise evaluation on the new benchmark rows.
- [ ] Refresh `model_win_rates` again after the new corpus lands.
- [ ] Re-run KNN evaluation on new prompts to measure generalization.
