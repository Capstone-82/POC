import pandas as pd
import numpy as np
import ast, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('phase2/prompt_classifier_phase1_phase2_merged_cleaned.csv')
p1 = df[df['source'] == 'phase1']
p2 = df[df['source'] == 'phase2']

print(f"Total: {len(df)}, Phase1: {len(p1)}, Phase2: {len(p2)}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")

print("\n" + "="*80)
print("ISSUE CHECK 1: D-score vocabulary validation")
print("="*80)
valid_scores = {0.0, 0.25, 0.5, 0.75, 1.0}
for col in ['d1','d2','d3','d4','d5']:
    vals = set(df[col].dropna().unique())
    invalid = vals - valid_scores
    missing = valid_scores - vals
    print(f"  {col}: invalid={invalid if invalid else 'None'}, missing_levels={missing if missing else 'None'}")

print("\n" + "="*80)
print("ISSUE CHECK 2: Intent validation")
print("="*80)
valid_intents = {'FACTUAL', 'ANALYTICAL', 'SYNTHETIC', 'STRATEGIC'}
intent_vals = set(df['intent'].dropna().unique())
invalid_intents = intent_vals - valid_intents
print(f"  Valid intents: {intent_vals & valid_intents}")
print(f"  Invalid intents: {invalid_intents if invalid_intents else 'None'}")
print(f"  Null intents: {df['intent'].isna().sum()}")

print("\n" + "="*80)
print("ISSUE CHECK 3: Task type validation")
print("="*80)
valid_tasks = {'classification', 'generation', 'reasoning', 'coding', 'summarisation', 'sparql_generation', 'formatting'}
task_vals = set(df['task_type'].dropna().unique())
invalid_tasks = task_vals - valid_tasks
print(f"  Valid: {task_vals & valid_tasks}")
print(f"  Invalid: {invalid_tasks if invalid_tasks else 'None'}")
print(f"  Null: {df['task_type'].isna().sum()}")
print(f"  Note: sparql_generation count = {(df['task_type']=='sparql_generation').sum()}")

print("\n" + "="*80)
print("ISSUE CHECK 4: reasoning_chain_detected type and distribution")
print("="*80)
print(f"  dtype: {df['reasoning_chain_detected'].dtype}")
print(f"  values: {df['reasoning_chain_detected'].value_counts().to_dict()}")
print(f"  nulls: {df['reasoning_chain_detected'].isna().sum()}")

print("\n" + "="*80)
print("ISSUE CHECK 5: Tier derivation correctness")
print("="*80)
cs_check = df['d1']*0.35 + df['d2']*0.20 + df['d3']*0.20 + df['d4']*0.15 + df['d5']*0.10
tier_check = cs_check.apply(lambda s: 'T1' if s < 0.4 else ('T2' if s < 0.7 else 'T3'))
mismatched_tiers = (df['tier'] != tier_check).sum()
print(f"  Tier derivation mismatches: {mismatched_tiers}")
if mismatched_tiers > 0:
    bad = df[df['tier'] != tier_check][['prompt','d1','d2','d3','d4','d5','tier','source']].head(5)
    print(bad.to_string())

# Check complexity_score too
if 'complexity_score' in df.columns:
    cs_diff = (df['complexity_score'] - cs_check).abs()
    print(f"  Max complexity_score deviation: {cs_diff.max():.6f}")

print("\n" + "="*80)
print("ISSUE CHECK 6: Phase 1 intent derivation sanity")
print("="*80)
print("  Phase 1 D1→Intent mapping check:")
print(pd.crosstab(p1['d1'], p1['intent']))

print("\n" + "="*80)
print("ISSUE CHECK 7: Duplicate prompts")
print("="*80)
dupe_prompts = df[df['prompt'].duplicated(keep=False)]
print(f"  Total rows with duplicate prompts: {len(dupe_prompts)}")
print(f"  Unique duplicate prompts: {df['prompt'].duplicated().sum()}")
# Check if any are cross-source duplicates
if len(dupe_prompts) > 0:
    cross = dupe_prompts.groupby('prompt')['source'].nunique()
    cross_source = (cross > 1).sum()
    print(f"  Cross-source duplicates (in BOTH phase1 and phase2): {cross_source}")

print("\n" + "="*80)
print("ISSUE CHECK 8: D-score imbalance (merged)")
print("="*80)
for col in ['d1','d2','d3','d4','d5']:
    vc = df[col].value_counts()
    ratio = vc.max() / max(vc.min(), 1)
    print(f"  {col}: majority={vc.idxmax()}({vc.max()}), minority={vc.idxmin()}({vc.min()}), ratio={ratio:.1f}x")

print("\n" + "="*80)
print("ISSUE CHECK 9: Task type by source")
print("="*80)
print(pd.crosstab(df['source'], df['task_type']))

print("\n" + "="*80)
print("ISSUE CHECK 10: research_signals parseable")
print("="*80)
parse_errors = 0
for i, val in df['research_signals'].items():
    try:
        if pd.notna(val):
            ast.literal_eval(str(val))
    except:
        parse_errors += 1
print(f"  Parse errors: {parse_errors}")
non_empty = df[df['research_signals'].apply(lambda x: str(x) not in ['[]', 'nan', ''])].shape[0]
print(f"  Non-empty signals: {non_empty}")

print("\n" + "="*80)
print("ISSUE CHECK 11: Null counts in critical columns")
print("="*80)
critical = ['prompt','intent','task_type','reasoning_chain_detected','d1','d2','d3','d4','d5','tier','source']
for c in critical:
    n = df[c].isna().sum()
    if n > 0:
        print(f"  *** {c}: {n} nulls")
    else:
        print(f"  {c}: 0 nulls ✓")

print("\n" + "="*80)
print("ISSUE CHECK 12: Phase 1 task_type heuristic quality")
print("="*80)
# Check if question-ending prompts got classified as 'classification'
p1_questions = p1[p1['prompt'].str.strip().str.endswith('?')]
print(f"  Phase 1 prompts ending with '?': {len(p1_questions)}")
print(f"  Of those, task_type='classification': {(p1_questions['task_type']=='classification').sum()}")
print(f"  This may be wrong — questions like 'why is X?' are reasoning, not classification")
# Show examples
print("\n  Sample question prompts classified as 'classification':")
q_class = p1_questions[p1_questions['task_type'] == 'classification']
for _, r in q_class.head(5).iterrows():
    p = r['prompt'][:80]
    print(f"    [{r['intent']}] {p}")
