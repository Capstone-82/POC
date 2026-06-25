import pandas as pd
import numpy as np
import sys, ast
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('phase2/prompt_example_classifier_bedrock_output.csv')

print("="*80)
print("DATA QUALITY ISSUES")
print("="*80)

# 1. D-scores: missing 0.25 values
print("\n--- Issue 1: D-score vocabulary gaps ---")
valid = {0.0, 0.25, 0.5, 0.75, 1.0}
for col in ['d1', 'd2', 'd3', 'd4', 'd5']:
    used = set(df[col].dropna().unique())
    missing = valid - used
    print(f"  {col.upper()}: used={sorted(used)}, MISSING values={sorted(missing)}")

# 2. Intent field: invalid values
print("\n--- Issue 2: Intent field contamination ---")
valid_intents = {'FACTUAL', 'ANALYTICAL', 'SYNTHETIC', 'STRATEGIC'}
for val, count in df['intent'].value_counts().items():
    status = "OK" if val in valid_intents else "INVALID"
    print(f"  '{val}': {count} [{status}]")
invalid_intent_count = df[~df['intent'].isin(valid_intents) & df['intent'].notna()].shape[0]
print(f"  Total invalid: {invalid_intent_count}")

# 3. Task type: check for pipe-separated and missing types
print("\n--- Issue 3: Task type issues ---")
valid_tasks = {'classification', 'generation', 'reasoning', 'coding', 'summarisation', 'sparql_generation', 'formatting'}
for val, count in df['task_type'].value_counts().items():
    status = "OK" if val in valid_tasks else "NEEDS REVIEW"
    print(f"  '{val}': {count} [{status}]")

# 4. reasoning_chain_detected should be bool, is string
print("\n--- Issue 4: reasoning_chain_detected type ---")
print(f"  dtype: {df['reasoning_chain_detected'].dtype}")
print(f"  values: {df['reasoning_chain_detected'].value_counts().to_dict()}")

# 5. Complexity vs tier mapping
print("\n--- Issue 5: complexity field vs derived tier ---")
cs = df['d1']*0.35 + df['d2']*0.20 + df['d3']*0.20 + df['d4']*0.15 + df['d5']*0.10
df['derived_cs'] = cs
df['derived_tier'] = cs.apply(lambda s: 'T1' if s < 0.4 else ('T2' if s < 0.7 else 'T3'))
tier_to_complexity = {'T1': 'low', 'T2': 'medium', 'T3': 'high'}
df['expected_complexity'] = df['derived_tier'].map(tier_to_complexity)
mismatch = df[df['complexity'] != df['expected_complexity']]
print(f"  Complexity label vs derived tier mismatches: {len(mismatch)}/{len(df)}")
print(f"  Crosstab:")
print(pd.crosstab(df['complexity'], df['derived_tier']))

# 6. Duplicates
print("\n--- Issue 6: Duplicate prompts ---")
dupes = df[df['prompt'].duplicated(keep=False)]
print(f"  Duplicate prompt rows: {len(dupes)} ({df['prompt'].duplicated().sum()} non-first)")

# 7. Null rows
print("\n--- Issue 7: Null summary ---")
nulls = df.isna().sum()
print(nulls[nulls > 0])

# 8. D-score imbalance
print("\n--- Issue 8: D-score class imbalance ---")
for col in ['d1', 'd2', 'd3', 'd4', 'd5']:
    vc = df[col].value_counts()
    majority = vc.max()
    minority = vc.min()
    print(f"  {col.upper()}: majority={majority}, minority={minority}, ratio={majority/max(minority,1):.1f}x")

# 9. prompt_type distribution
print("\n--- Issue 9: prompt_type (extra column) ---")
print(df['prompt_type'].value_counts().to_string())

# 10. good_prompt vs prompt identity
print("\n--- Issue 10: prompt == good_prompt? ---")
same = (df['prompt'] == df['good_prompt']).sum()
print(f"  prompt == good_prompt: {same}/{len(df)} ({same/len(df)*100:.1f}%)")

# 11. research_signals parsing
print("\n--- Issue 11: research_signals format ---")
empty_signals = df[df['research_signals'] == '[]'].shape[0]
print(f"  Empty signals: {empty_signals}/{len(df)} ({empty_signals/len(df)*100:.1f}%)")
non_empty = df[df['research_signals'] != '[]']
print(f"  Non-empty: {len(non_empty)}")
# Try parsing
try:
    parsed = non_empty['research_signals'].apply(ast.literal_eval)
    flat = [item for sublist in parsed for item in sublist]
    print(f"  Unique signal types: {set(flat)}")
except Exception as e:
    print(f"  Parse error: {e}")

# 12. Confidence at 0.0
print("\n--- Issue 12: Low confidence rows ---")
low_conf = df[df['confidence'] <= 0.5]
print(f"  Confidence <= 0.5: {len(low_conf)} rows")
if len(low_conf) > 0:
    print(f"  Their intents: {low_conf['intent'].value_counts().to_dict()}")
