import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('phase2/prompt_classifier_phase2_v5_dataset.csv')

# 1. Boundary analysis: how many prompts sit near T1/T2 and T2/T3 boundaries?
cs = df['d1']*0.35 + df['d2']*0.20 + df['d3']*0.20 + df['d4']*0.15 + df['d5']*0.10
df['cs'] = cs

print("="*80)
print("BOUNDARY ANALYSIS")
print("="*80)
# T1/T2 boundary: 0.35-0.44
t12_boundary = df[(cs >= 0.35) & (cs <= 0.44)]
print(f"T1/T2 boundary (0.35-0.44): {len(t12_boundary)} prompts ({len(t12_boundary)/len(df)*100:.1f}%)")

# T2/T3 boundary: 0.65-0.74
t23_boundary = df[(cs >= 0.65) & (cs <= 0.74)]
print(f"T2/T3 boundary (0.65-0.74): {len(t23_boundary)} prompts ({len(t23_boundary)/len(df)*100:.1f}%)")

# Core zones (safe predictions)
safe_t1 = df[cs < 0.30]
safe_t2 = df[(cs >= 0.45) & (cs <= 0.64)]
safe_t3 = df[cs >= 0.75]
print(f"\nSafe T1 (cs < 0.30): {len(safe_t1)}")
print(f"Safe T2 (0.45-0.64): {len(safe_t2)}")
print(f"Safe T3 (cs >= 0.75): {len(safe_t3)}")
print(f"Safe total: {len(safe_t1)+len(safe_t2)+len(safe_t3)}/{len(df)} ({(len(safe_t1)+len(safe_t2)+len(safe_t3))/len(df)*100:.1f}%)")

# 2. D1 confusion potential: how many prompts differ by exactly one D1 step?
print("\n" + "="*80)
print("D1 SCORE DISTRIBUTION BY TIER")
print("="*80)
print(pd.crosstab(df['tier'], df['d1']))

print("\n" + "="*80)
print("D2 SCORE DISTRIBUTION BY TIER")
print("="*80)
print(pd.crosstab(df['tier'], df['d2']))

# 3. Source quality analysis
print("\n" + "="*80)
print("ACCURACY CEILING: LABEL NOISE INDICATORS")
print("="*80)
# Prompts where D1 and intent don't align
intent_map = {'FACTUAL': [0.0, 0.25], 'ANALYTICAL': [0.5], 'SYNTHETIC': [0.75], 'STRATEGIC': [1.0]}
misaligned = 0
for _, row in df.iterrows():
    expected = intent_map.get(row['intent'], [])
    if row['d1'] not in expected:
        misaligned += 1
print(f"D1 ↔ Intent misalignment: {misaligned}/{len(df)} ({misaligned/len(df)*100:.1f}%)")

# 4. Complexity score distribution (histogram)
print("\n" + "="*80)
print("COMPLEXITY SCORE BUCKETS")
print("="*80)
buckets = pd.cut(cs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
print(buckets.value_counts().sort_index())

# 5. Feature density: average prompt length by tier
print("\n" + "="*80)
print("PROMPT LENGTH BY TIER")
print("="*80)
df['prompt_len'] = df['prompt'].str.len()
print(df.groupby('tier')['prompt_len'].describe()[['mean','std','min','max']].to_string())

# 6. Source breakdown by tier
print("\n" + "="*80)
print("SOURCE x TIER")
print("="*80)
print(pd.crosstab(df['source'], df['tier']))

# 7. How many unique complexity scores exist?
print(f"\nUnique complexity scores: {cs.nunique()}")
print("Top 15:")
print(cs.value_counts().head(15))
