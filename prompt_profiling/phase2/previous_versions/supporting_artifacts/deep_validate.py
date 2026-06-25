import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('phase2/prompt_classifier_phase1_phase2_merged_cleaned.csv')
p1 = df[df['source']=='phase1']

# Issue A: question-mark prompts classified as 'classification'
q = p1[(p1['prompt'].str.strip().str.endswith('?')) & (p1['task_type']=='classification')]
print("="*80)
print("ISSUE A: Phase 1 question prompts misclassified as 'classification'")
print("="*80)
print(f"Count: {len(q)}")
for _, r in q.iterrows():
    print(f"  [{r['intent']}] D1={r['d1']} | {r['prompt'][:90]}")
print()
print("By intent:", q['intent'].value_counts().to_dict())

# Issue B: Phase 1 duplicates detail
print()
print("="*80)
print("ISSUE B: Phase 1 duplicate prompts")
print("="*80)
dupes = p1[p1['prompt'].duplicated(keep=False)]
print(f"Rows with duplicate prompts: {len(dupes)}")
print(f"Unique duplicated texts: {dupes['prompt'].nunique()}")
for p in list(dupes['prompt'].unique())[:5]:
    subset = dupes[dupes['prompt']==p]
    ids = list(subset['id'])
    d1s = list(subset['d1'])
    print(f"  \"{p[:70]}...\"")
    print(f"    appears {len(subset)}x | ids={ids} | d1_scores={d1s}")

# Issue C: D3 imbalance detail
print()
print("="*80)
print("ISSUE C: D3 still heavily imbalanced after merge")
print("="*80)
for col in ['d1','d2','d3','d4','d5']:
    vc = df[col].value_counts().sort_index()
    pct_majority = vc.max() / len(df) * 100
    print(f"  {col}: majority={vc.idxmax()} at {pct_majority:.1f}%")
    print(f"       {vc.to_dict()}")

# Issue D: reasoning_chain_detected imbalance
print()
print("="*80)
print("ISSUE D: reasoning_chain_detected imbalance")
print("="*80)
print(df['reasoning_chain_detected'].value_counts().to_dict())
print(f"True ratio: {df['reasoning_chain_detected'].sum()/len(df)*100:.1f}%")

# Issue E: confidence column state
print()
print("="*80)
print("ISSUE E: confidence column analysis")
print("="*80)
print(f"Nulls: {df['confidence'].isna().sum()}")
print(f"Non-null: {df['confidence'].notna().sum()}")
print(f"Phase 1 confidence nulls: {p1['confidence'].isna().sum()} (expected: {len(p1)})")
p2 = df[df['source']=='phase2']
print(f"Phase 2 confidence nulls: {p2['confidence'].isna().sum()}")
print(f"Phase 2 confidence distribution:")
print(p2['confidence'].describe())

# Issue F: cross-check intent vs task_type coherence
print()
print("="*80)
print("ISSUE F: Intent vs task_type coherence")
print("="*80)
print(pd.crosstab(df['intent'], df['task_type']))
