import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('phase2/prompt_example_classifier_bedrock_output.csv')
print(f"Shape: {df.shape}")
print(f"\n{'='*80}")
print("COLUMN DETAILS")
print(f"{'='*80}")
for c in df.columns:
    nulls = df[c].isna().sum()
    uniq = df[c].nunique()
    dtype = df[c].dtype
    print(f"\n  [{c}]")
    print(f"    dtype={dtype}, nulls={nulls} ({nulls/len(df)*100:.1f}%), unique={uniq}")
    if uniq <= 15 and dtype == 'object':
        print(f"    values: {dict(df[c].value_counts())}")
    elif dtype in ['float64', 'int64'] and uniq <= 20:
        print(f"    values: {dict(df[c].value_counts().sort_index())}")
    elif dtype == 'object':
        print(f"    sample: {df[c].dropna().iloc[0][:100] if len(df[c].dropna()) > 0 else 'N/A'}")

print(f"\n{'='*80}")
print("SCORE DISTRIBUTIONS")
print(f"{'='*80}")
for col in ['d1', 'd2', 'd3', 'd4', 'd5']:
    if col in df.columns:
        print(f"\n  {col.upper()}:")
        print(f"    {dict(df[col].value_counts().sort_index())}")
        valid = {0.0, 0.25, 0.5, 0.75, 1.0}
        invalid = set(df[col].dropna().unique()) - valid
        if invalid:
            print(f"    *** INVALID VALUES: {invalid}")

print(f"\n{'='*80}")
print("NEW FIELD DISTRIBUTIONS")
print(f"{'='*80}")
for col in ['intent', 'task_type', 'reasoning_chain_detected', 'research_signals', 'confidence']:
    if col in df.columns:
        print(f"\n  {col}:")
        if df[col].dtype == 'object':
            print(f"    {dict(df[col].value_counts())}")
        else:
            print(f"    min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.3f}")
            print(f"    {dict(df[col].value_counts().sort_index().head(10))}")

# Check for columns that might map to the required fields
print(f"\n{'='*80}")
print("COLUMN NAME MATCHING")
print(f"{'='*80}")
required = ['intent', 'task_type', 'reasoning_chain_detected', 'research_signals', 'confidence']
for r in required:
    matches = [c for c in df.columns if r.lower() in c.lower()]
    print(f"  '{r}' -> matches: {matches if matches else 'NONE'}")
