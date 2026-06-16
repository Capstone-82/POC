import pandas as pd
df = pd.read_csv('dataset_prompt_profiling_v4.csv')
print('Dataset:', len(df), 'rows')

cs = df['d1']*0.35 + df['d2']*0.20 + df['d3']*0.20 + df['d4']*0.15 + df['d5']*0.10
df['cs'] = cs
df['tier'] = cs.apply(lambda s: 'T1' if s < 0.4 else ('T2' if s < 0.7 else 'T3'))

print('\nTier distribution:')
print(df['tier'].value_counts().sort_index())

# Boundary analysis
b12 = df[(df['cs'] >= 0.35) & (df['cs'] < 0.45)]
b23 = df[(df['cs'] >= 0.65) & (df['cs'] < 0.75)]
print(f'\nT1/T2 boundary [0.35,0.45): {len(b12)} prompts')
print(f'T2/T3 boundary [0.65,0.75): {len(b23)} prompts')

print('\nT2/T3 boundary tier split:')
print(b23.groupby('tier').size())

# How many prompts are within 1 dimension error of flipping?
# A single D score error of 0.25 can shift cs by 0.25*0.35=0.0875 (D1) or 0.25*0.10=0.025 (D5)
print('\n--- Flip vulnerability analysis ---')
for name, boundary, below, above in [('T1/T2', 0.40, 'T1', 'T2'), ('T2/T3', 0.70, 'T2', 'T3')]:
    within_1err = df[abs(df['cs'] - boundary) < 0.0875]
    print(f'{name} boundary: {len(within_1err)} prompts within 1 D1-error of flipping')
    within_half = df[abs(df['cs'] - boundary) < 0.05]
    print(f'{name} boundary: {len(within_half)} prompts within 0.05 of boundary')

# T2/T3 confusion source: how many T2 have cs > 0.60? how many T3 have cs < 0.80?
t2_high = df[(df['tier'] == 'T2') & (df['cs'] >= 0.60)]
t3_low = df[(df['tier'] == 'T3') & (df['cs'] < 0.80)]
print(f'\nT2 with cs >= 0.60 (vulnerable to T3 misclassification): {len(t2_high)}')
print(f'T3 with cs < 0.80 (vulnerable to T2 misclassification): {len(t3_low)}')

# Phrasing style of the error-prone zone
print('\nPhrasing style in T2/T3 boundary zone:')
print(b23.groupby('phrasing_style').size())
