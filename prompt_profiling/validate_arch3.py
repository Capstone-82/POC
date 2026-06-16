# ── Imports ──────────────────────────────────────────────────────────────────
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity

from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 100)

print("All imports successful.")

def complexity_score(d1, d2, d3, d4, d5):
    """Compute weighted complexity score from five dimension scores."""
    return (d1 * 0.35) + (d2 * 0.20) + (d3 * 0.20) + (d4 * 0.15) + (d5 * 0.10)


def get_tier(score):
    """Map a complexity score to a tier label."""
    if score < 0.40:
        return 'T1'
    elif score < 0.70:
        return 'T2'
    else:
        return 'T3'


# Quick sanity check
assert get_tier(0.10) == 'T1'
assert get_tier(0.50) == 'T2'
assert get_tier(0.85) == 'T3'
print("Helper functions defined and verified.")

# ── Load CSV ─────────────────────────────────────────────────────────────────
df = pd.read_csv('dataset_prompt_profiling.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()

# ── Basic data checks ────────────────────────────────────────────────────────
print(f"Null counts:\n{df.isnull().sum()}")
print(f"\nDimension score ranges:")
for col in ['d1', 'd2', 'd3', 'd4', 'd5']:
    print(f"  {col}: [{df[col].min()}, {df[col].max()}]")

# ── Tier extraction from id column ───────────────────────────────────────────
def extract_tier_from_id(row_id):
    """Extract tier from id using regex pattern _(t[123])_ (case-insensitive)."""
    match = re.search(r'_(t[123])_', str(row_id), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


df['tier_extracted'] = df['id'].apply(extract_tier_from_id)

extracted_count = df['tier_extracted'].notna().sum()
missing_count = df['tier_extracted'].isna().sum()
print(f"Tier extracted from id: {extracted_count} rows")
print(f"Tier NOT extractable:   {missing_count} rows")

# ── Compute complexity score & derive tier for ALL rows ─────────────────────
df['complexity_score'] = df.apply(
    lambda r: complexity_score(r['d1'], r['d2'], r['d3'], r['d4'], r['d5']), axis=1
)
df['tier_derived'] = df['complexity_score'].apply(get_tier)

# For rows where extraction failed → use derived tier
df['tier'] = df['tier_extracted'].fillna(df['tier_derived'])

print(f"Tier distribution (before mismatch fix):")
print(df['tier'].value_counts().sort_index())

# ── Fix label/score mismatches ───────────────────────────────────────────────
# Trust scores over labels: if extracted tier != derived tier, override with derived
mismatch_mask = (df['tier_extracted'].notna()) & (df['tier_extracted'] != df['tier_derived'])
mismatch_count = mismatch_mask.sum()
print(f"Rows where extracted tier != derived tier: {mismatch_count}")

if mismatch_count > 0:
    print("\nSample mismatches:")
    print(df.loc[mismatch_mask, ['id', 'tier_extracted', 'tier_derived', 'complexity_score']].head(10))

# Override: trust derived tier for ALL rows
df['tier'] = df['tier_derived']

print(f"\nFinal tier distribution (after mismatch fix):")
print(df['tier'].value_counts().sort_index())
print(f"\nTotal rows: {len(df)}")

# ── Summary after preprocessing ──────────────────────────────────────────────
print(f"DataFrame shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
df[['id', 'prompt', 'd1', 'd2', 'd3', 'd4', 'd5', 'complexity_score', 'tier']].head(10)

def get_d4(prompt):
    """Rule-based D4 score using case-insensitive substring matching.
    
    Priority order (first match wins):
      1.0  → market research, competitive analysis, analyst report,
             industry trends, industry report
      0.75 → latest, recent, current, pricing, market,
             forecast, survey, compare vendors
      0.50 → compare, evaluate, benchmark, analysis, study
      0.0  → otherwise
    """
    text = prompt.lower()

    # Priority 1: D4 = 1.0
    keywords_1 = ["market research", "competitive analysis", "analyst report",
                  "industry trends", "industry report"]
    if any(kw in text for kw in keywords_1):
        return 1.0

    # Priority 2: D4 = 0.75
    keywords_075 = ["latest", "recent", "current", "pricing", "market",
                    "forecast", "survey", "compare vendors"]
    if any(kw in text for kw in keywords_075):
        return 0.75

    # Priority 3: D4 = 0.50
    keywords_050 = ["compare", "evaluate", "benchmark", "analysis", "study"]
    if any(kw in text for kw in keywords_050):
        return 0.50

    # Default
    return 0.0


# Apply rule engine to all prompts
df['d4_rule'] = df['prompt'].apply(get_d4)

# Compute MAE between d4_rule and ground-truth d4
d4_mae = mean_absolute_error(df['d4'], df['d4_rule'])
print(f"D4 Rule Engine MAE vs ground-truth: {d4_mae:.4f}")

# Show distribution of rule-based D4 scores
print(f"\nD4 rule-based score distribution:")
print(df['d4_rule'].value_counts().sort_index())

# Show some examples
print(f"\nSample rows with D4 comparison:")
df[['prompt', 'd4', 'd4_rule']].sample(5, random_state=42)

# ── Generate embeddings ──────────────────────────────────────────────────────
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

prompts = df['prompt'].tolist()
embeddings = embed_model.encode(prompts, show_progress_bar=True, batch_size=64)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dtype:  {embeddings.dtype}")
print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")

# ── Prepare features and targets ─────────────────────────────────────────────
X = embeddings  # (470, 384)
y = df[['d1', 'd2', 'd3', 'd5']].values  # D4 excluded (rule-based)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Target columns: d1, d2, d3, d5")

# ── Train/Test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index,
    test_size=0.20,
    stratify=df['tier'],
    random_state=42
)

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}  |  y_test: {y_test.shape}")
print(f"\nTier distribution in train set:")
print(df.loc[idx_train, 'tier'].value_counts().sort_index())
print(f"\nTier distribution in test set:")
print(df.loc[idx_test, 'tier'].value_counts().sort_index())

# ── Scale features ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled  shape: {X_test_scaled.shape}")
print(f"\nPost-scaling stats (train):")
print(f"  Mean ~ {X_train_scaled.mean():.6f}")
print(f"  Std  ~ {X_train_scaled.std():.6f}")

# ── Train model ──────────────────────────────────────────────────────────────
xgb_base = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

multi_model = MultiOutputRegressor(estimator=xgb_base)
multi_model.fit(X_train_scaled, y_train)

print("Model training complete.")
print(f"Number of sub-estimators: {len(multi_model.estimators_)}")
print(f"Targets: D1, D2, D3, D5")

# ── Predict & clip ───────────────────────────────────────────────────────────
y_pred_raw = multi_model.predict(X_test_scaled)
y_pred = np.clip(y_pred_raw, 0.0, 1.0)

print(f"Predictions shape: {y_pred.shape}")
print(f"Prediction range before clip: [{y_pred_raw.min():.4f}, {y_pred_raw.max():.4f}]")
print(f"Prediction range after clip:  [{y_pred.min():.4f}, {y_pred.max():.4f}]")

# Preview
pred_df = pd.DataFrame(y_pred, columns=['d1_pred', 'd2_pred', 'd3_pred', 'd5_pred'])
pred_df.head()

# ── Per-dimension metrics ────────────────────────────────────────────────────
dim_names = ['D1', 'D2', 'D3', 'D5']
metrics_results = {}

print("Per-Dimension Metrics (Test Set)")
print("=" * 50)
for i, dim in enumerate(dim_names):
    mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2   = r2_score(y_test[:, i], y_pred[:, i])
    metrics_results[dim] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    print(f"  {dim}:  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

print("=" * 50)

# ── Tier prediction ──────────────────────────────────────────────────────────
# ACTUAL tier: ground-truth d1-d5 → complexity_score → get_tier
test_df = df.loc[idx_test].copy().reset_index(drop=True)

actual_complexity = test_df.apply(
    lambda r: complexity_score(r['d1'], r['d2'], r['d3'], r['d4'], r['d5']), axis=1
)
actual_tiers = actual_complexity.apply(get_tier)

# PREDICTED tier: predicted D1/D2/D3/D5 + d4_rule for D4 → complexity_score → get_tier
pred_d4_rule = test_df['d4_rule'].values

pred_complexity = np.array([
    complexity_score(y_pred[i, 0], y_pred[i, 1], y_pred[i, 2], pred_d4_rule[i], y_pred[i, 3])
    for i in range(len(y_pred))
])
pred_tiers = pd.Series(pred_complexity).apply(get_tier)

# Tier accuracy
tier_acc = accuracy_score(actual_tiers, pred_tiers)
print(f"Tier Accuracy: {tier_acc:.4f} ({tier_acc*100:.2f}%)")

# ── Confusion matrix & classification report ─────────────────────────────────
tier_labels = ['T1', 'T2', 'T3']

cm = confusion_matrix(actual_tiers, pred_tiers, labels=tier_labels)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=tier_labels, columns=tier_labels))

print(f"\nClassification Report:")
print(classification_report(actual_tiers, pred_tiers, labels=tier_labels, zero_division=0))

def profile_prompt(prompt_text, embed_model, scaler, model, df_train, X_train_scaled):
    """
    Profile a single prompt through the full Architecture 3 pipeline.
    
    Parameters
    ----------
    prompt_text : str
        The raw prompt to profile.
    embed_model : SentenceTransformer
        The embedding model (all-MiniLM-L6-v2).
    scaler : StandardScaler
        Fitted scaler for feature normalization.
    model : MultiOutputRegressor
        Trained multi-output XGBoost model.
    df_train : pd.DataFrame
        Training dataframe (for retrieving similar prompts).
    X_train_scaled : np.ndarray
        Scaled training embeddings (for cosine similarity search).
    
    Returns
    -------
    dict with keys:
        tier, complexity_score, d1, d2, d3, d4_rule, d4_final, d5,
        tier_probabilities (raw complexity score), top3_similar_prompts
    """
    # 1. Embed the prompt
    emb = embed_model.encode([prompt_text])
    
    # 2. Scale
    emb_scaled = scaler.transform(emb)
    
    # 3. Predict D1, D2, D3, D5 via ML model
    preds = model.predict(emb_scaled)
    preds = np.clip(preds, 0.0, 1.0)
    d1_pred, d2_pred, d3_pred, d5_pred = preds[0]
    
    # 4. D4 via rule engine
    d4_rule_val = get_d4(prompt_text)
    
    # 5. D4 final: max(predicted_d4_from_knn_fallback, d4_rule)
    #    Since this architecture has no KNN, d4_final = d4_rule
    d4_final = d4_rule_val
    
    # 6. Complexity score & tier
    cs = complexity_score(d1_pred, d2_pred, d3_pred, d4_final, d5_pred)
    tier = get_tier(cs)
    
    # 7. Top 3 similar prompts (cosine similarity on scaled embeddings)
    similarities = cosine_similarity(emb_scaled, X_train_scaled)[0]
    top3_idx = np.argsort(similarities)[-3:][::-1]
    top3_prompts = df_train.iloc[top3_idx]['prompt'].tolist()
    top3_scores = similarities[top3_idx].tolist()
    
    return {
        'tier': tier,
        'tier_probabilities': cs,  # raw complexity score (regression, not classification)
        'd1': round(float(d1_pred), 4),
        'd2': round(float(d2_pred), 4),
        'd3': round(float(d3_pred), 4),
        'd4_rule': d4_rule_val,
        'd4_final': d4_final,
        'd5': round(float(d5_pred), 4),
        'complexity_score': round(float(cs), 4),
        'top3_similar_prompts': [
            {'prompt': p, 'similarity': round(s, 4)}
            for p, s in zip(top3_prompts, top3_scores)
        ]
    }


print("Inference function defined.")

# ── Prepare training dataframe for inference ─────────────────────────────────
df_train = df.loc[idx_train].reset_index(drop=True)
print(f"df_train shape: {df_train.shape}")

# ── Test Prompt 1 ────────────────────────────────────────────────────────────
test_prompts = [
    "why is our AWS bill so high this month",
    "build a full GenAI cost attribution platform with market research across AWS, Azure and GCP pricing",
    "set up SSO between okta and our internal tool"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"TEST PROMPT {i}: \"{prompt}\"")
    print(f"{'='*80}")
    
    result = profile_prompt(
        prompt_text=prompt,
        embed_model=embed_model,
        scaler=scaler,
        model=multi_model,
        df_train=df_train,
        X_train_scaled=X_train_scaled
    )
    
    print(f"  Tier:             {result['tier']}")
    print(f"  Complexity Score: {result['complexity_score']}")
    print(f"  D1 (Scope):       {result['d1']}")
    print(f"  D2 (Specificity): {result['d2']}")
    print(f"  D3 (Cross-domain):{result['d3']}")
    print(f"  D4 (Rule-based):  {result['d4_rule']}")
    print(f"  D4 (Final):       {result['d4_final']}")
    print(f"  D5 (Output):      {result['d5']}")
    print(f"\n  Top 3 Similar Training Prompts:")
    for j, sim in enumerate(result['top3_similar_prompts'], 1):
        print(f"    {j}. [sim={sim['similarity']:.4f}] {sim['prompt'][:100]}...")

# ── 8.1 Tier Distribution: Actual vs Predicted ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

actual_counts = actual_tiers.value_counts().reindex(tier_labels, fill_value=0)
pred_counts = pred_tiers.value_counts().reindex(tier_labels, fill_value=0)

x = np.arange(len(tier_labels))
width = 0.35

bars1 = ax.bar(x - width/2, actual_counts.values, width, label='Actual', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, pred_counts.values, width, label='Predicted', color='#FF9800', alpha=0.85)

ax.set_xlabel('Tier', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('8.1 — Tier Distribution: Actual vs Predicted (Test Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tier_labels)
ax.legend()

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('viz_8_1_tier_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: viz_8_1_tier_distribution.png")

# ── 8.2 Confusion Matrix Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=tier_labels, yticklabels=tier_labels,
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_xlabel('Predicted Tier', fontsize=12)
ax.set_ylabel('Actual Tier', fontsize=12)
ax.set_title(f'8.2 — Tier Confusion Matrix (Accuracy: {tier_acc*100:.1f}%)',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('viz_8_2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: viz_8_2_confusion_matrix.png")

# ── 8.3 Actual vs Predicted Scatter Plots (2×2 grid) ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('8.3 — Actual vs Predicted (Test Set)', fontsize=16, fontweight='bold', y=1.02)

colors = ['#1E88E5', '#43A047', '#E53935', '#8E24AA']

for i, (dim, color) in enumerate(zip(dim_names, colors)):
    ax = axes[i // 2][i % 2]
    
    ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, color=color, edgecolors='white', s=50)
    
    # Perfect prediction line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Perfect')
    
    ax.set_xlabel(f'Actual {dim}', fontsize=11)
    ax.set_ylabel(f'Predicted {dim}', fontsize=11)
    ax.set_title(f'{dim}  (R²={metrics_results[dim]["R²"]:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz_8_3_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: viz_8_3_scatter_plots.png")

# ── 8.4 Distribution Plots: Actual vs Predicted/Rule-based ───────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('8.4 — Score Distributions: Actual vs Predicted (Test Set)',
             fontsize=16, fontweight='bold', y=1.02)

# D1, D2, D3, D5 — ML predicted
all_dims = ['D1', 'D2', 'D3', 'D5', 'D4']
plot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

for idx, (dim, pos) in enumerate(zip(all_dims, plot_positions)):
    ax = axes[pos[0]][pos[1]]
    
    if dim == 'D4':
        # D4: actual vs rule-based
        ax.hist(test_df['d4'], bins=20, alpha=0.6, label='Actual D4', color='#1E88E5', edgecolor='white')
        ax.hist(test_df['d4_rule'], bins=20, alpha=0.6, label='Rule-based D4', color='#FF9800', edgecolor='white')
        ax.set_title(f'{dim} (Actual vs Rule-based)', fontsize=12, fontweight='bold')
    else:
        # ML dimensions
        ml_idx = ['D1', 'D2', 'D3', 'D5'].index(dim)
        ax.hist(y_test[:, ml_idx], bins=20, alpha=0.6, label=f'Actual {dim}', color='#1E88E5', edgecolor='white')
        ax.hist(y_pred[:, ml_idx], bins=20, alpha=0.6, label=f'Predicted {dim}', color='#FF9800', edgecolor='white')
        ax.set_title(f'{dim} (Actual vs Predicted)', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Score', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Hide unused subplot
axes[1][2].set_visible(False)

plt.tight_layout()
plt.savefig('viz_8_4_distribution_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: viz_8_4_distribution_plots.png")

# ── 8.5 XGBoost Feature Importance (Top 20 Embedding Dims) ───────────────────
# Average feature importance across all 4 regressors
importances = np.zeros(X_train_scaled.shape[1])
for est in multi_model.estimators_:
    importances += est.feature_importances_
importances /= len(multi_model.estimators_)

# Get top 20
top20_idx = np.argsort(importances)[-20:][::-1]
top20_vals = importances[top20_idx]
top20_labels = [f'Emb_{i}' for i in top20_idx]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(top20_labels)), top20_vals[::-1], color='#1E88E5', alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(top20_labels)))
ax.set_yticklabels(top20_labels[::-1], fontsize=10)
ax.set_xlabel('Average Feature Importance', fontsize=12)
ax.set_title('8.5 — XGBoost Avg Feature Importance (Top 20 Embedding Dimensions)',
             fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('viz_8_5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: viz_8_5_feature_importance.png")

# ── Save models ──────────────────────────────────────────────────────────────
joblib.dump(multi_model, 'xgb_multioutput_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Models saved:")
print("  → xgb_multioutput_model.joblib")
print("  → scaler.joblib")

# ── Final Summary Table ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("        FINAL SUMMARY — Architecture 3 Results")
print("=" * 60)
print(f"{'Dimension':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 40)

for dim in dim_names:
    m = metrics_results[dim]
    print(f"{dim:<12} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R²']:>8.4f}")

print("-" * 40)
print(f"{'Tier Acc.':<12} {tier_acc*100:>7.2f}%")
print(f"{'D4 Rule MAE':<12} {d4_mae:>8.4f}")
print("=" * 60)

# ── Verify saved models load correctly ───────────────────────────────────────
loaded_model = joblib.load('xgb_multioutput_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# Quick verification: predict on first test sample
verify_pred = loaded_model.predict(loaded_scaler.transform(X_test[:1]))
verify_pred = np.clip(verify_pred, 0.0, 1.0)
print(f"Model load verification — prediction on first test sample:")
print(f"  D1={verify_pred[0][0]:.4f}, D2={verify_pred[0][1]:.4f}, "
      f"D3={verify_pred[0][2]:.4f}, D5={verify_pred[0][3]:.4f}")
print("\n✅ Architecture 3 pipeline complete!")
