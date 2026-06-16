"""
Create Architecture3_v4_Final.ipynb from the v3 notebook without executing it.
"""

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
V3_NOTEBOOK = BASE_DIR / "Architecture3_v3_Final.ipynb"
V4_NOTEBOOK = BASE_DIR / "Architecture3_v4_Final.ipynb"


def src(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def set_source(nb: dict, idx: int, text: str) -> None:
    nb["cells"][idx]["source"] = src(text)


def clear_outputs(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


HELPERS = r'''
# -- Score <-> class mappings -------------------------------------------------
SCORE_TO_CLASS = {0.0: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.0: 4}
CLASS_TO_SCORE = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
TIER_TO_INT = {'T1': 0, 'T2': 1, 'T3': 2}
INT_TO_TIER = {0: 'T1', 1: 'T2', 2: 'T3'}

DIMENSION_LABELS = {
    'D1': 'Semantic Complexity',
    'D2': 'Domain Specificity',
    'D3': 'Output Formality',
    'D4': 'Research Dependency',
    'D5': 'Context Requirement',
}

BOUNDARY_WINDOWS = ((0.35, 0.45), (0.65, 0.75))


def complexity_score(d1, d2, d3, d4, d5):
    """Compute weighted complexity score from the source rubric."""
    return (d1 * 0.35) + (d2 * 0.20) + (d3 * 0.20) + (d4 * 0.15) + (d5 * 0.10)


def get_tier(score):
    """Map complexity score to tier."""
    if score < 0.40:
        return 'T1'
    if score < 0.70:
        return 'T2'
    return 'T3'


def is_boundary_score(score):
    """Return True when a score is close to a tier boundary."""
    return any(lo <= score < hi for lo, hi in BOUNDARY_WINDOWS)


def choose_final_tier(stage1_int, derived_int, derived_score):
    """Boundary-aware tier policy.

    Rules:
      1. If both models agree, keep that tier.
      2. Protect likely T3 prompts from under-routing.
      3. Near tier boundaries, trust the direct tier classifier.
      4. Away from boundaries, trust the dimension-derived score.
    """
    if stage1_int == derived_int:
        return stage1_int

    t3 = TIER_TO_INT['T3']
    if stage1_int == t3 and derived_score >= 0.65:
        return stage1_int
    if derived_int == t3 and derived_score >= 0.70:
        return derived_int

    if is_boundary_score(derived_score):
        return stage1_int

    return derived_int


# Sanity checks
assert get_tier(0.10) == 'T1'
assert get_tier(0.55) == 'T2'
assert get_tier(0.85) == 'T3'
assert choose_final_tier(TIER_TO_INT['T2'], TIER_TO_INT['T3'], 0.71) == TIER_TO_INT['T3']
print("Helper functions verified.")
'''


FEATURES = r'''
def extract_handcrafted_features(prompt, phrasing_style=None):
    """Extract rubric-aligned structural features.

    D3 is Output Formality.
    D5 is Context Requirement.
    D4 is Research Dependency.
    """
    text = str(prompt).lower()
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    attachment_terms = [
        'attached', 'uploaded', 'pasted', 'screenshot', 'file attached',
        'the attached', 'i have uploaded', "i've uploaded", 'uploaded corpus',
        'context bundle', 'full archive', 'evidence folder'
    ]
    formal_outputs = [
        'requirements document', 'technical specification', 'specification',
        'executive summary', 'roadmap', 'framework', 'architecture',
        'policy', 'checklist', 'runbook', 'report', 'proposal', 'strategy',
        'matrix', 'blueprint', 'playbook', 'operating model'
    ]
    package_outputs = [
        'package', 'appendix', 'annex', 'toc', 'table of contents',
        'evidence mapping', 'risk register', 'cost model', 'governance controls'
    ]

    features = {
        # Text statistics
        'word_count': len(words),
        'char_count': len(text),
        'sentence_count': max(len(sentences), 1),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'question_marks': text.count('?'),
        'comma_count': text.count(','),

        # D5: Context requirement
        'has_attachment': int(any(kw in text for kw in attachment_terms)),
        'provided_artifact_count': sum(1 for kw in [
            'diagram', 'log', 'export', 'contract', 'catalog', 'policy',
            'evidence', 'inventory', 'ticket', 'postmortem', 'dashboard',
            'configuration', 'archive', 'corpus'
        ] if kw in text),
        'large_context_signal': int(any(kw in text for kw in [
            'full uploaded', 'complete uploaded', 'full archive', 'complete corpus',
            'three years', 'two years', '18 months', 'multi-document',
            'all documents', 'extended conversation', 'context bundle'
        ])),
        'multi_document_signal': int(any(kw in text for kw in [
            'and', 'multiple', 'across', 'bundle', 'corpus', 'archive'
        ]) and any(kw in text for kw in attachment_terms)),

        # D3: Output formality
        'has_formal_deliverable': int(any(kw in text for kw in formal_outputs)),
        'has_report_package': int(any(kw in text for kw in package_outputs)),
        'has_long_output_signal': int(any(kw in text for kw in [
            'comprehensive', 'enterprise-grade', 'formal', 'detailed',
            'full', 'complete', 'end-to-end', 'appendix-level'
        ])),
        'structured_section_count': sum(1 for kw in [
            'executive summary', 'gap analysis', 'roadmap', 'risk register',
            'implementation plan', 'success metrics', 'dependencies',
            'recommendations', 'findings', 'appendix'
        ] if kw in text),

        # D1: Semantic complexity
        'has_scope_words': int(any(kw in text for kw in [
            'multi-stage', 'cross-domain', 'strategic', 'synthesize',
            'root causes', 'prioritized', 'tradeoffs', 'decision',
            'governance', 'operating model'
        ])),
        'action_verb_count': sum(1 for kw in [
            'build', 'design', 'evaluate', 'draft', 'develop', 'create',
            'produce', 'generate', 'implement', 'establish', 'restructure',
            'migrate', 'optimize', 'analyze', 'recommend'
        ] if kw in text),
        'multi_stage_signal': int(any(kw in text for kw in [
            'phase', 'phased', 'step-by-step', 'roadmap', 'workflow',
            'implementation', 'migration', 'operating model'
        ])),

        # D2: Domain specificity
        'has_compliance': int(any(kw in text for kw in [
            'nist', 'hipaa', 'gdpr', 'soc 2', 'soc2', 'iso 27001',
            'pci dss', 'eu ai act', 'ccpa', 'cpra', 'fcra', 'eeoc',
            'regulatory', 'control library'
        ])),
        'cloud_providers_mentioned': sum(1 for kw in ['aws', 'azure', 'gcp', 'google cloud'] if kw in text),
        'systems_mentioned': sum(1 for kw in [
            'salesforce', 'sap', 'workday', 'hubspot', 'jira', 'servicenow',
            'okta', 'terraform', 'kubernetes', 'kafka', 'snowflake',
            'databricks', 'mulesoft', 'boomi', 'bigquery', 'bedrock',
            'vertex ai', 'openai'
        ] if kw in text),
        'domain_framework_count': sum(1 for kw in [
            'finops', 'ai governance', 'purdue model', 'zero trust',
            'data mesh', 'devsecops', 'itil', 'nist ai rmf'
        ] if kw in text),

        # D4: Research dependency
        'external_data_score': min(1.0, sum(1 for kw in [
            'market research', 'competitive analysis', 'analyst report',
            'industry trends', 'industry benchmark', 'latest pricing',
            'current pricing', 'current market', 'vendor pricing',
            'vendor comparison', 'compare vendors', 'real-time pricing',
            'pricing models', 'pricing tiers', 'competitor', 'survey data',
            'gartner', 'forrester', 'benchmark data', 'market rates',
            'regulatory updates', 'latest vendor documentation'
        ] if kw in text) / 3.0),
        'has_time_reference': int(any(kw in text for kw in [
            'this quarter', 'this month', 'last quarter', 'last month',
            'next quarter', 'this year', 'current', 'latest', 'recent',
            'real-time', 'today'
        ])),
        'vendor_tool_count': sum(1 for kw in [
            'openai', 'anthropic', 'cohere', 'google', 'microsoft',
            'stripe', 'twilio', 'segment', 'tealium', 'mparticle',
            'sailpoint', 'saviynt', 'cyberark', 'sift', 'forter',
            'mulesoft', 'boomi', 'splunk', 'datadog', 'pagerduty'
        ] if kw in text),
        'has_market_terms': int(any(kw in text for kw in [
            'market', 'pricing', 'vendor', 'competitor', 'benchmark',
            'industry', 'gartner', 'cost of', 'salary survey'
        ])),
        'has_cost_comparison': int(any(kw in text for kw in [
            'cost comparison', 'price comparison', 'total cost of ownership',
            'tco', 'roi projection', 'cost-benefit', 'cost benefit',
            'cost savings', 'pay bands', 'compensation benchmark'
        ])),

        # Boundary / production risk signals
        'has_comparison': int(any(kw in text for kw in [
            ' vs ', 'versus', 'compare', 'comparison', 'difference between',
            'differ', 'comparing'
        ])),
        'stakeholder_mentions': sum(1 for kw in [
            'ceo', 'cto', 'cfo', 'ciso', 'cmo', 'board', 'executive',
            'leadership', 'director', 'vp of', 'vice president'
        ] if kw in text),
        'risk_language': int(any(kw in text for kw in [
            'liability', 'exposure', 'penalty', 'breach', 'violation',
            'lawsuit', 'audit', 'compliance gap', 'risk assessment',
            'risk posture', 'remediation'
        ])),

        # Phrasing style
        'phrasing_explicit': 0,
        'phrasing_implicit': 0,
        'phrasing_vague': 0,
    }

    if phrasing_style:
        style = str(phrasing_style).lower().strip()
        if style == 'explicit':
            features['phrasing_explicit'] = 1
        elif style == 'implicit':
            features['phrasing_implicit'] = 1
        elif style == 'vague':
            features['phrasing_vague'] = 1

    return features


test_feats = extract_handcrafted_features(
    "Using the complete uploaded corpus, research latest AWS and Azure pricing and produce an executive summary, risk register, roadmap, and appendix-level evidence mapping",
    phrasing_style="explicit"
)
print(f"Feature count: {len(test_feats)}")
print(f"D3/D5 features: formal={test_feats['has_formal_deliverable']}, "
      f"large_context={test_feats['large_context_signal']}, artifacts={test_feats['provided_artifact_count']}")
print(f"D4 features: external={test_feats['external_data_score']:.2f}, "
      f"time={test_feats['has_time_reference']}, market={test_feats['has_market_terms']}")
'''


LOAD_DATA = r'''
# -- Load v4 expanded CSV -----------------------------------------------------
from pathlib import Path

DATASET_PATH = Path('/content/dataset_prompt_profiling_v4.csv')
if not DATASET_PATH.exists():
    DATASET_PATH = Path('dataset_prompt_profiling_v4.csv')

df = pd.read_csv(DATASET_PATH)
print(f"Loaded: {DATASET_PATH}")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nD5 value counts:")
print(df['d5'].value_counts().sort_index())
df.head()
'''


DATA_AUDIT = r'''
# -- Dataset audit ------------------------------------------------------------
required_cols = ['id', 'prompt', 'phrasing_style', 'domain', 'd1', 'd2', 'd3', 'd4', 'd5']
score_values = {0.0, 0.25, 0.5, 0.75, 1.0}

assert all(c in df.columns for c in required_cols), "Missing required columns"
assert df[required_cols].isna().sum().sum() == 0, "Nulls found in required columns"
assert df['id'].duplicated().sum() == 0, "Duplicate ids found"

for col in ['d1', 'd2', 'd3', 'd4', 'd5']:
    bad = set(df[col].unique()) - score_values
    assert not bad, f"{col} has invalid score values: {bad}"

print("Dataset integrity checks passed.")
print("\nPer-dimension score counts:")
for col in ['d1', 'd2', 'd3', 'd4', 'd5']:
    print(f"\n{col.upper()} - {DIMENSION_LABELS[col.upper()]}")
    print(df[col].value_counts().sort_index())
'''


VERIFY_FEATURES = r'''
# -- Verify rubric-specific features by tier ----------------------------------
hc_df['tier'] = df['tier'].values
print("D3 output-formality feature means by tier:")
print(hc_df.groupby('tier')[[
    'has_formal_deliverable', 'has_report_package',
    'has_long_output_signal', 'structured_section_count'
]].mean().round(3))

print("\nD5 context-requirement feature means by tier:")
print(hc_df.groupby('tier')[[
    'has_attachment', 'provided_artifact_count',
    'large_context_signal', 'multi_document_signal'
]].mean().round(3))

print("\nD4 research-dependency feature means by tier:")
print(hc_df.groupby('tier')[[
    'external_data_score', 'has_time_reference',
    'vendor_tool_count', 'has_cost_comparison', 'has_market_terms'
]].mean().round(3))
hc_df.drop('tier', axis=1, inplace=True)
'''

RESULTS = r'''
# -- Per-dimension CV metrics -------------------------------------------------
print("Per-Dimension Metrics (5-Fold CV: mean +/- std)")
print("=" * 75)
print("{:<5} {:>12} {:>12} {:>12} {:>14}".format("Dim", "MAE", "RMSE", "R2", "Accuracy"))
print("-" * 75)

cv_results = {}
for dim in dim_names:
    mae_m, mae_s = np.mean(fold_dim_maes[dim]), np.std(fold_dim_maes[dim])
    rmse_m, rmse_s = np.mean(fold_dim_rmses[dim]), np.std(fold_dim_rmses[dim])
    r2_m, r2_s = np.mean(fold_dim_r2s[dim]), np.std(fold_dim_r2s[dim])
    acc_m, acc_s = np.mean(fold_dim_accs[dim]), np.std(fold_dim_accs[dim])
    cv_results[dim] = {'MAE': mae_m, 'RMSE': rmse_m, 'R2': r2_m, 'Acc': acc_m}
    print(f"  {dim:<4} {mae_m:>5.4f}+/-{mae_s:.3f} {rmse_m:>5.4f}+/-{rmse_s:.3f} "
          f"{r2_m:>5.4f}+/-{r2_s:.3f} {acc_m:>6.2%}+/-{acc_s:.2%}")

print("-" * 75)
tier_m, tier_s = np.mean(fold_tier_accs), np.std(fold_tier_accs)
s1_m, s1_s = np.mean(fold_tier_s1_accs), np.std(fold_tier_s1_accs)
print(f"  {'Tier (S1)':<10} {'':<12} {'':<12} {'':<12} {s1_m:>6.2%}+/-{s1_s:.2%}")
print(f"  {'Tier Final':<10} {'':<12} {'':<12} {'':<12} {tier_m:>6.2%}+/-{tier_s:.2%}")
print("=" * 75)
'''


CONFIG = r'''
# -- Configuration ------------------------------------------------------------
N_PCA = 35
N_FOLDS = 5
RANDOM_STATE = 42
T3_RECALL_WEIGHT = 1.35
BOUNDARY_SAMPLE_WEIGHT = 1.15

XGB_PARAMS = dict(
    objective='multi:softprob',
    num_class=5,
    n_estimators=250,
    max_depth=4,
    learning_rate=0.08,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.75,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

TIER_XGB_PARAMS = dict(
    objective='multi:softprob',
    num_class=3,
    n_estimators=250,
    max_depth=4,
    learning_rate=0.08,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.75,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

print(f"Config: PCA={N_PCA}d, folds={N_FOLDS}, T3 weight={T3_RECALL_WEIGHT}, boundary weight={BOUNDARY_SAMPLE_WEIGHT}")
'''


CV = r'''
# -- Run 5-Fold CV ------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_dim_maes = {d: [] for d in dim_names}
fold_dim_rmses = {d: [] for d in dim_names}
fold_dim_r2s = {d: [] for d in dim_names}
fold_dim_accs = {d: [] for d in dim_names}
fold_tier_accs = []
fold_tier_s1_accs = []

all_actual_tiers = []
all_pred_tiers = []
all_actual_scores = []
all_pred_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, tier_labels)):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*60}")

    emb_tr, emb_te = embeddings[train_idx], embeddings[test_idx]
    hc_tr, hc_te = hc_features[train_idx], hc_features[test_idx]
    y_cls_tr, y_cls_te = y_classes[train_idx], y_classes[test_idx]
    y_sc_tr, y_sc_te = y_scores[train_idx], y_scores[test_idx]
    tier_tr, tier_te = tier_int[train_idx], tier_int[test_idx]

    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    emb_tr_pca = pca.fit_transform(emb_tr)
    emb_te_pca = pca.transform(emb_te)

    X_tr_base = np.hstack([emb_tr_pca, hc_tr])
    X_te_base = np.hstack([emb_te_pca, hc_te])

    scaler_base = StandardScaler()
    X_tr_base_s = scaler_base.fit_transform(X_tr_base)
    X_te_base_s = scaler_base.transform(X_te_base)

    cs_tr = df.iloc[train_idx]['complexity_score'].values
    boundary_tr = np.array([is_boundary_score(cs) for cs in cs_tr])

    tier_clf = XGBClassifier(**TIER_XGB_PARAMS)
    tier_weights = compute_sample_weight('balanced', tier_tr).astype(float)
    tier_weights[tier_tr == TIER_TO_INT['T3']] *= T3_RECALL_WEIGHT
    tier_weights[boundary_tr] *= BOUNDARY_SAMPLE_WEIGHT
    tier_clf.fit(X_tr_base_s, tier_tr, sample_weight=tier_weights)

    tier_pred_tr = tier_clf.predict(X_tr_base_s)
    tier_pred_te = tier_clf.predict(X_te_base_s)

    s1_acc = accuracy_score(tier_te, tier_pred_te)
    fold_tier_s1_accs.append(s1_acc)
    print(f"  Stage 1 Tier Accuracy: {s1_acc:.4f}")

    X_tr_aug = np.hstack([X_tr_base_s, tier_pred_tr.reshape(-1, 1)])
    X_te_aug = np.hstack([X_te_base_s, tier_pred_te.reshape(-1, 1)])

    multi_model = MultiOutputClassifier(estimator=XGBClassifier(**XGB_PARAMS))
    dim_weights = np.ones(len(train_idx), dtype=float)
    dim_weights[tier_tr == TIER_TO_INT['T3']] *= 1.20
    dim_weights[boundary_tr] *= BOUNDARY_SAMPLE_WEIGHT
    multi_model.fit(X_tr_aug, y_cls_tr, sample_weight=dim_weights)

    y_pred_cls = multi_model.predict(X_te_aug)
    y_pred_sc = np.vectorize(CLASS_TO_SCORE.get)(y_pred_cls).astype(float)

    for i, dim in enumerate(dim_names):
        mae = mean_absolute_error(y_sc_te[:, i], y_pred_sc[:, i])
        rmse = np.sqrt(mean_squared_error(y_sc_te[:, i], y_pred_sc[:, i]))
        r2 = r2_score(y_sc_te[:, i], y_pred_sc[:, i])
        acc = accuracy_score(y_cls_te[:, i], y_pred_cls[:, i])
        fold_dim_maes[dim].append(mae)
        fold_dim_rmses[dim].append(rmse)
        fold_dim_r2s[dim].append(r2)
        fold_dim_accs[dim].append(acc)

    derived_cs = np.array([complexity_score(*row) for row in y_pred_sc])
    derived_tiers = np.array([TIER_TO_INT[get_tier(cs)] for cs in derived_cs])

    final_tiers = np.array([
        choose_final_tier(s1, derived, cs)
        for s1, derived, cs in zip(tier_pred_te, derived_tiers, derived_cs)
    ])

    ens_acc = accuracy_score(tier_te, final_tiers)
    fold_tier_accs.append(ens_acc)
    print(f"  Boundary-aware Tier Accuracy: {ens_acc:.4f}")

    all_actual_tiers.extend([INT_TO_TIER[t] for t in tier_te])
    all_pred_tiers.extend([INT_TO_TIER[t] for t in final_tiers])
    all_actual_scores.append(y_sc_te)
    all_pred_scores.append(y_pred_sc)

print(f"\n{'='*60}")
print("  5-FOLD CV COMPLETE")
print(f"{'='*60}")
'''


FINAL_TRAIN = r'''
# -- Stage 1: Final Tier Classifier ------------------------------------------
final_tier_clf = XGBClassifier(**TIER_XGB_PARAMS)
final_tier_weights = compute_sample_weight('balanced', tier_int).astype(float)
final_tier_weights[tier_int == TIER_TO_INT['T3']] *= T3_RECALL_WEIGHT
final_tier_weights[np.array([is_boundary_score(cs) for cs in df['complexity_score'].values])] *= BOUNDARY_SAMPLE_WEIGHT
final_tier_clf.fit(X_all_base_s, tier_int, sample_weight=final_tier_weights)

tier_pred_all = final_tier_clf.predict(X_all_base_s)
print(f"Stage 1 tier classifier trained on {len(tier_int)} samples.")

# -- Stage 2: Final Dimension Classifier -------------------------------------
X_all_aug = np.hstack([X_all_base_s, tier_pred_all.reshape(-1, 1)])

final_multi_model = MultiOutputClassifier(estimator=XGBClassifier(**XGB_PARAMS))
final_dim_weights = np.ones(len(df), dtype=float)
final_dim_weights[tier_int == TIER_TO_INT['T3']] *= 1.20
final_dim_weights[np.array([is_boundary_score(cs) for cs in df['complexity_score'].values])] *= BOUNDARY_SAMPLE_WEIGHT
final_multi_model.fit(X_all_aug, y_classes, sample_weight=final_dim_weights)

print("Stage 2 dimension classifier trained.")
print(f"Sub-estimators: {len(final_multi_model.estimators_)} ({', '.join(dim_names)})")
'''


INFERENCE = r'''
def profile_prompt(prompt_text, embed_model, pca, scaler,
                   tier_clf, dim_model,
                   df_train, X_train_emb_raw,
                   phrasing_style=None):
    """Profile a prompt through the v4 boundary-aware two-stage pipeline."""
    emb_raw = embed_model.encode([prompt_text])
    emb_pca = pca.transform(emb_raw)

    hc_feats = extract_handcrafted_features(prompt_text, phrasing_style)
    hc_array = np.array([list(hc_feats.values())])

    combined = np.hstack([emb_pca, hc_array])
    combined_scaled = scaler.transform(combined)

    tier_pred_int = tier_clf.predict(combined_scaled)
    tier_pred_label = INT_TO_TIER[tier_pred_int[0]]

    combined_aug = np.hstack([combined_scaled, tier_pred_int.reshape(-1, 1)])
    pred_cls = dim_model.predict(combined_aug)

    d1, d2, d3, d4, d5 = [CLASS_TO_SCORE[c] for c in pred_cls[0]]

    cs = complexity_score(d1, d2, d3, d4, d5)
    derived_tier = get_tier(cs)
    derived_int = TIER_TO_INT[derived_tier]
    final_int = choose_final_tier(tier_pred_int[0], derived_int, cs)
    final_tier = INT_TO_TIER[final_int]

    similarities = cosine_similarity(emb_raw, X_train_emb_raw)[0]
    top3_idx = np.argsort(similarities)[-3:][::-1]
    top3_prompts = df_train.iloc[top3_idx]['prompt'].tolist()
    top3_scores = similarities[top3_idx].tolist()

    return {
        'tier': final_tier,
        'tier_stage1': tier_pred_label,
        'tier_derived': derived_tier,
        'boundary_score': is_boundary_score(cs),
        'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5,
        'complexity_score': round(float(cs), 4),
        'dimension_labels': DIMENSION_LABELS,
        'top3_similar': [
            {'prompt': p[:100], 'similarity': round(s, 4)}
            for p, s in zip(top3_prompts, top3_scores)
        ]
    }


print("Inference function defined.")
'''


TEST_PROMPTS = r'''
# -- Test on representative prompts ------------------------------------------
test_prompts = [
    ("why is our AWS bill so high this month", None),
    ("build a full GenAI cost attribution platform with market research across AWS, Azure and GCP pricing", None),
    ("set up SSO between okta and our internal tool", None),
    ("Research the latest vendor pricing for Snowflake vs Databricks vs BigQuery and produce a 3-year TCO comparison report for the CFO with industry benchmarks from Gartner", "explicit"),
    ("Using the complete uploaded corpus of cloud billing exports, architecture diagrams, contracts and incident reports, create an enterprise cost governance package with roadmap and evidence mapping", "explicit"),
]

for i, (prompt, style) in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: \"{prompt[:80]}{'...' if len(prompt)>80 else ''}\"")
    print(f"{'='*80}")

    result = profile_prompt(
        prompt, embed_model, final_pca, final_scaler,
        final_tier_clf, final_multi_model,
        df, embeddings, phrasing_style=style
    )

    print(f"  Tier:       {result['tier']} (S1={result['tier_stage1']}, derived={result['tier_derived']}, boundary={result['boundary_score']})")
    print(f"  Score:      {result['complexity_score']}")
    print(f"  D1={result['d1']} ({DIMENSION_LABELS['D1']})")
    print(f"  D2={result['d2']} ({DIMENSION_LABELS['D2']})")
    print(f"  D3={result['d3']} ({DIMENSION_LABELS['D3']})")
    print(f"  D4={result['d4']} ({DIMENSION_LABELS['D4']})")
    print(f"  D5={result['d5']} ({DIMENSION_LABELS['D5']})")
    print("  Similar:")
    for j, s in enumerate(result['top3_similar'], 1):
        print(f"    {j}. [{s['similarity']:.3f}] {s['prompt']}")
'''


SAVE = r'''
# -- Save all pipeline components --------------------------------------------
joblib.dump(final_tier_clf, 'v4_tier_classifier.joblib')
joblib.dump(final_multi_model, 'v4_dimension_classifier.joblib')
joblib.dump(final_scaler, 'v4_scaler.joblib')
joblib.dump(final_pca, 'v4_pca.joblib')
joblib.dump(hc_feature_names, 'v4_handcrafted_feature_names.joblib')

print("Models saved:")
print("  -> v4_tier_classifier.joblib")
print("  -> v4_dimension_classifier.joblib")
print("  -> v4_scaler.joblib")
print("  -> v4_pca.joblib")
print("  -> v4_handcrafted_feature_names.joblib")
'''


SUMMARY = r'''
# -- FINAL SUMMARY ------------------------------------------------------------
print("\n" + "=" * 78)
print("     FINAL SUMMARY - Architecture 3 v4 (5-Fold CV Results)")
print("=" * 78)
print(f"{'Dim':<5} {'Rubric Label':<24} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Accuracy':>10}")
print("-" * 78)

for dim in dim_names:
    r = cv_results[dim]
    print(f"{dim:<5} {DIMENSION_LABELS[dim]:<24} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f} {r['Acc']:>9.2%}")

print("-" * 78)
print(f"{'Tier Stage1':<30} {s1_m:>45.2%}")
print(f"{'Tier Final':<30} {tier_m:>45.2%}")
print("=" * 78)
print("\nImprovements applied:")
print(f"  - Expanded dataset: {len(df)} prompts")
print("  - Rubric-aligned D3/D5 naming and features")
print("  - D4 remains ML-predicted")
print("  - Direct tier classifier retained")
print("  - Boundary-aware tier policy added")
print(f"  - Cost-sensitive T3 weighting: {T3_RECALL_WEIGHT}x")
print(f"  - Boundary sample weighting: {BOUNDARY_SAMPLE_WEIGHT}x")
print(f"  - PCA: {N_PCA} components")
print(f"  - Hand-crafted features: {len(hc_feature_names)}")
print(f"  - Feature-to-sample ratio: 1:{X_all_base_s.shape[0]/X_all_base_s.shape[1]:.1f}")
'''


VERIFY = r'''
# -- Verify saved models ------------------------------------------------------
l_tier = joblib.load('v4_tier_classifier.joblib')
l_dim = joblib.load('v4_dimension_classifier.joblib')
l_scaler = joblib.load('v4_scaler.joblib')
l_pca = joblib.load('v4_pca.joblib')

v_emb = embed_model.encode(["test prompt"])
v_pca = l_pca.transform(v_emb)
v_hc = extract_handcrafted_features("test prompt")
v_combined = np.hstack([v_pca, np.array([list(v_hc.values())])])
v_scaled = l_scaler.transform(v_combined)
v_tier = l_tier.predict(v_scaled)
v_aug = np.hstack([v_scaled, v_tier.reshape(-1, 1)])
v_pred = l_dim.predict(v_aug)
v_scores = [CLASS_TO_SCORE[c] for c in v_pred[0]]
v_cs = complexity_score(*v_scores)
v_final = choose_final_tier(v_tier[0], TIER_TO_INT[get_tier(v_cs)], v_cs)

print(f"Model verification: D1={v_scores[0]}, D2={v_scores[1]}, "
      f"D3={v_scores[2]}, D4={v_scores[3]}, D5={v_scores[4]}")
print(f"Tier: {INT_TO_TIER[v_final]}")
print("\nArchitecture 3 v4 pipeline complete.")
'''


def main() -> None:
    nb = json.loads(V3_NOTEBOOK.read_text(encoding="utf-8"))

    set_source(nb, 0, """# Architecture 3 v4 - Prompt Profiling Pipeline\n\nv4 keeps the v3 two-stage architecture and adds:\n\n- expanded/audited v4 dataset\n- rubric-aligned D3/D5 feature naming\n- targeted features for Output Formality and Context Requirement\n- cost-sensitive T3 weighting\n- boundary-aware final tier policy\n- v4 model artifact names""")
    set_source(nb, 1, nb["cells"][1]["source"][0] + "import re\nimport warnings\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\n\nfrom pathlib import Path\n\nfrom sklearn.model_selection import train_test_split, StratifiedKFold\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.multioutput import MultiOutputClassifier\nfrom sklearn.metrics import (\n    mean_absolute_error, mean_squared_error, r2_score,\n    confusion_matrix, classification_report, accuracy_score\n)\nfrom sklearn.metrics.pairwise import cosine_similarity\nfrom sklearn.utils.class_weight import compute_sample_weight\n\nfrom xgboost import XGBClassifier\nfrom sentence_transformers import SentenceTransformer\n\nwarnings.filterwarnings('ignore')\npd.set_option('display.max_colwidth', 100)\n\nprint(\"All imports successful.\")")
    set_source(nb, 3, HELPERS)
    set_source(nb, 4, FEATURES)
    set_source(nb, 6, LOAD_DATA)
    set_source(nb, 8, "---\n## Step 2 - Feature Engineering")
    set_source(nb, 11, VERIFY_FEATURES)
    set_source(nb, 15, CONFIG)
    set_source(nb, 16, CV)
    set_source(nb, 18, RESULTS)
    set_source(nb, 22, FINAL_TRAIN)
    set_source(nb, 24, INFERENCE)
    set_source(nb, 25, TEST_PROMPTS)
    set_source(nb, 32, SAVE)
    set_source(nb, 33, SUMMARY)
    set_source(nb, 34, VERIFY)

    # Insert dataset audit after tier derivation.
    audit_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(DATA_AUDIT),
    }
    nb["cells"].insert(8, audit_cell)

    # After insertion, update affected later markdown heading if needed.
    clear_outputs(nb)
    for cell in nb["cells"]:
        cell["source"] = [
            line.replace("viz_v3_", "viz_v4_")
                .replace("Key v3 change", "Key v4 carry-forward")
            for line in cell.get("source", [])
        ]
    V4_NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {V4_NOTEBOOK}")


if __name__ == "__main__":
    main()
