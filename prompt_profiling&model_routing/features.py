import re
import numpy as np
import pandas as pd

SCORE_COLS = ['d1', 'd2', 'd3', 'd4', 'd5']
VALID_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]
SCORE_TO_CLASS = {score: idx for idx, score in enumerate(VALID_SCORES)}
CLASS_TO_SCORE = {idx: score for score, idx in SCORE_TO_CLASS.items()}

DIMENSION_LABELS = {
    'd1': 'Semantic Complexity',
    'd2': 'Domain Specificity',
    'd3': 'Output Formality',
    'd4': 'Research Dependency',
    'd5': 'Context Requirement',
}

D1_TO_INTENT = {
    0.00: 'FACTUAL',
    0.25: 'FACTUAL',
    0.50: 'ANALYTICAL',
    0.75: 'SYNTHETIC',
    1.00: 'STRATEGIC',
}

VALID_DOMAINS = [
    "Artificial Intelligence", "Biology", "Business", "Cloud Computing",
    "Cybersecurity", "Data Science", "Deep Learning", "DevOps",
    "Digital Transformation", "Education", "Embedded Systems", "FinOps",
    "Finance", "General Knowledge", "Human Resources", "Legal",
    "Machine Learning", "Marketing", "Mathematics", "Natural Language Processing",
    "Programming", "Software Engineering", "Statistics", "Supply Chain"
]

ARTIFACT_TERMS = ['csv', 'json', 'pdf', 'log', 'yaml', 'yml', 'xlsx', 'docx', 'transcript', 'diagram']
CLOUD_PROVIDERS = ['aws', 'azure', 'gcp', 'google cloud', 'oci']
SYSTEMS = ['salesforce', 'servicenow', 'jira', 'workday', 'sap', 'snowflake', 'databricks', 'okta', 'hubspot', 'github', 'gitlab']
FRAMEWORKS = ['itil', 'finops', 'togaf', 'owasp', 'dora', 'nist', 'hipaa', 'soc 2', 'soc2', 'gdpr', 'iso 27001', 'pci-dss', 'pci dss']
VENDOR_TOOLS = sorted(set(CLOUD_PROVIDERS + SYSTEMS + ['openai', 'anthropic', 'bedrock', 'terraform', 'kubernetes', 'docker', 'jenkins', 'splunk']))

DOMAIN_BUCKETS = {
    'cloud': ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'terraform'],
    'finops': ['finops', 'cost', 'budget', 'chargeback', 'showback'],
    'security': ['security', 'vulnerability', 'iam', 'zero trust', 'soc'],
    'devops': ['devops', 'ci/cd', 'pipeline', 'sre', 'deployment'],
    'data': ['data pipeline', 'etl', 'warehouse', 'lakehouse', 'spark'],
    'ai': ['ai', 'llm', 'genai', 'machine learning', 'model'],
    'hr': ['hr', 'employee', 'talent', 'workforce', 'recruiting'],
    'supply': ['supply chain', 'inventory', 'procurement', 'logistics'],
}

D1_COMPLEXITY_TERMS = [
    'strategic', 'cross-domain', 'enterprise-wide', 'synthesize', 'multi-cloud',
    'governance', 'architecture', 'framework', 'transformation', 'lifecycle',
    'holistic', 'end-to-end', 'migration', 'orchestration',
]

SIMPLE_FACTUAL_PATTERNS = [
    r'^what is\b', r'^define\b', r'^who is\b', r'^when\b',
    r'^list\b', r'^name\b', r'\bwhat does .{1,30} mean\b',
]

D2_DOMAIN_TERMS = [
    'kubernetes', 'terraform', 'sagemaker', 'databricks', 'snowflake',
    'cicd', 'ci/cd', 'finops', 'mlops', 'devsecops', 'apigee',
    'oauth', 'saml', 'oidc', 'vpc', 'subnet', 'iam',
]

RESEARCH_SIGNAL_KEYWORDS = {
    'market_research': ['market', 'industry', 'trend', 'tam', 'sam', 'som'],
    'competitive_analysis': ['competitor', 'competitive', 'benchmark', 'rival'],
    'regulatory_compliance': ['regulation', 'regulatory', 'compliance', 'gdpr', 'hipaa', 'sox', 'eu ai act'],
    'security': ['security', 'vulnerability', 'threat', 'risk', 'iam', 'zero trust'],
    'cloud_infrastructure': ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'terraform'],
    'finops': ['finops', 'cost', 'spend', 'budget', 'showback', 'chargeback'],
    'devops': ['ci/cd', 'pipeline', 'deployment', 'sre', 'devops', 'observability'],
    'data_engineering': ['data pipeline', 'etl', 'warehouse', 'lakehouse', 'spark'],
    'ai_governance': ['ai governance', 'llm', 'model risk', 'genai', 'guardrail'],
    'system_integration': ['integration', 'api', 'webhook', 'middleware'],
    'supply_chain': ['supply chain', 'inventory', 'procurement', 'logistics'],
    'hr_tech': ['hr', 'employee', 'talent', 'workforce', 'recruiting'],
    'vendor_analysis': ['vendor', 'rfi', 'rfp', 'procurement'],
}

def count_terms_safe(text, terms):
    return sum(1 for term in terms if term in text)

def any_terms_safe(text, terms):
    return any(term in text for term in terms)

def get_style_at(phrasing_styles, i):
    if phrasing_styles is None:
        return None
    try:
        value = phrasing_styles.iloc[i]
    except AttributeError:
        value = phrasing_styles[i]
    return None if pd.isna(value) else str(value).strip().lower()

def handcrafted_features(prompts, phrasing_styles=None, domains=None):
    rows = []
    for i, prompt in enumerate(prompts):
        text = str(prompt)
        lower = text.lower()
        words = re.findall(r'\b\w+\b', lower)
        unique_words = set(words)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        lines = [line for line in text.splitlines() if line.strip()]
        style = get_style_at(phrasing_styles, i)

        row = {}
        row['char_len'] = len(text)
        row['word_count'] = len(words)
        row['sentence_count'] = max(1, len(sentences))
        row['avg_word_len'] = float(np.mean([len(w) for w in words])) if words else 0.0
        row['unique_word_ratio'] = len(unique_words) / max(1, len(words))
        row['line_count'] = len(lines)

        # PDF D5 Aligned Token buckets
        estimated_tokens = len(words) * 1.3
        row['token_bracket_factual'] = int(estimated_tokens < 1000)
        row['token_bracket_short'] = int(1000 <= estimated_tokens < 4000)
        row['token_bracket_medium'] = int(4000 <= estimated_tokens < 16000)
        row['token_bracket_long'] = int(16000 <= estimated_tokens < 32000)
        row['token_bracket_extreme'] = int(estimated_tokens >= 32000)

        # PDF D4 Aligned retrieval / live keywords
        row['live_retrieval_request'] = int(any_terms_safe(lower, ['real-time', 'live pricing', 'fetch latest', 'api call', 'lookup online', 'search the web', 'browse the web', 'current status']))
        row['has_attachment'] = int(any_terms_safe(lower, ['uploaded', 'attached', 'provided file', 'document below', 'context below', 'see below']))
        row['provided_artifact_count'] = count_terms_safe(lower, ARTIFACT_TERMS)
        row['large_context_signal'] = int(any_terms_safe(lower, ['across all', 'entire', 'all of our', 'company-wide', 'large context', 'full document']))
        row['multi_document_signal'] = int(any_terms_safe(lower, ['multiple', 'all the', 'each of the', 'various', 'several documents', 'set of files']))

        # PDF D3 Aligned Output Formality
        row['has_formal_deliverable'] = int(any_terms_safe(lower, ['report', 'brief', 'proposal', 'specification', 'whitepaper', 'requirements doc']))
        row['has_report_package'] = int(any_terms_safe(lower, ['appendix', 'table of contents', 'toc', 'risk register', 'executive summary', 'roadmap', 'implementation plan']))
        row['has_long_output_signal'] = int(any_terms_safe(lower, ['comprehensive', 'detailed', 'thorough', 'in-depth', 'end-to-end']))
        row['structured_section_count'] = count_terms_safe(lower, ['executive summary', 'timeline', 'roadmap', 'risk register', 'assumptions', 'recommendations', 'next steps', 'success metrics'])

        row['has_scope_words'] = int(any_terms_safe(lower, ['strategic', 'cross-domain', 'enterprise-wide', 'synthesize', 'multi-cloud', 'governance']))
        row['action_verb_count'] = count_terms_safe(lower, ['build', 'design', 'evaluate', 'integrate', 'optimize', 'develop', 'assess', 'recommend', 'compare'])
        row['multi_stage_signal'] = int(bool(re.search(r'\bphase\b|\bstage\b|\bstep\s*1\b|\bmilestone\b|\bsequentially\b|\bfirst\b.*\bthen\b', lower)))

        row['has_compliance'] = int(any_terms_safe(lower, ['nist', 'hipaa', 'soc2', 'soc 2', 'gdpr', 'iso 27001', 'pci-dss', 'pci dss', 'compliance']))
        row['cloud_providers_mentioned'] = count_terms_safe(lower, CLOUD_PROVIDERS)
        row['systems_mentioned'] = count_terms_safe(lower, SYSTEMS)
        row['domain_framework_count'] = count_terms_safe(lower, FRAMEWORKS)

        row['external_data_score'] = count_terms_safe(lower, ['market research', 'industry report', 'analyst', 'third-party', 'external data', 'latest', 'current'])
        row['has_time_reference'] = int(bool(re.search(r'\b20\d{2}\b|\bfy\d{2}\b|\bthis quarter\b|\blatest\b|\bcurrent\b|\brecent\b|\btoday\b|\bnow\b', lower)))
        row['vendor_tool_count'] = count_terms_safe(lower, VENDOR_TOOLS)
        row['has_market_terms'] = int(any_terms_safe(lower, ['competitor', 'market share', 'tam', 'sam', 'som', 'benchmark', 'industry trend']))
        row['has_cost_comparison'] = int(any_terms_safe(lower, ['pricing', 'tco', 'roi', 'showback', 'chargeback', 'cheapest']) or 'cost analysis' in lower)

        row['has_comparison'] = int(any_terms_safe(lower, ['compare', 'versus', 'tradeoff']) or any(phrase in lower for phrase in [' vs ', 'difference between']))
        row['stakeholder_mentions'] = count_terms_safe(lower, ['ceo', 'cto', 'cio', 'cfo', 'board', 'leadership', 'management', 'executive'])
        row['risk_language'] = count_terms_safe(lower, ['risk', 'threat', 'vulnerability', 'mitigation', 'breach', 'exposure', 'audit'])

        row['has_role_prompt'] = int(bool(re.search(r'\byou are\b|\bact as\b|\bassume the role\b', lower)))
        row['has_step_request'] = int(bool(re.search(r'\bstep[- ]by[- ]step\b|\bfirst\b.*\bthen\b|\bsequentially\b', lower)))
        row['has_chain_of_thought'] = int(bool(re.search(r'\bthink through\b|\breason about\b|\blet.s think\b|\bchain of thought\b|\bwalk me through\b', lower)))
        if '?' not in text:
            row['question_complexity'] = 0
        elif any(phrase in lower for phrase in ['what should', 'design a']) or any_terms_safe(lower, ['recommend', 'propose', 'strategy']):
            row['question_complexity'] = 3
        elif any_terms_safe(lower, ['why', 'how', 'compare', 'analyze', 'evaluate', 'assess']):
            row['question_complexity'] = 2
        else:
            row['question_complexity'] = 1
        row['multi_domain_count'] = sum(1 for bucket_terms in DOMAIN_BUCKETS.values() if any_terms_safe(lower, bucket_terms))

        row['has_code_block'] = int('```' in text)
        row['has_output_format'] = int(bool(re.search(r'\bin json\b|\bas a table\b|\bformat as\b|\bcsv output\b|\bin yaml\b|\bas a table\b|\bas markdown\b|\bstrict yaml\b|\bstrict json\b', lower)))
        row['has_creative_language'] = int(any_terms_safe(lower, ['imagine', 'creative', 'story', 'compose', 'fictional']) or 'write a' in lower)
        row['has_classification_request'] = int(any_terms_safe(lower, ['classify', 'categorize', 'label']) or any(phrase in lower for phrase in ['which category', 'sort into']))
        row['enumeration_signal'] = int(bool(re.search(r'\blist\b|\btop \d+\b|\benumerate\b|\bbullet point\b|\brank\b', lower)))

        row['d1_strategic_signal_count'] = count_terms_safe(lower, D1_COMPLEXITY_TERMS)
        row['d1_simple_factual_signal'] = int(any(re.search(pattern, lower) for pattern in SIMPLE_FACTUAL_PATTERNS))
        row['d1_multi_constraint_count'] = count_terms_safe(lower, ['include', 'cover', 'consider', 'account for', 'must'])
        row['d1_solution_design_signal'] = int(any_terms_safe(lower, ['design', 'architect', 'plan', 'strategy', 'roadmap']))
        row['d2_domain_term_count'] = count_terms_safe(lower, D2_DOMAIN_TERMS)
        row['d2_acronym_count'] = len(re.findall(r'\b[A-Z]{2,6}\b', text))
        row['d2_vendor_or_framework_signal'] = int(row['vendor_tool_count'] > 0 or row['domain_framework_count'] > 0)
        row['d2_generic_prompt_signal'] = int(row['d2_domain_term_count'] == 0 and row['cloud_providers_mentioned'] == 0 and row['systems_mentioned'] == 0)

        row['phrasing_explicit'] = int(style == 'explicit')
        row['phrasing_implicit'] = int(style == 'implicit')
        row['phrasing_vague'] = int(style == 'vague')

        rows.append(row)

    feature_df = pd.DataFrame(rows).fillna(0)

    if domains is not None:
        domains_categorical = pd.Categorical(domains, categories=VALID_DOMAINS)
        df_dom = pd.get_dummies(domains_categorical, prefix='dom', dtype=int)
        feature_df = pd.concat([feature_df, df_dom.reset_index(drop=True)], axis=1)
    else:
        for dom in VALID_DOMAINS:
            feature_df[f'dom_{dom}'] = 0

    return feature_df

def extract_research_signals(prompt, d4_score):
    if d4_score <= 0:
        return []
    text = str(prompt).lower()
    signals = []
    for signal, keywords in RESEARCH_SIGNAL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            signals.append(signal)
    return signals if signals else ['external_research']

def complexity_score_from_dims(d1, d2, d3, d4, d5):
    return (
        d1 * 0.35 +
        d2 * 0.20 +
        d3 * 0.20 +
        d4 * 0.15 +
        d5 * 0.10
    )

def tier_from_score(score):
    if score < 0.40:
        return 'T1'
    if score < 0.70:
        return 'T2'
    return 'T3'

def construct_nn_features(embeddings_part, knn_model, train_df, is_train=False):
    distances, indices = knn_model.kneighbors(embeddings_part)
    if is_train:
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    else:
        indices = indices[:, :-1]
        distances = distances[:, :-1]

    train_tier_numeric = np.array([{'T1': 0, 'T2': 1, 'T3': 2}[t] for t in train_df['tier'].values])
    train_d1 = train_df['d1'].values
    train_d2 = train_df['d2'].values

    feats = []
    for i in range(len(indices)):
        idxs = indices[i]
        dists = distances[i]
        feats.append([
            train_tier_numeric[idxs].mean(),
            train_tier_numeric[idxs].std(),
            train_d1[idxs].mean(),
            train_d1[idxs].std(),
            train_d2[idxs].mean(),
            train_d2[idxs].std(),
            dists.mean(),
        ])
    return np.array(feats)

def transform_shared_features(embeddings_part, hand_features_part, pca, scaler, knn_model, train_df):
    emb_pca = pca.transform(embeddings_part)
    knn_feats = construct_nn_features(embeddings_part, knn_model, train_df, is_train=False)
    raw = np.hstack([emb_pca, hand_features_part, knn_feats])
    return scaler.transform(raw)
