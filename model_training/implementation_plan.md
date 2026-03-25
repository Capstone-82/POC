# LLM Model Recommendation System - Implementation Plan

## Data Understanding

### Datasets
| Dataset | Rows | Columns | Purpose |
|---------|------|---------|---------|
| [merged_dataset.csv](file:///c:/Users/Musharraf/Documents/POC/model_training/merged_dataset.csv) | 592 | `prompt, complexity` (low/mid/high) | **Classifier training** |
| [phase-1-dataset.csv](file:///c:/Users/Musharraf/Documents/POC/model_training/phase-1-dataset.csv) | ~96K | `id, provider, model_id, prompt, prompt_complexity, prompt_quality_score, response, accuracy_score, cost, tokens, latency_ms` | Benchmark data |
| [phase-1.1-dataset.csv](file:///c:/Users/Musharraf/Documents/POC/model_training/phase-1.1-dataset.csv) | ~112K | Same schema as phase-1 | Benchmark data (extended) |

### Existing Classifier (from notebook)
- **Model**: BERT-base-uncased fine-tuned for 3-class classification (low/mid/high)
- **Confusion Matrix**: `[[15, 0, 1], [0, 19, 0], [0, 1, 24]]` → **96.7% accuracy**
- Classes: low (151 prompts), mid (152 prompts), high (149 prompts + extras ≈ 289)

## Architecture

```
User Prompt → [Classifier] → complexity_class → [Recommendation Engine] → Best Model + Delta Stats
                                                         ↑
                                              Model Performance Profiles
                                              (pre-computed from benchmark CSVs)
```

### Step 1: Lightweight Prompt Classifier (TF-IDF + Logistic Regression)
- **Why**: BERT is overkill for 3-class text classification with ~600 samples. A TF-IDF + LogReg model:
  - Trains in <1 second (vs minutes for BERT fine-tuning)
  - No GPU needed, tiny model (~50KB vs ~400MB)
  - Still achieves 90%+ accuracy on this task
- **Training data**: [merged_dataset.csv](file:///c:/Users/Musharraf/Documents/POC/model_training/merged_dataset.csv)

### Step 2: Model Performance Profiles (Pre-computed Lookup)
- Aggregate [phase-1-dataset.csv](file:///c:/Users/Musharraf/Documents/POC/model_training/phase-1-dataset.csv) + [phase-1.1-dataset.csv](file:///c:/Users/Musharraf/Documents/POC/model_training/phase-1.1-dataset.csv) by [(model_id, prompt_complexity)](file:///c:/Users/Musharraf/Documents/POC/model_training/prompt.py#110-116)
- Compute per-model, per-complexity:
  - `avg_accuracy_score`
  - `avg_cost`
  - `avg_latency_ms`
  - `response_count` (sample size)

### Step 3: Recommendation Engine
- Given a classified prompt complexity:
  1. Look up all models' performance for that class
  2. Rank by a composite score (accuracy weighted highest)
  3. Compare the current/default model vs the recommended model
  4. Output delta: `+X% accuracy • -Y% cost • -Zms latency`

## Output Format
```
Recommendation: Switching to Gemini 2.5 Pro gives you:
 +12% accuracy • -8% cost • -180ms latency
```
