# LLM Recommendation System — Project Overview

## What Are We Building?

A two-part web application that:

1. **Trains** a benchmark dataset by sending prompts to multiple LLMs, evaluating their responses, and storing results in Supabase
2. **Recommends** the best LLM for a given prompt based on that benchmark data

---

## The Big Picture

```
USER TYPES PROMPT
      ↓
[TRAINING VIEW]
Evaluator model scores the prompt → complexity + quality
Same prompt sent to all Bedrock models → responses collected
Evaluator scores each response → accuracy score
Everything saved to Supabase

[INFERENCE VIEW]
User types new prompt + selects use case + selects current model
Backend classifies prompt → runs recommendation model
Returns: "Switch to X → +12% accuracy, -8% cost, -180ms latency"
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | Python FastAPI |
| Database | Supabase (PostgreSQL) |
| LLM Providers | AWS Bedrock (14 models) |
| Evaluator Models | Gemini 2.0 Flash / GPT-4o / Claude Sonnet 4.6 |
| Real-time updates | SSE (Server Sent Events) |
| Styling | Tailwind CSS |

---

## Two Views

### View 1 — Training

**Purpose:** Build the benchmark dataset

**What the user does:**
- Selects an evaluator model (default: Gemini 2.0 Flash)
- Either types a single prompt OR uploads a CSV with a `prompt` column
- Clicks Run
- Watches a live log stream as each model processes each prompt
- Sees a success message when done

**What happens in the background (per prompt):**
```
Step 1: Send prompt to evaluator → get prompt_complexity + prompt_quality_score
Step 2: Enrich prompt with those two fields
Step 3: Send enriched prompt to all 14 Bedrock models in parallel
Step 4: For each response → send to evaluator → get accuracy_score
Step 5: Save one row per model to Supabase
```

**Each Supabase row contains:**
```
model_id, provider, prompt, prompt_complexity, prompt_quality_score,
response, accuracy_score, cost, tokens, latency_ms
```

---

### View 2 — Inference / Recommendation

**Purpose:** Recommend the best LLM for a new prompt

**What the user does:**
- Types a prompt
- Selects use case from dropdown
- Selects their current model from dropdown
- Clicks Get Recommendation

**What they see:**
```
Your prompt:     high complexity  •  quality: 78/100

You chose:       GPT-4o

Recommended:     Gemini 2.5 Pro
  + 12% accuracy
  -  8% cost
  - 180ms latency

Reason: High-complexity reasoning prompts. Gemini 2.5 Pro
outperforms GPT-4o on this task type in our benchmark data.
```

---

## Dataset Schema (Supabase Table: `benchmark_results`)

| Column | Type | Description |
|---|---|---|
| `id` | uuid | Auto generated primary key |
| `created_at` | timestamp | Auto generated |
| `provider` | text | e.g. Amazon, OpenAI, Google |
| `model_id` | text | e.g. gpt-4o, gemini-2.5-pro |
| `prompt` | text | Original prompt text |
| `prompt_complexity` | text | low / mid / high |
| `prompt_quality_score` | int | 0-100 |
| `response` | text | Raw model response |
| `accuracy_score` | int | 0-100 |
| `cost` | float | USD cost of the API call |
| `tokens` | int | Total tokens used |
| `latency_ms` | int | Response time in ms |

---

## Models in Scope (Bedrock)

| Provider | Models |
|---|---|
| Amazon | nova-lite-v1, nova-pro-v1 |
| Meta | llama-3.3-70b, llama-4-scout |
| Mistral | mistral-large, mistral-small, pixtral-large |
| DeepSeek | deepseek-r1, deepseek-v3 |

Plus any additional Bedrock models you add.

---

## Use Cases (Inference Dropdown)

Chat · Code · Reasoning · RAG · Summarization · Structured Output · Tool Calling · Vision · Multimodality

---

## Key Design Decisions

**Why SSE for training?**
Training one prompt across 14 models takes time. Without live updates the user sees a blank screen and has no idea if it's working. SSE streams one log line per model per prompt as it completes — gives instant feedback.

**Why user-selected use case?**
Classifying use case from prompt text alone is ambiguous and error-prone. User knows their intent — let them select it. Eliminates one source of classification error entirely.

**Why evaluator at temperature=0 with grouped scoring bands?**
LLM evaluators are non-deterministic — same response can score 82 one call and 86 the next. Temperature 0 reduces variance significantly. Grouped scoring bands (e.g. 90-100 = fully correct) anchor the model to defined buckets rather than free-picking any number, reducing variance further.

**Why store the response text?**
During training we store it so you can audit and verify evaluator scores. In the final ML training pipeline you may drop it — only the scores matter for the model.

---

## Folder Structure

```
project/
  frontend/         ← React Vite app
  backend/          ← FastAPI app
  README.md
  PROJECT_OVERVIEW.md   ← this file
  FRONTEND_SPEC.md
  BACKEND_SPEC.md
```

---

## Build Order

1. Set up Supabase table
2. Build FastAPI skeleton with all endpoints returning mock data
3. Build frontend against mock endpoints
4. Wire in real Bedrock + evaluator calls
5. Test end to end with single prompt
6. Test with CSV upload
