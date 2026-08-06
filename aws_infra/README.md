# AWS Model Routing Layer

Adds a three-layer control plane in front of every AI model request.

```
Application / Frontend
        │
        ▼
Amazon API Gateway  ──► Throttling (Usage Plans + API Keys)
        │
        ▼
AWS Lambda          ──► (1) Allow-List check
                        (2) Context Window limit check
        │
        ▼
Amazon Bedrock / Vertex AI / Anthropic
```

---

## Directory Structure

```
aws_infra/
├── lambda/
│   └── model_router/
│       ├── handler.py          # Lambda entry point
│       ├── config_loader.py    # SSM config reader (cached)
│       ├── token_estimator.py  # Token count estimator
│       └── requirements.txt
├── config/
│   ├── app_configs.json        # Sample app configs
│   └── ssm_bootstrap.py        # Push configs to SSM
└── api_gateway/
    └── setup_api_gateway.py    # Provision API Gateway + Usage Plans
```

---

## Setup Guide

### Prerequisites
- AWS credentials configured (`aws configure` or env vars)
- Python 3.11+
- `boto3` installed (`pip install boto3`)

---

### Step 1 — Push App Configs to SSM

Edit `aws_infra/config/app_configs.json` to define your apps, then run:

```bash
python aws_infra/config/ssm_bootstrap.py --region us-east-1
```

Add `--dry-run` to preview without writing anything.

---

### Step 2 — Deploy the Lambda

1. Package the Lambda:
```bash
cd aws_infra/lambda/model_router
pip install -r requirements.txt -t package/
cp handler.py config_loader.py token_estimator.py package/
cd package && zip -r ../model_router.zip . && cd ..
```

2. Create the Lambda function in AWS Console or via CLI:
```bash
aws lambda create-function \
  --function-name model-router \
  --runtime python3.11 \
  --handler handler.handler \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --zip-file fileb://aws_infra/lambda/model_router/model_router.zip \
  --environment Variables="{AWS_REGION=us-east-1,SSM_PATH_PREFIX=/model-router}"
```

The Lambda execution role needs:
- `ssm:GetParameter`, `ssm:GetParametersByPath`
- `bedrock:InvokeModel`, `bedrock:Converse`

---

### Step 3 — Provision API Gateway

```bash
python aws_infra/api_gateway/setup_api_gateway.py \
  --lambda-arn arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:model-router \
  --region us-east-1 \
  --stage prod
```

This prints:
- The **Invoke URL** (e.g. `https://abc123.execute-api.us-east-1.amazonaws.com/prod/invoke`)
- The **API Key values** per app

---

## Usage

Call the endpoint with the app's API Key:

```bash
curl -X POST https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod/invoke \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "modelmatrix",
    "env": "prod",
    "model_id": "nova-lite",
    "prompt": "Summarize the key risks in this contract.",
    "max_output_tokens": 512
  }'
```

### Response examples

**Allowed:**
```json
{ "statusCode": 200, "body": { "allowed": true, "model_id": "nova-lite", "response": "..." } }
```

**Allow-list blocked:**
```json
{ "statusCode": 403, "body": { "error": "model_not_allowed", "message": "Model 'gpt-5' is not on the allow-list..." } }
```

**Context limit exceeded:**
```json
{ "statusCode": 400, "body": { "error": "context_window_exceeded", "message": "Estimated 9,200 input tokens exceeds limit of 8,000..." } }
```

**Throttled (by API Gateway):**
```json
{ "statusCode": 429, "message": "Too Many Requests" }
```

---

## Frontend

Navigate to **AWS Routing** in the ModelMatrix UI to:
- View, create, edit, and delete app configurations
- Toggle which models each app is allowed to use
- Set context window limits and throttle settings
- Dry-run a routing decision without invoking any model

---

## Configuration Schema

```json
{
  "app_id": "modelmatrix",
  "env": "prod",
  "allowed_models": ["nova-lite", "llama4-scout"],
  "context_limits": {
    "max_input_tokens": 8000,
    "max_output_tokens": 4096,
    "max_total_tokens": 10000
  },
  "throttle": {
    "rate_limit": 100,
    "burst_limit": 200,
    "quota_per_day": 10000
  }
}
```

SSM parameter path: `/model-router/{app_id}/{env}`
