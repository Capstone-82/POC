import boto3
import json
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)

AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")


def _meta_arn(model_short: str) -> str:
    return (
        f"arn:aws:bedrock:{AWS_REGION}:{AWS_ACCOUNT_ID}:"
        f"inference-profile/{model_short}"
    )


# ─── Model Registry ──────────────────────────────────────────
# fmt options:
#   "meta"     → prompt + max_gen_len  (Llama family)
#   "nova"     → messages + inferenceConfig  (Amazon Nova)
#   "messages" → messages + max_tokens  (Mistral, DeepSeek, Anthropic)

BEDROCK_MODELS = [
    # ── Meta Llama ───────────────────────────────────────────
    {
        "model_id": _meta_arn("us.meta.llama3-3-70b-instruct-v1:0"),
        "provider": "Meta",
        "short_id": "llama3-3-70b",
        "fmt":      "meta",
    },
    {
        "model_id": _meta_arn("us.meta.llama3-2-90b-instruct-v1:0"),
        "provider": "Meta",
        "short_id": "llama3-2-90b",
        "fmt":      "meta",
    },
    {
        "model_id": _meta_arn("us.meta.llama3-1-70b-instruct-v1:0"),
        "provider": "Meta",
        "short_id": "llama3-1-70b",
        "fmt":      "meta",
    },
    # ── Amazon Nova ──────────────────────────────────────────
    {
        "model_id": "us.amazon.nova-lite-v1:0",
        "provider": "Amazon",
        "short_id": "nova-lite",
        "fmt":      "nova",
    },
    {
        "model_id": "us.amazon.nova-pro-v1:0",
        "provider": "Amazon",
        "short_id": "nova-pro",
        "fmt":      "nova",
    },
    {
        "model_id": "us.amazon.nova-premier-v1:0",
        "provider": "Amazon",
        "short_id": "nova-premier",
        "fmt":      "nova",
    },
    # ── Mistral ──────────────────────────────────────────────
    {
        "model_id": "mistral.mistral-large-2402-v1:0",
        "provider": "Mistral AI",
        "short_id": "mistral-large",
        "fmt":      "messages",
    },
    {
        "model_id": "mistral.mistral-small-2402-v1:0",
        "provider": "Mistral AI",
        "short_id": "mistral-small",
        "fmt":      "messages",
    },
    {
        "model_id": "us.mistral.pixtral-large-2502-v1:0",
        "provider": "Mistral AI",
        "short_id": "pixtral-large",
        "fmt":      "messages",
    },
    # ── DeepSeek ─────────────────────────────────────────────
    {
        "model_id": "us.deepseek.r1-v1:0",
        "provider": "DeepSeek",
        "short_id": "deepseek-r1",
        "fmt":      "messages",
    },
    # ── Anthropic Claude via Bedrock ─────────────────────────
    {
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "provider": "Anthropic",
        "short_id": "claude-3-5-sonnet",
        "fmt":      "messages",
    },
    {
        "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "provider": "Anthropic",
        "short_id": "claude-3-haiku",
        "fmt":      "messages",
    },
]

# ─── Per-model pricing (USD per 1K tokens) ───────────────────
MODEL_PRICING = {
    "llama3-3-70b":      {"input": 0.00072, "output": 0.00072},
    "llama3-2-90b":      {"input": 0.00072, "output": 0.00072},
    "llama3-1-70b":      {"input": 0.00072, "output": 0.00072},
    "nova-lite":         {"input": 0.00006, "output": 0.00024},
    "nova-pro":          {"input": 0.0008,  "output": 0.0032},
    "nova-premier":      {"input": 0.0025,  "output": 0.0125},
    "mistral-large":     {"input": 0.004,   "output": 0.012},
    "mistral-small":     {"input": 0.001,   "output": 0.003},
    "pixtral-large":     {"input": 0.002,   "output": 0.006},
    "deepseek-r1":       {"input": 0.00135, "output": 0.00540},
    "claude-3-5-sonnet": {"input": 0.003,   "output": 0.015},
    "claude-3-haiku":    {"input": 0.00025, "output": 0.00125},
}


# ─── Request body builders ────────────────────────────────────

def _build_body(fmt: str, prompt: str) -> str:
    if fmt == "meta":
        return json.dumps({"prompt": prompt, "max_gen_len": 1024})
    elif fmt == "nova":
        return json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 1024},
        })
    else:
        # Anthropic / Mistral / DeepSeek
        return json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        })


# ─── Response text extractors ─────────────────────────────────

def _extract_text(fmt: str, body_json: dict) -> str:
    if fmt == "meta":
        return body_json.get("generation", "")
    elif fmt == "nova":
        return (
            body_json.get("output", {})
                     .get("message", {})
                     .get("content", [{}])[0]
                     .get("text", "")
        )
    else:
        # Anthropic / Mistral: content[0].text
        # DeepSeek: choices[0].message.content
        return (
            body_json.get("content", [{}])[0].get("text", "")
            or body_json.get("choices", [{}])[0]
                        .get("message", {}).get("content", "")
            or ""
        )


def _extract_tokens(fmt: str, body_json: dict) -> tuple[int, int]:
    if fmt == "meta":
        inp = body_json.get("prompt_token_count", 0)
        out = body_json.get("generation_token_count", 0)
        return inp, out

    if fmt == "nova":
        # Nova: usage.inputTokens / usage.outputTokens
        usage = body_json.get("usage", {})
        inp = usage.get("inputTokens", 0)
        out = usage.get("outputTokens", 0)
        return inp, out

    # Anthropic:  usage.input_tokens  / output_tokens
    # Mistral:    usage.prompt_tokens / completion_tokens
    # DeepSeek:   usage.prompt_tokens / completion_tokens
    usage = body_json.get("usage", {})
    inp = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or 0
    )
    out = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )
    return int(inp), int(out)


# ─── Cost calculator ─────────────────────────────────────────

def _calculate_cost(short_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(short_id, {"input": 0.000002, "output": 0.000002})
    return round(
        (input_tokens / 1000 * pricing["input"]) +
        (output_tokens / 1000 * pricing["output"]),
        6,
    )


# ─── Single model caller ──────────────────────────────────────

def _call_single_model(model_id: str, provider: str, short_id: str, fmt: str, prompt: str) -> dict:
    body = _build_body(fmt, prompt)
    start = time.time()
    response = bedrock.invoke_model(modelId=model_id, body=body)
    latency_ms = int((time.time() - start) * 1000)

    body_json = json.loads(response["body"].read())
    text = _extract_text(fmt, body_json)
    input_tokens, output_tokens = _extract_tokens(fmt, body_json)
    tokens = input_tokens + output_tokens
    cost = _calculate_cost(short_id, input_tokens, output_tokens)

    return {
        "model_id":   short_id,
        "provider":   provider,
        "response":   text,
        "tokens":     tokens,
        "cost":       cost,
        "latency_ms": latency_ms,
    }


# ─── Parallel caller ─────────────────────────────────────────

async def call_all_models(prompt: str) -> list[dict]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=len(BEDROCK_MODELS)) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _call_single_model,
                m["model_id"],
                m["provider"],
                m["short_id"],
                m["fmt"],
                prompt,
            )
            for m in BEDROCK_MODELS
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[BEDROCK ERROR] {BEDROCK_MODELS[i]['short_id']}: {r}")
        else:
            clean.append(r)
    return clean
