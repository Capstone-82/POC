"""
Use-case → model mapping.

Each use case maps to a set of short_ids that should be invoked.
Models not listed under a use case will NOT be called for that use case.
"""

# ─── Text Generation ─────────────────────────────────────────
# Strong instruction following, long-form content, summarization,
# multilingual output, and general chat.

TEXT_GENERATION_MODELS = {
    # Bedrock
    "llama4-scout",
    "llama4-maverick",
    "llama3-3-70b",
    "llama3-2-90b",
    "llama3-1-70b",
    "nova-lite",
    "nova-pro",
    "nova-premier",
    "mistral-large",
    "mistral-small",
    "pixtral-large-2",
    # Vertex
    "gemini-3-1-pro",
    "gemini-3-1-flash-lite",
    "gemini-2-5-pro",
    "gemini-2-5-flash",
    "gemini-2-0-flash",
    "gemini-2-0-flash-lite",
}

# ─── Code Generation ─────────────────────────────────────────
# Strong HumanEval, SWE-bench, LiveCodeBench, or explicitly
# designed for coding.

CODE_GENERATION_MODELS = {
    # Bedrock
    "devstral-2",
    "llama4-maverick",
    "llama3-3-70b",
    "nova-pro",
    "nova-premier",
    "pixtral-large-2",
    "mistral-large",
    "magistral-small",
    "deepseek-r1",
    "ministral-3-8b",
    # Vertex
    "gemini-3-1-pro",
    "gemini-2-5-pro",
    "gemini-2-5-flash",
    "gemini-2-0-flash",
}

# ─── Reasoning ────────────────────────────────────────────────
# Trained/evaluated for AIME, GPQA, MATH, MMLU, chain-of-thought,
# multi-step problem solving.

REASONING_MODELS = {
    # Bedrock
    "deepseek-r1",
    "magistral-small",
    "nova-premier",
    "nova-lite",
    "llama4-maverick",
    "pixtral-large-2",
    "mistral-large",
    "nova-pro",
    # Vertex
    "gemini-3-1-pro",
    "gemini-2-5-pro",
    "gemini-2-5-flash",
    "gemini-2-0-flash",
}

# ─── Master lookup ────────────────────────────────────────────

USE_CASE_MODELS = {
    "text-generation":  TEXT_GENERATION_MODELS,
    "code-generation":  CODE_GENERATION_MODELS,
    "reasoning":        REASONING_MODELS,
}


def get_model_ids_for_use_case(use_case: str) -> set[str]:
    """Return the set of short_ids that should be invoked for a given use case."""
    models = USE_CASE_MODELS.get(use_case)
    if models is None:
        raise ValueError(
            f"Unknown use_case '{use_case}'. "
            f"Must be one of: {', '.join(USE_CASE_MODELS.keys())}"
        )
    return models
