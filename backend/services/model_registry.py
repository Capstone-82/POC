"""
Use-case -> model mapping and balanced rotation helpers.

Each use case maps to a set of short_ids that should be invoked.
Models not listed under a use case will NOT be called for that use case.
"""

from __future__ import annotations

import hashlib
from typing import Iterable


TEXT_GENERATION_MODELS = {
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
}

CODE_GENERATION_MODELS = {
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
}

REASONING_MODELS = {
    "deepseek-r1",
    "magistral-small",
    "nova-premier",
    "nova-lite",
    "llama4-maverick",
    "pixtral-large-2",
    "mistral-large",
    "nova-pro",
}

USE_CASE_MODELS = {
    "text-generation": TEXT_GENERATION_MODELS,
    "code-generation": CODE_GENERATION_MODELS,
    "reasoning": REASONING_MODELS,
}

MODEL_PROVIDER = {
    "llama4-scout": "Meta",
    "llama4-maverick": "Meta",
    "llama3-3-70b": "Meta",
    "llama3-2-90b": "Meta",
    "llama3-1-70b": "Meta",
    "nova-lite": "Amazon",
    "nova-pro": "Amazon",
    "nova-premier": "Amazon",
    "devstral-2": "Mistral AI",
    "ministral-3-8b": "Mistral AI",
    "magistral-small": "Mistral AI",
    "pixtral-large-2": "Mistral AI",
    "mistral-large": "Mistral AI",
    "mistral-small": "Mistral AI",
    "deepseek-r1": "DeepSeek",
}

DEFAULT_ROTATION_TARGET = {
    "text-generation": 4,
    "code-generation": 4,
    "reasoning": 4,
}


def _stable_int(seed_text: str) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _rotate(items: list[str], start: int) -> list[str]:
    if not items:
        return []
    pivot = start % len(items)
    return items[pivot:] + items[:pivot]


def get_ordered_model_ids_for_use_case(use_case: str) -> list[str]:
    models = USE_CASE_MODELS.get(use_case)
    if models is None:
        raise ValueError(
            f"Unknown use_case '{use_case}'. "
            f"Must be one of: {', '.join(USE_CASE_MODELS.keys())}"
        )
    return sorted(models, key=lambda model_id: (MODEL_PROVIDER.get(model_id, "zzz"), model_id))


def _round_robin_by_provider(model_ids: Iterable[str], start_seed: int) -> list[str]:
    provider_buckets: dict[str, list[str]] = {}
    for model_id in model_ids:
        provider = MODEL_PROVIDER.get(model_id, "Unknown")
        provider_buckets.setdefault(provider, []).append(model_id)

    ordered_providers = _rotate(sorted(provider_buckets), start_seed)
    for index, provider in enumerate(ordered_providers):
        provider_buckets[provider] = _rotate(provider_buckets[provider], start_seed + index)

    merged: list[str] = []
    while True:
        progressed = False
        for provider in ordered_providers:
            bucket = provider_buckets[provider]
            if bucket:
                merged.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return merged


def select_rotating_models_for_prompt(
    use_case: str,
    prompt_hash: str,
    prompt_complexity: str,
    clarity: str,
    min_models: int = 3,
    max_models: int = 5,
) -> list[str]:
    """
    Deterministically select a balanced subset of models for a prompt.

    The seed uses prompt metadata so:
    - repeated runs are reproducible
    - neighboring prompts do not always hit the same model subset
    - provider diversity is preserved when possible
    """
    ordered = get_ordered_model_ids_for_use_case(use_case)
    if not ordered:
        return []

    min_models = max(1, min_models)
    max_models = max(min_models, max_models)

    seed = _stable_int(f"{use_case}|{prompt_hash}|{prompt_complexity}|{clarity}")
    rotated = _round_robin_by_provider(ordered, start_seed=seed)

    span = max_models - min_models + 1
    target = DEFAULT_ROTATION_TARGET.get(use_case, min_models)
    target = max(min_models, min(max_models, target))
    if span > 1:
        target = min_models + (seed % span)

    return rotated[: min(target, len(rotated))]


def get_model_ids_for_use_case(use_case: str) -> set[str]:
    """Return the set of short_ids that should be invoked for a given use case."""
    return set(get_ordered_model_ids_for_use_case(use_case))
