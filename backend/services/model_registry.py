"""
Use-case -> model mapping and balanced rotation helpers.

Each use case maps to a set of short_ids that should be invoked.
Models not listed under a use case will NOT be called for that use case.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

# ─── Static Fallbacks & Registry Paths ──────────────────────────
SERVICES_DIR = Path(__file__).parent.resolve()
POC_DIR = SERVICES_DIR.parent.parent
REGISTRY_PATH = POC_DIR / "prompt_profiling&model_routing" / "model_registry_v3.json"

STATIC_TEXT_GENERATION_MODELS = {
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

STATIC_CODE_GENERATION_MODELS = {
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

STATIC_REASONING_MODELS = {
    "deepseek-r1",
    "magistral-small",
    "nova-premier",
    "nova-lite",
    "llama4-maverick",
    "pixtral-large-2",
    "mistral-large",
    "nova-pro",
}

STATIC_MODEL_PROVIDER = {
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


def normalize_model_id(name: str) -> str:
    """Normalize model ID by stripping common prefixes, replacing dots, and formatting llama."""
    name = name.lower().replace("_", "-").replace(" ", "-").strip()
    for prefix in ["amazon-", "google-", "meta-", "anthropic-", "openai-", "mistral-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace(".", "-")
    name = re.sub(r'^llama-(\d+)', r'llama\1', name)
    return name


def models_match(id1: str, id2: str) -> bool:
    """Check if two model IDs match under normalization."""
    n1 = normalize_model_id(id1)
    n2 = normalize_model_id(id2)
    return n1 == n2 or n1.startswith(n2) or n2.startswith(n1)


class DynamicModelRegistry:
    def __init__(self):
        self.last_loaded = 0.0
        self.providers = dict(STATIC_MODEL_PROVIDER)
        self.use_cases = {
            "text-generation": set(STATIC_TEXT_GENERATION_MODELS),
            "code-generation": set(STATIC_CODE_GENERATION_MODELS),
            "reasoning": set(STATIC_REASONING_MODELS),
        }

    def _reload_if_needed(self):
        if not REGISTRY_PATH.exists():
            return
        try:
            mtime = os.path.getmtime(REGISTRY_PATH)
            if mtime <= self.last_loaded:
                return
        except Exception:
            return

        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            dyn_text_gen = set()
            dyn_code_gen = set()
            dyn_reasoning = set()
            dyn_providers = {}

            for m in data.get("models", []):
                model_id = m["model_id"]
                provider = m.get("provider", "unknown")
                domain_strengths = [s.lower() for s in m.get("domain_strengths", [])]
                reasoning_mode = bool(m.get("reasoning_mode", False))

                # Normalize model_id
                norm_id = normalize_model_id(model_id)

                # Store provider names
                dyn_providers[norm_id] = provider
                dyn_providers[model_id] = provider

                # Categorize based on strengths & capabilities
                is_coding = any(
                    s in domain_strengths
                    for s in ["coding", "complex_coding", "agentic_coding", "code_completion", "fill_in_the_middle"]
                )
                is_reasoning = reasoning_mode or any(
                    s in domain_strengths
                    for s in [
                        "reasoning",
                        "complex_reasoning",
                        "strategic_reasoning",
                        "highest_stakes_reasoning",
                        "cost_efficient_reasoning",
                        "cheap_reasoning_with_large_context",
                    ]
                )

                # All models support text generation
                dyn_text_gen.add(norm_id)
                dyn_text_gen.add(model_id)

                if is_coding:
                    dyn_code_gen.add(norm_id)
                    dyn_code_gen.add(model_id)
                if is_reasoning:
                    dyn_reasoning.add(norm_id)
                    dyn_reasoning.add(model_id)

            # Update cache
            self.providers = dict(STATIC_MODEL_PROVIDER)
            self.providers.update(dyn_providers)

            self.use_cases["text-generation"] = STATIC_TEXT_GENERATION_MODELS.union(dyn_text_gen)
            self.use_cases["code-generation"] = STATIC_CODE_GENERATION_MODELS.union(dyn_code_gen)
            self.use_cases["reasoning"] = STATIC_REASONING_MODELS.union(dyn_reasoning)

            self.last_loaded = mtime
            print(f"[REGISTRY] Loaded {len(dyn_providers) // 2} models dynamically from {REGISTRY_PATH}")
        except Exception as e:
            print(f"[REGISTRY ERROR] Failed to reload dynamic model registry: {e}")

    def get_model_ids_for_use_case(self, use_case: str) -> set[str]:
        self._reload_if_needed()
        return self.use_cases.get(use_case, set())

    def get_provider(self, model_id: str) -> str:
        self._reload_if_needed()
        norm_id = normalize_model_id(model_id)
        if norm_id in self.providers:
            return self.providers[norm_id]
        return self.providers.get(model_id, "unknown")


# Instantiate global dynamic registry
_registry_instance = DynamicModelRegistry()


# ─── Dict Proxies for Backward Compatibility ────────────────────
class ProviderProxy(dict):
    def get(self, key, default=None):
        val = _registry_instance.get_provider(key)
        return val if val != "unknown" else default

    def __getitem__(self, key):
        val = _registry_instance.get_provider(key)
        if val == "unknown":
            raise KeyError(key)
        return val

    def __contains__(self, key):
        return _registry_instance.get_provider(key) != "unknown"


class UseCaseModelsProxy(dict):
    def get(self, key, default=None):
        val = _registry_instance.get_model_ids_for_use_case(key)
        return val if val else default

    def __getitem__(self, key):
        val = _registry_instance.get_model_ids_for_use_case(key)
        if not val:
            raise KeyError(key)
        return val

    def __contains__(self, key):
        return len(_registry_instance.get_model_ids_for_use_case(key)) > 0

    def keys(self):
        return ["text-generation", "code-generation", "reasoning"]


MODEL_PROVIDER = ProviderProxy()
USE_CASE_MODELS = UseCaseModelsProxy()

# Dummy collections to prevent import issues in other scripts
TEXT_GENERATION_MODELS = _registry_instance.get_model_ids_for_use_case("text-generation")
CODE_GENERATION_MODELS = _registry_instance.get_model_ids_for_use_case("code-generation")
REASONING_MODELS = _registry_instance.get_model_ids_for_use_case("reasoning")


# ─── Stable Seed Selection helpers ──────────────────────────────
def _stable_int(seed_text: str) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _rotate(items: list[str], start: int) -> list[str]:
    if not items:
        return []
    pivot = start % len(items)
    return items[pivot:] + items[:pivot]


def get_ordered_model_ids_for_use_case(use_case: str) -> list[str]:
    models = _registry_instance.get_model_ids_for_use_case(use_case)
    if not models:
        raise ValueError(
            f"Unknown use_case '{use_case}' or no models loaded. "
            f"Must be one of: {', '.join(USE_CASE_MODELS.keys())}"
        )
    return sorted(models, key=lambda model_id: (_registry_instance.get_provider(model_id), model_id))


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
    """Deterministically select a balanced subset of models for a prompt."""
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
    return _registry_instance.get_model_ids_for_use_case(use_case)
