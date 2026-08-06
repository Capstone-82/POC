"""
FastAPI router — /api/model-routing/*

Exposes the AWS Model Routing layer to the frontend without requiring
the frontend to hold AWS credentials.  All SSM reads/writes happen here
on the backend (same AWS credentials used by the rest of the backend).

Endpoints:
  GET  /api/model-routing/apps                  — list all app configs
  POST /api/model-routing/apps                  — create / update an app config
  DELETE /api/model-routing/apps/{app_id}/{env} — delete an app config
  POST /api/model-routing/test                  — dry-run routing decision
  GET  /api/model-routing/models                — list available models from the registry
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── Add aws_infra to path so we can reuse config_loader ──────────
_INFRA_PATH = Path(__file__).parent.parent.parent / "aws_infra" / "lambda" / "model_router"
if str(_INFRA_PATH) not in sys.path:
    sys.path.insert(0, str(_INFRA_PATH))

# Registry path for available models
_REGISTRY_PATH = (
    Path(__file__).parent.parent.parent
    / "prompt_profiling&model_routing"
    / "model_registry_v3.json"
)

router = APIRouter(tags=["model-routing"])


# ── Pydantic Schemas ──────────────────────────────────────────────

class ContextLimits(BaseModel):
    max_input_tokens:  int = Field(8000,  ge=1)
    max_output_tokens: int = Field(4096,  ge=1)
    max_total_tokens:  int = Field(10000, ge=1)


class ThrottleSettings(BaseModel):
    rate_limit:    int = Field(100,   ge=1)
    burst_limit:   int = Field(200,   ge=1)
    quota_per_day: int = Field(10000, ge=1)


class AppConfig(BaseModel):
    app_id:         str
    env:            str                  = "prod"
    allowed_models: List[str]            = []
    context_limits: ContextLimits        = ContextLimits()
    throttle:       ThrottleSettings     = ThrottleSettings()


class TestRouteRequest(BaseModel):
    app_id:            str
    env:               str = "prod"
    model_id:          str
    prompt:            str
    max_output_tokens: int = 0


# ── Helpers ───────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.35))


def _load_config_loader():
    """Lazy-import the SSM config_loader to avoid crashing at startup
    when AWS credentials are not available in the dev environment."""
    try:
        import config_loader as cl
        return cl
    except Exception:
        return None


def _fallback_configs() -> list[dict]:
    """Return configs from the local JSON file when SSM is unavailable."""
    local = (
        Path(__file__).parent.parent.parent
        / "aws_infra" / "config" / "app_configs.json"
    )
    if local.exists():
        with open(local, "r") as f:
            return json.load(f).get("apps", [])
    return []


def _available_models() -> list[str]:
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH, "r") as f:
            data = json.load(f)
        return [m["model_id"] for m in data.get("models", [])]
    return []


# ── Routes ────────────────────────────────────────────────────────

@router.get("/apps", response_model=List[Dict[str, Any]])
def list_apps():
    """Return all app configs from SSM (or local fallback)."""
    cl = _load_config_loader()
    if cl:
        try:
            return cl.list_all_app_configs()
        except Exception:
            pass
    return _fallback_configs()


@router.post("/apps", response_model=Dict[str, str])
def upsert_app(config: AppConfig):
    """Create or update an app configuration in SSM."""
    cl = _load_config_loader()
    payload = config.model_dump()
    if cl:
        try:
            cl.put_app_config(payload)
            return {"status": "ok", "message": f"Config saved for {config.app_id}/{config.env}."}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"SSM write failed: {exc}")

    # Fallback: write to local JSON file (dev mode)
    local_path = (
        Path(__file__).parent.parent.parent
        / "aws_infra" / "config" / "app_configs.json"
    )
    if local_path.exists():
        with open(local_path, "r") as f:
            data = json.load(f)
        apps = data.get("apps", [])
        # Replace if exists else append
        apps = [
            a for a in apps
            if not (a["app_id"] == config.app_id and a.get("env") == config.env)
        ]
        apps.append(payload)
        data["apps"] = apps
        with open(local_path, "w") as f:
            json.dump(data, f, indent=2)
    return {"status": "ok", "message": f"Config saved locally for {config.app_id}/{config.env}."}


@router.delete("/apps/{app_id}/{env}", response_model=Dict[str, str])
def delete_app(app_id: str, env: str):
    """Delete an app configuration from SSM."""
    cl = _load_config_loader()
    if cl:
        try:
            cl.delete_app_config(app_id, env)
            return {"status": "ok", "message": f"Deleted {app_id}/{env}."}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"SSM delete failed: {exc}")

    # Fallback: remove from local JSON
    local_path = (
        Path(__file__).parent.parent.parent
        / "aws_infra" / "config" / "app_configs.json"
    )
    if local_path.exists():
        with open(local_path, "r") as f:
            data = json.load(f)
        data["apps"] = [
            a for a in data.get("apps", [])
            if not (a["app_id"] == app_id and a.get("env") == env)
        ]
        with open(local_path, "w") as f:
            json.dump(data, f, indent=2)
    return {"status": "ok", "message": f"Deleted {app_id}/{env} locally."}


@router.post("/test", response_model=Dict[str, Any])
def test_route(req: TestRouteRequest):
    """
    Dry-run the routing decision for a given app + model + prompt.
    Does NOT actually invoke the model — returns allow/reject/context decision only.
    """
    # Load config
    cl = _load_config_loader()
    config = None
    if cl:
        try:
            config = cl.get_app_config(req.app_id, req.env)
        except Exception:
            pass

    if config is None:
        # Try local fallback
        for cfg in _fallback_configs():
            if cfg.get("app_id") == req.app_id and cfg.get("env", "prod") == req.env:
                config = cfg
                break

    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"No config found for app '{req.app_id}' / env '{req.env}'.",
        )

    allowed_models = [m.lower() for m in config.get("allowed_models", [])]

    # Allow-list check
    if req.model_id.lower() not in allowed_models:
        return {
            "decision":       "rejected",
            "reason":         "model_not_allowed",
            "message":        (
                f"Model '{req.model_id}' is not on the allow-list for "
                f"'{req.app_id}/{req.env}'."
            ),
            "allowed_models": config.get("allowed_models", []),
        }

    # Context window check
    limits       = config.get("context_limits", {})
    max_in       = limits.get("max_input_tokens",  8000)
    max_out_lim  = limits.get("max_output_tokens", 4096)
    max_tot      = limits.get("max_total_tokens",  10000)

    est_input    = _estimate_tokens(req.prompt)
    est_output   = req.max_output_tokens if req.max_output_tokens > 0 else max_out_lim
    est_total    = est_input + est_output

    if est_input > max_in:
        return {
            "decision":                 "rejected",
            "reason":                   "context_window_exceeded",
            "violation":                "max_input_tokens",
            "message":                  f"Estimated {est_input:,} input tokens exceeds limit of {max_in:,}.",
            "estimated_input_tokens":   est_input,
            "estimated_output_tokens":  est_output,
            "estimated_total_tokens":   est_total,
            "limits":                   limits,
        }

    if est_output > max_out_lim:
        return {
            "decision":                 "rejected",
            "reason":                   "context_window_exceeded",
            "violation":                "max_output_tokens",
            "message":                  f"Requested {est_output:,} output tokens exceeds limit of {max_out_lim:,}.",
            "estimated_input_tokens":   est_input,
            "estimated_output_tokens":  est_output,
            "estimated_total_tokens":   est_total,
            "limits":                   limits,
        }

    if est_total > max_tot:
        return {
            "decision":                 "rejected",
            "reason":                   "context_window_exceeded",
            "violation":                "max_total_tokens",
            "message":                  f"Estimated total {est_total:,} tokens exceeds limit of {max_tot:,}.",
            "estimated_input_tokens":   est_input,
            "estimated_output_tokens":  est_output,
            "estimated_total_tokens":   est_total,
            "limits":                   limits,
        }

    return {
        "decision":                 "allowed",
        "reason":                   "all_checks_passed",
        "message":                  f"Request approved — model '{req.model_id}' is allowed and fits within context limits.",
        "model_id":                 req.model_id,
        "app_id":                   req.app_id,
        "env":                      req.env,
        "estimated_input_tokens":   est_input,
        "estimated_output_tokens":  est_output,
        "estimated_total_tokens":   est_total,
        "limits":                   limits,
        "throttle":                 config.get("throttle", {}),
    }


@router.get("/models", response_model=List[str])
def get_available_models():
    """Return all model IDs available in the registry."""
    return _available_models()
