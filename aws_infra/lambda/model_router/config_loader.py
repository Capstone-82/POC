"""
Config loader for the model router Lambda.
Fetches per-app configurations from AWS SSM Parameter Store and caches them
for up to CACHE_TTL_SECONDS to avoid cold SSM calls on every request.
"""

import json
import os
import time
import boto3
from typing import Optional

CACHE_TTL_SECONDS = int(os.environ.get("CONFIG_CACHE_TTL", "300"))  # 5 minutes
AWS_REGION        = os.environ.get("AWS_REGION", "us-east-1")
SSM_PATH_PREFIX   = os.environ.get("SSM_PATH_PREFIX", "/model-router")

_ssm_client  = None
_cache: dict = {}          # key: "{app_id}/{env}"  value: {"data": ..., "ts": float}


def _get_ssm():
    global _ssm_client
    if _ssm_client is None:
        _ssm_client = boto3.client("ssm", region_name=AWS_REGION)
    return _ssm_client


def get_app_config(app_id: str, env: str = "prod") -> Optional[dict]:
    """
    Return the configuration for the given app and environment.
    Returns None if not found.
    """
    cache_key = f"{app_id}/{env}"
    now = time.time()

    cached = _cache.get(cache_key)
    if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
        return cached["data"]

    param_name = f"{SSM_PATH_PREFIX}/{app_id}/{env}"
    try:
        resp  = _get_ssm().get_parameter(Name=param_name, WithDecryption=False)
        value = resp["Parameter"]["Value"]
        data  = json.loads(value)
        _cache[cache_key] = {"data": data, "ts": now}
        return data
    except _get_ssm().exceptions.ParameterNotFound:
        return None
    except Exception as exc:
        print(f"[CONFIG LOADER] Failed to fetch {param_name}: {exc}")
        return None


def list_all_app_configs() -> list[dict]:
    """
    Return all app configs stored under SSM_PATH_PREFIX by paginating GetParametersByPath.
    """
    ssm    = _get_ssm()
    path   = SSM_PATH_PREFIX
    params = []
    kwargs = {"Path": path, "Recursive": True, "WithDecryption": False}

    while True:
        resp = ssm.get_parameters_by_path(**kwargs)
        for p in resp.get("Parameters", []):
            try:
                params.append(json.loads(p["Value"]))
            except Exception:
                pass
        next_token = resp.get("NextToken")
        if not next_token:
            break
        kwargs["NextToken"] = next_token

    return params


def put_app_config(config: dict) -> None:
    """
    Write or overwrite an app config in SSM Parameter Store.
    Clears the local cache entry.
    """
    app_id = config["app_id"]
    env    = config.get("env", "prod")
    param_name = f"{SSM_PATH_PREFIX}/{app_id}/{env}"

    _get_ssm().put_parameter(
        Name=param_name,
        Value=json.dumps(config),
        Type="String",
        Overwrite=True,
    )
    # Invalidate cache
    _cache.pop(f"{app_id}/{env}", None)


def delete_app_config(app_id: str, env: str = "prod") -> None:
    """
    Delete an app config from SSM Parameter Store and clear local cache.
    """
    param_name = f"{SSM_PATH_PREFIX}/{app_id}/{env}"
    try:
        _get_ssm().delete_parameter(Name=param_name)
    except Exception as exc:
        print(f"[CONFIG LOADER] Delete failed for {param_name}: {exc}")
    _cache.pop(f"{app_id}/{env}", None)
