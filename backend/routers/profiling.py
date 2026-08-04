import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dataclasses import asdict

# Add prompt_profiling&model_routing to Python path
POC_DIR = Path(__file__).parent.parent.parent.resolve()
MODULE_DIR = POC_DIR / "prompt_profiling&model_routing"

if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from router import route_model, PromptProfiler, ModelRegistry, ModelRouter

router = APIRouter()

# Global router instance for low latency
_profiler = None
_registry = None
_router = None

def get_router():
    global _profiler, _registry, _router
    if _router is None:
        pkl_path = str(MODULE_DIR / "prompt_profiler.pkl")
        registry_path = str(MODULE_DIR / "model_registry_v3.json")
        if not os.path.exists(pkl_path):
            raise RuntimeError(f"Pickle bundle not found at {pkl_path}")
        if not os.path.exists(registry_path):
            raise RuntimeError(f"Registry JSON not found at {registry_path}")
        
        _profiler = PromptProfiler(pkl_path)
        _registry = ModelRegistry(registry_path)
        _router = ModelRouter(_profiler, _registry)
    return _router


class ProfileAndRouteRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    include_legacy: Optional[bool] = False
    top_n: Optional[int] = 3
    enterprise_criticality: Optional[str] = "standard"
    required_capabilities: Optional[List[str]] = None


@router.post("/route")
def profile_and_route(req: ProfileAndRouteRequest) -> Dict[str, Any]:
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty.")
    
    try:
        r = get_router()
        result = r.route(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            include_legacy=req.include_legacy if req.include_legacy is not None else False,
            top_n=req.top_n if req.top_n is not None else 3,
            enterprise_criticality=req.enterprise_criticality or "standard",
            required_capabilities=req.required_capabilities or [],
        )
        return asdict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling & routing failed: {str(e)}")


@router.get("/health")
def health_check():
    try:
        r = get_router()
        return {"status": "ok", "models_loaded": len(r.registry.get_models())}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
