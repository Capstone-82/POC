from fastapi import APIRouter
from models.schemas import InferenceRequest, InferenceResponse
from services.recommender import get_recommendation, get_recommendation_options

router = APIRouter()


@router.get("/options")
async def recommendation_options():
    return await get_recommendation_options()


@router.post("/recommend", response_model=InferenceResponse)
async def recommend(req: InferenceRequest):
    result = await get_recommendation(
        use_case=req.use_case,
        prompt=req.prompt,
        current_model=req.current_model,
        min_accuracy_gain=req.min_accuracy_gain,
        max_cost_increase_pct=req.max_cost_increase_pct,
        max_latency_increase_pct=req.max_latency_increase_pct,
    )
    return InferenceResponse(**result)
