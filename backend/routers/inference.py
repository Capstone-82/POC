from fastapi import APIRouter
from models.schemas import InferenceRequest, InferenceResponse
from services.evaluator import evaluate_prompt
from services.recommender import get_recommendation

router = APIRouter()


@router.post("/recommend", response_model=InferenceResponse)
async def recommend(req: InferenceRequest):

    # Step 1: Classify prompt
    prompt_eval = await evaluate_prompt(req.prompt, evaluator_model="gemini-2.0-flash")

    # Step 2: Run recommendation logic against Supabase benchmark data
    result = await get_recommendation(
        use_case=req.use_case,
        complexity=prompt_eval["prompt_complexity"],
        quality_score=prompt_eval["prompt_quality_score"],
        current_model=req.current_model,
    )

    return InferenceResponse(
        complexity=prompt_eval["prompt_complexity"],
        quality_score=prompt_eval["prompt_quality_score"],
        current_model=req.current_model,
        **result,
    )
