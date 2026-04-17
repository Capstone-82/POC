"""
Full end-to-end recommender test — confirms win_rate flows through KNN.
Run from: backend/
"""
import os, sys, asyncio
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()

from services.recommender import get_recommendation

async def test(prompt, use_case, current_model):
    print(f"\nprompt   : {prompt[:60]}")
    print(f"use_case : {use_case}  current_model: {current_model}")
    print("-" * 60)
    result = await get_recommendation(
        use_case=use_case,
        prompt=prompt,
        current_model=current_model,
    )
    print(f"data_source       : {result['data_source']}")
    print(f"filter_level      : {result['filter_level']}")
    print(f"knn_neighbors     : {result.get('knn_neighbors_used')}")
    print(f"recommended_model : {result['recommended_model']}")
    print(f"expected_win_rate : {result.get('expected_win_rate')}")
    print(f"win_rate_delta    : {result.get('win_rate_delta')}")
    print(f"switch_recommended: {result['switch_recommended']}")
    print(f"policy_reason     : {result['policy_reason']}")
    print()
    print("Top candidates:")
    for c in result.get("top_candidates", [])[:5]:
        wr = c.get("win_rate")
        print(f"  {c['model_id']:28} win_rate={wr}  value_score={c.get('value_score')}")

async def main():
    tests = [
        ("Write a binary search algorithm in Python with type hints", "code-generation", "nova-pro"),
        ("Explain the CAP theorem to a junior engineer", "reasoning", "llama3-3-70b"),
        ("Write a professional email declining a job offer", "text-generation", "nova-pro"),
    ]
    for args in tests:
        await test(*args)

asyncio.run(main())
