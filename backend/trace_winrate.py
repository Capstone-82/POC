import os, sys, asyncio
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from supabase import create_client
from services.supabase_client import get_model_win_rates
from services.recommender import aggregate_knn_signals_v2, score_and_rank_knn_candidates
from services.model_registry import get_model_ids_for_use_case

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

async def trace():
    use_case = 'code-generation'
    complexity = 'low'

    win_rates = await get_model_win_rates(use_case=use_case, complexity=complexity)
    print('win_rates count:', len(win_rates))

    emb_row = sb.table('prompt_embeddings').select('embedding').limit(1).execute()
    emb = emb_row.data[0]['embedding']

    result = sb.rpc('knn_search', {
        'query_embedding': emb, 'target_use_case': use_case,
        'result_limit': 240, 'min_similarity': 0.50,
    }).execute()
    neighbors = result.data or []
    print('KNN neighbors:', len(neighbors))

    signals = aggregate_knn_signals_v2(neighbors, use_case=use_case, win_rates=win_rates)
    print('Aggregated models:', len(signals))
    print()
    for mid, sig in signals.items():
        print(f'  {mid:30} win_rate={sig["win_rate"]}  sample_n={sig["sample_n"]}  p50_cost={sig["p50_cost"]}')

    print()
    allowed = get_model_ids_for_use_case(use_case)
    filtered = {k: v for k, v in signals.items() if k in allowed}
    print('After allowed filter:', len(filtered))
    ranked = score_and_rank_knn_candidates(filtered, use_case=use_case)
    print('Ranked top-5:')
    for r in ranked[:5]:
        print(f'  {r["model_id"]:30} win_rate={r.get("win_rate")}  value_score={r.get("value_score")}')

asyncio.run(trace())
