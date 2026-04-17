import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from supabase import create_client
import json

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# 1. Embedding dimension
row = sb.table('prompt_embeddings').select('prompt_hash, embedding').limit(1).execute()
if row.data:
    emb = row.data[0]['embedding']
    if isinstance(emb, list):
        print(f'[OK] embedding dim: {len(emb)}  (should be 1536)')
    elif isinstance(emb, str):
        parsed = json.loads(emb)
        print(f'[OK] embedding dim (str): {len(parsed)}  (should be 1536)')
    else:
        print(f'[??] embedding type={type(emb).__name__} val[:60]={str(emb)[:60]}')

# 2. Null count
null_r = sb.table('prompt_embeddings').select('id', count='exact').is_('embedding', 'null').execute()
total  = sb.table('prompt_embeddings').select('id', count='exact').execute()
print(f'prompt_embeddings total={total.count}  null={null_r.count}')

# 3. knn_search RPC with 1536-dim
try:
    r = sb.rpc('knn_search', {
        'query_embedding': [0.01]*1536,
        'target_use_case': 'code-generation',
        'result_limit': 10,
        'min_similarity': 0.0
    }).execute()
    rows = r.data or []
    print(f'knn_search 1536-dim: {len(rows)} rows')
    if rows:
        print(f'columns: {list(rows[0].keys())}')
        for r2 in rows[:3]:
            print(f'  model={r2["model_id"]:28} syntax_pass={r2.get("syntax_pass")} is_correct={r2.get("is_correct")}')
except Exception as e:
    print(f'knn_search error: {e}')

# 4. Spot-check win_rate join
from services.supabase_client import get_model_win_rates
import asyncio
async def check_wr():
    wr = await get_model_win_rates('code-generation', 'low')
    print(f'win_rates loaded: {len(wr)} models')
    for mid, v in list(wr.items())[:3]:
        print(f'  {mid}: win_rate={v["win_rate"]}')
asyncio.run(check_wr())
