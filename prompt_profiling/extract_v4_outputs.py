import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('Architecture3_v4_Final.ipynb', 'r', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    ec = cell.get('execution_count', '?')
    outputs = cell.get('outputs', [])
    texts = []
    for o in outputs:
        if o.get('output_type') == 'stream':
            texts.extend(o.get('text', []))
        elif o.get('output_type') == 'execute_result':
            texts.extend(o.get('data', {}).get('text/plain', []))
    if texts:
        print(f"=== Cell {i} (exec={ec}) ===")
        print(''.join(texts))
