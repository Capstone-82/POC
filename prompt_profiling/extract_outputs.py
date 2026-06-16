import json

with open('Architecture3_v2_Enhanced.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    exec_count = cell.get('execution_count', '?')
    outputs = cell.get('outputs', [])
    stream_texts = []
    for o in outputs:
        if o.get('output_type') == 'stream':
            stream_texts.extend(o.get('text', []))
    if stream_texts:
        print(f"=== Cell {i} (exec={exec_count}) ===")
        print(''.join(stream_texts))
