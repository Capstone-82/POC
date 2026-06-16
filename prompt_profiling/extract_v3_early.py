import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('Architecture3_v3_Final.ipynb', 'r', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    ec = cell.get('execution_count', '?')
    if ec and ec <= 10:
        outputs = cell.get('outputs', [])
        texts = []
        for o in outputs:
            if o.get('output_type') == 'stream':
                texts.extend(o.get('text', []))
        if texts:
            print(f"=== Cell {i} (exec={ec}) ===")
            print(''.join(texts))
