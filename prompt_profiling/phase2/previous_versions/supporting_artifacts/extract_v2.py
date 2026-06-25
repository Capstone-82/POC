import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('phase2/phase2_a1_v2.ipynb', 'r', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f"Code cells: {len(code_cells)}, Markdown cells: {len(md_cells)}")

for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    ec = cell.get('execution_count', '-')
    source = ''.join(cell.get('source', []))

    if ct == 'markdown':
        print(f"\n{'='*80}")
        print(f"MD {i}: {source[:200]}")
        continue

    if ct == 'code':
        print(f"\n{'='*80}")
        print(f"CODE {i} (exec={ec}) [{len(source)} chars]")
        print(source[:3000])
        if len(source) > 3000:
            print(f"... [truncated, {len(source)} total chars]")

        outputs = cell.get('outputs', [])
        if outputs:
            print(f"\n--- OUTPUT ({len(outputs)} items) ---")
            for o in outputs:
                otype = o.get('output_type', '')
                if otype == 'stream':
                    text = ''.join(o.get('text', []))
                    print(text[:2000])
                    if len(text) > 2000:
                        print(f"... [output truncated, {len(text)} chars]")
                elif otype == 'execute_result':
                    text = ''.join(o.get('data', {}).get('text/plain', []))
                    print(text[:800])
                elif otype == 'error':
                    ename = o.get('ename', '')
                    evalue = ''.join(o.get('evalue', ''))[:300]
                    print(f"ERROR: {ename}: {evalue}")
