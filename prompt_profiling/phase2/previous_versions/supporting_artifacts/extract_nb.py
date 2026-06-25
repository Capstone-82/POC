import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('phase2/phase2_a1_v1.ipynb', 'r', encoding='utf-8'))
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
        print(f"MARKDOWN CELL {i}")
        print(f"{'='*80}")
        print(source[:300])
        continue
    
    if ct == 'code':
        print(f"\n{'='*80}")
        print(f"CODE CELL {i} (exec={ec})")
        print(f"{'='*80}")
        print(source[:2500])
        if len(source) > 2500:
            print(f"... [truncated, {len(source)} total chars]")
        
        # Show outputs
        outputs = cell.get('outputs', [])
        if outputs:
            print(f"\n--- OUTPUT ({len(outputs)} items) ---")
            for o in outputs:
                otype = o.get('output_type', '')
                if otype == 'stream':
                    text = ''.join(o.get('text', []))
                    print(text[:1500])
                    if len(text) > 1500:
                        print(f"... [output truncated, {len(text)} chars]")
                elif otype == 'execute_result':
                    text = ''.join(o.get('data', {}).get('text/plain', []))
                    print(text[:800])
                elif otype == 'error':
                    print(f"ERROR: {o.get('ename','')}: {''.join(o.get('evalue',''))[:200]}")
