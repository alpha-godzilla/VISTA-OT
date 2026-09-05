#!/usr/bin/env python3
"""Summarize top-M mass-preserving single-UOT controls."""
import argparse, csv, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); rows=[]
with a.manifest.open(newline='',encoding='utf8') as f:
    for r in csv.DictReader(f,delimiter='\t'):
        m=json.loads(Path(r['chair_json']).read_text(encoding='utf8'))['overall_metrics']
        rows.append({**r,**{k:float(m[k]) for k in ('CHAIRs','CHAIRi','Recall','Precision','F1','Len')}})
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f'Wrote {a.output}')
