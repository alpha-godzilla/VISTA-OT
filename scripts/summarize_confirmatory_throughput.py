#!/usr/bin/env python3
"""Turn the short Phase-A benchmark into a conservative overnight budget."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--runtime",type=Path,required=True); p.add_argument("--generation-root",type=Path,required=True)
    p.add_argument("--gpus",type=int,required=True); p.add_argument("--phase-a-hours",type=float,default=5.0); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); run=json.loads(a.runtime.read_text()); rows=[]
    for path in a.generation_root.rglob("generation.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    seconds=max(1,run["seconds"]); actual=len({int(row["sample_id"]) for row in rows})
    image_gpu_hour=actual/(a.gpus*seconds/3600)
    suggested=int(image_gpu_hour*a.gpus*a.phase_a_hours*.90) # 10% flush/crash margin
    payload={"requested_images":run["requested"],"completed_images":actual,"failed_or_missing_images":run["requested"]-actual,
      "wall_seconds":seconds,"images_per_gpu_hour":image_gpu_hour,"mean_generation_length":sum(row["generation_length"] for row in rows)/max(1,len(rows)),
      "phase_a_budget_hours":a.phase_a_hours,"suggested_phase_a_images":suggested,
      "note":"Suggestion is throughput-only. Inspect mixed-image yield after CHAIR; it never implies an independent seed replicate."}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
