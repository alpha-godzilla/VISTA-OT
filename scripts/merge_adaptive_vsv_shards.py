#!/usr/bin/env python3
"""Validate and merge adaptive-VSV JSONL shards without accepting duplicates."""
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--shards",nargs="+",type=Path,required=True); p.add_argument("--image-id-file",type=Path,required=True); p.add_argument("--lambda-grid",required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); expected={int(x) for x in a.image_id_file.read_text().splitlines() if x.strip()}; grid={round(float(x),8) for x in a.lambda_grid.split(",")}; seen=set(); rows=[]
    for shard in a.shards:
        for number,line in enumerate(shard.read_text().splitlines(),1):
            try: row=json.loads(line); key=(int(row["image_id"]),round(float(row["base_lambda"]),8))
            except Exception as error: raise ValueError(f"Malformed {shard}:{number}: {error}")
            if key in seen: raise ValueError(f"Duplicate image/lambda entry {key}")
            if key[0] not in expected or key[1] not in grid: raise ValueError(f"Unexpected entry {key}")
            seen.add(key); rows.append(row)
    missing=[(image,lam) for image in sorted(expected) for lam in sorted(grid) if (image,lam) not in seen]
    if missing: raise ValueError(f"Missing {len(missing)} image/lambda entries; first={missing[:5]}")
    rows.sort(key=lambda row:(int(row["image_id"]),float(row["base_lambda"])))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("".join(json.dumps(row)+"\n" for row in rows)); print(f"merged={len(rows)} images={len(expected)} lambdas={len(grid)}")
if __name__=="__main__": main()
