#!/usr/bin/env python3
"""Offline, calibration/held-out law discovery from sweep and CHAIR outcomes."""
import argparse, csv, json
from pathlib import Path
import numpy as np

def rank(x):
    order=np.argsort(x,kind="mergesort"); out=np.empty(len(x),float); out[order]=np.arange(len(x)); return out
def corr(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return float("nan")
    return float(np.corrcoef(rank(x),rank(y))[0,1])
def main():
    p=argparse.ArgumentParser(); p.add_argument("--sweep",type=Path,required=True); p.add_argument("--outcomes",type=Path,required=True); p.add_argument("--calibration-ids",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); cal={int(x) for x in a.calibration_ids.read_text().splitlines() if x.strip()}; outcomes={}
    with a.outcomes.open() as f:
      for row in csv.DictReader(f): outcomes[(int(row["image_id"]),round(float(row["base_lambda"]),8))]=row
    rows=[json.loads(x) for x in a.sweep.read_text().splitlines() if x.strip()]; by={}
    for row in rows:
      key=(int(row["image_id"]),round(float(row["base_lambda"]),8)); outcome=outcomes.get(key)
      if outcome is None: continue
      by.setdefault(key[0],[]).append((float(row["base_lambda"]),row,outcome))
    targets={};
    for image,items in by.items():
      safe=[item for item in items if int(float(item[2].get("wrong_count",1)))==0]
      if safe: targets[image]=min(safe,key=lambda item:item[0])[0]
    features=("raw_diff_norm_mean","raw_diff_norm_cv","relative_raw_diff_norm_mean","angular_pos_neg_mean","official_raw_cos_mean","layer_adjacent_cos_mean","delta_logit_l2","js","cosine_logit_change")
    summary=[]
    for name in features:
      values=[]; target=[]; split=[]
      for image,items in by.items():
        if image not in targets: continue
        base=min(items,key=lambda item:abs(item[0]-.17))[1]; value=base.get("features",{}).get(name)
        if value is None: continue
        values.append(float(value)); target.append(targets[image]); split.append("calibration" if image in cal else "heldout")
      for part in ("calibration","heldout"):
        mask=np.array([x==part for x in split]); summary.append({"feature":name,"split":part,"images":int(mask.sum()),"spearman":corr(np.asarray(values)[mask],np.asarray(target)[mask]) if mask.any() else np.nan})
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"law_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=True));
    with (a.output_dir/"law_summary.csv").open("w",newline="") as f:
      writer=csv.DictWriter(f,fieldnames=("feature","split","images","spearman"));writer.writeheader();writer.writerows(summary)
if __name__=="__main__": main()
