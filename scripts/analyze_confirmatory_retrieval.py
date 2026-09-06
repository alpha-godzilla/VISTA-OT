#!/usr/bin/env python3
"""Pre-registered cluster-bootstrap analysis; no head/layer selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METRICS = ("query_content_effect", "m_V", "compat_GV", "mass_ratio_log", "m_G", "new_K_effect", "query_position_effect")
PRIMARY = {"query_content_effect": "negative", "m_V": "positive"}

def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))

def aggregate_pairs(pair_rows, events, windows, cohort):
    ids, relative, valid = windows["event_ids"].astype(int), windows["relative_t"].astype(int), windows["valid"].astype(bool)
    index = {item: i for i, item in enumerate(ids)}
    event = {int(row["event_id"]): row for row in events}
    metric = {name: windows[name].astype(np.float32).mean(axis=(2, 3)) for name in METRICS}
    output = []
    for pair in pair_rows:
        h, g = int(pair["hall_event_id"]), int(pair["ground_event_id"])
        cluster = pair["image_id"] if cohort == "same_image" else pair["hall_image_id"]
        for ri, rel in enumerate(relative):
            if rel < -5 or rel > 0 or not (valid[index[h], ri] and valid[index[g], ri]): continue
            for name in METRICS:
                output.append({"cohort": cohort, "matched_pair_id": pair["matched_pair_id"], "cluster": cluster, "relative_t": int(rel),
                               "metric": name, "difference": float(metric[name][index[h], ri] - metric[name][index[g], ri])})
    return output

def bootstrap_cluster(values, clusters, rng, reps):
    by_cluster = {}
    for value, cluster in zip(values, clusters): by_cluster.setdefault(cluster, []).append(value)
    cluster_values = np.array([np.mean(value) for value in by_cluster.values()], dtype=float)
    if not len(cluster_values): return (np.nan,) * 5
    draws = np.array([cluster_values[rng.integers(len(cluster_values), size=len(cluster_values))].mean() for _ in range(reps)])
    return float(cluster_values.mean()), float(np.median(cluster_values)), float(np.quantile(draws,.025)), float(np.quantile(draws,.975)), float(cluster_values.mean() / max(cluster_values.std(ddof=1), 1e-12)) if len(cluster_values)>1 else np.nan

def main():
    p = argparse.ArgumentParser(); p.add_argument("--events", type=Path, required=True); p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--same-image-pairs", type=Path, required=True); p.add_argument("--same-category-pairs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--bootstrap", type=int, default=5000); p.add_argument("--seed", type=int, default=20260907)
    args=p.parse_args(); event_rows=rows(args.events)
    with np.load(args.windows, allow_pickle=False) as data:
        raw={key: data[key] for key in data.files}
    effects=aggregate_pairs(rows(args.same_image_pairs), event_rows, raw, "same_image") + aggregate_pairs(rows(args.same_category_pairs), event_rows, raw, "same_category")
    rng=np.random.default_rng(args.seed); summary=[]
    for cohort in ("same_image", "same_category"):
        for rel in range(-5,1):
            for name in METRICS:
                selection=[row for row in effects if row["cohort"]==cohort and row["relative_t"]==rel and row["metric"]==name]
                mean,median,lo,hi,effect=bootstrap_cluster([row["difference"] for row in selection],[row["cluster"] for row in selection],rng,args.bootstrap)
                summary.append({"cohort":cohort,"relative_t":rel,"metric":name,"pair_count":len(selection),"image_clusters":len({row["cluster"] for row in selection}),"mean_difference":mean,"median_difference":median,"ci_low":lo,"ci_high":hi,"standardized_effect":effect,"role":"primary" if rel==0 and name in PRIMARY else ("secondary" if rel in (-1,0) else "negative_temporal_control")})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fields=list(summary[0])
    with (args.output_dir/"confirmatory_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(summary)
    primary={row["metric"]:row for row in summary if row["cohort"]=="same_image" and row["relative_t"]==0 and row["metric"] in PRIMARY}
    success=(primary.get("query_content_effect",{}).get("ci_high",np.inf)<0 and primary.get("m_V",{}).get("ci_low",-np.inf)>0)
    decision="strong_replication" if success else "inconclusive_or_no_replication"
    (args.output_dir/"confirmatory_decision.json").write_text(json.dumps({"primary_cohort":"same_image tolerance=5", "primary_endpoints":primary, "success":success, "decision":decision, "note":"Association only; layer/head are averaged within event before all inference."},indent=2))
    print(args.output_dir/"confirmatory_decision.json")

if __name__=="__main__": main()
