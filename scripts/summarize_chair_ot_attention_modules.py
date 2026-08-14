#!/usr/bin/env python3
"""Summarize paired coverage/adaptive-alpha attention-OT ablations."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def metrics(path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)["overall_metrics"]
    return {name: float(values[name]) for name in METRICS}


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    baselines, rows = {}, []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            seed = int(entry["seed"])
            if entry["method"] == "vista":
                baselines[seed] = metrics(Path(entry["chair_json"]))
                continue
            baseline = baselines.get(seed)
            if baseline is None:
                raise ValueError(f"Missing VISTA baseline for seed={seed}")
            values = metrics(Path(entry["chair_json"]))
            rows.append({
                "method": entry["method"],
                "setting": entry["setting"],
                "seed": seed,
                **values,
                **{f"delta_{name}": values[name] - baseline[name] for name in METRICS},
            })
    if not rows:
        raise ValueError("Manifest contains no OT ablations")

    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["setting"])].append(row)
    for (method, setting), group in grouped.items():
        aggregate = {"method": method, "setting": setting, "seeds": len(group)}
        for name in METRICS:
            for prefix in ("", "delta_"):
                mean, std = mean_std([row[f"{prefix}{name}"] for row in group])
                aggregate[f"{prefix}{name}_mean"] = mean
                aggregate[f"{prefix}{name}_std"] = std
        summary.append(aggregate)
    summary.sort(key=lambda row: (row["method"], float(row["setting"]) if row["setting"] != "base" else -1))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Paired attention-OT module ablation",
        "",
        "Each row is paired against original VISTA with the same seed and fixed `logits_alpha=0.3`.",
        "",
        "| Method | Setting | Seeds | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta Precision | Delta F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['setting']} | {row['seeds']} | "
            f"{row['delta_CHAIRs_mean']:+.4f} +/- {row['delta_CHAIRs_std']:.4f} | "
            f"{row['delta_CHAIRi_mean']:+.4f} +/- {row['delta_CHAIRi_std']:.4f} | "
            f"{row['delta_Recall_mean']:+.4f} +/- {row['delta_Recall_std']:.4f} | "
            f"{row['delta_Precision_mean']:+.4f} +/- {row['delta_Precision_std']:.4f} | "
            f"{row['delta_F1_mean']:+.4f} +/- {row['delta_F1_std']:.4f} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
