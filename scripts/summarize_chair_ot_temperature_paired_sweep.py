#!/usr/bin/env python3
"""Summarize paired OT-temperature CHAIR results and layer-weight dispersion."""
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
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--weight-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--ot-topk", type=int, required=True)
    parser.add_argument("--ot-visual-tokens", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--vsv-lambda", type=float, required=True)
    return parser.parse_args()


def read_metrics(path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)["overall_metrics"]
    return {name: float(values[name]) for name in METRICS}


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def read_weight_stats(path):
    vectors = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                vector = np.asarray(json.loads(line)["mean_layer_weights"], dtype=float)
                if vector.ndim != 1:
                    raise ValueError(f"Invalid mean_layer_weights in {path}")
                vectors.append(vector)
    if not vectors:
        raise ValueError(f"No OT diagnostics in {path}")
    matrix = np.stack(vectors)
    # Per-image population variance across the selected layers.  This measures
    # how non-uniform w_l is, independently of its six-layer mean (=1/6).
    return np.var(matrix, axis=1), matrix.mean(axis=0), len(vectors)


def main():
    args = parse_args()
    baselines, ot_rows, weight_rows = {}, [], []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            seed = int(entry["seed"])
            if entry["method"] == "vista":
                baselines[seed] = read_metrics(Path(entry["chair_json"]))
                continue
            temperature = float(entry["temperature"])
            metrics = read_metrics(Path(entry["chair_json"]))
            variances, mean_weights, image_count = read_weight_stats(Path(entry["stats_jsonl"]))
            row = {"temperature": temperature, "seed": seed, **metrics}
            row.update({f"delta_{name}": metrics[name] - baselines[seed][name] for name in METRICS})
            ot_rows.append(row)
            variance_mean, variance_std = mean_std(variances)
            weight_rows.append({
                "temperature": temperature, "seed": seed, "images": image_count,
                "wl_variance_mean": variance_mean, "wl_variance_std": variance_std,
                "mean_layer_weights": json.dumps(mean_weights.tolist()),
            })
    missing = {row["seed"] for row in ot_rows} - set(baselines)
    if missing:
        raise ValueError(f"Missing VISTA baselines for seeds: {sorted(missing)}")

    summary = []
    for temperature in sorted({row["temperature"] for row in ot_rows}):
        group = [row for row in ot_rows if row["temperature"] == temperature]
        weights = [row for row in weight_rows if row["temperature"] == temperature]
        aggregate = {"temperature": temperature, "seeds": len(group)}
        for name in METRICS:
            for prefix in ("", "delta_"):
                mean, std = mean_std([row[f"{prefix}{name}"] for row in group])
                aggregate[f"{prefix}{name}_mean"] = mean
                aggregate[f"{prefix}{name}_std"] = std
        for name in ("wl_variance_mean", "wl_variance_std"):
            mean, std = mean_std([row[name] for row in weights])
            aggregate[f"{name}_across_seeds_mean"] = mean
            aggregate[f"{name}_across_seeds_std"] = std
        aggregate["mean_layer_weights"] = json.dumps(np.mean([json.loads(row["mean_layer_weights"]) for row in weights], axis=0).tolist())
        summary.append(aggregate)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with args.weight_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weight_rows[0])); writer.writeheader(); writer.writerows(weight_rows)
    lines = ["# Paired VISTA–OT Layer-temperature Sweep", "", f"Fixed VSV lambda={args.vsv_lambda:g}, gamma={args.gamma:g}; OT topk={args.ot_topk}, visual_tokens={args.ot_visual_tokens}.", "`wl_variance` is the population variance across the six per-image mean layer weights; higher values mean a sharper, less-uniform layer selection.", "", "| Temperature | Seeds | Delta F1 | Delta CHAIRs | Mean wl variance | Mean layer weights |", "|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        lines.append(f"| {row['temperature']:g} | {row['seeds']} | {row['delta_F1_mean']:+.4f} +/- {row['delta_F1_std']:.4f} | {row['delta_CHAIRs_mean']:+.4f} +/- {row['delta_CHAIRs_std']:.4f} | {row['wl_variance_mean_across_seeds_mean']:.6f} | {row['mean_layer_weights']} |")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.weight_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
