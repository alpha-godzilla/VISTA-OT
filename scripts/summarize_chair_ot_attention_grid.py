#!/usr/bin/env python3
"""Summarize paired, unpooled layer-aligned attention-OT CHAIR sweeps."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")
CONFIG_FIELDS = ("gamma", "layer_temperature", "attention_power", "uniform_mix")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--weight-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)["overall_metrics"]
    return {name: float(values[name]) for name in METRICS}


def mean_std(values):
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    )


def read_weight_stats(path):
    vectors = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                vectors.append(np.asarray(json.loads(line)["mean_layer_weights"], dtype=float))
    if not vectors:
        raise ValueError(f"No OT diagnostics in {path}")
    matrix = np.stack(vectors)
    return np.var(matrix, axis=1), matrix.mean(axis=0), len(vectors)


def config_from_entry(entry):
    return tuple(float(entry[name]) for name in CONFIG_FIELDS)


def main():
    args = parse_args()
    baselines = {}
    rows = []
    weight_rows = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            seed = int(entry["seed"])
            gamma = float(entry["gamma"])
            if entry["method"] == "vista":
                baselines[(seed, gamma)] = read_metrics(Path(entry["chair_json"]))
                continue
            baseline = baselines.get((seed, gamma))
            if baseline is None:
                raise ValueError(f"Missing VISTA baseline for seed={seed}, gamma={gamma}")
            metrics = read_metrics(Path(entry["chair_json"]))
            config = config_from_entry(entry)
            row = dict(zip(CONFIG_FIELDS, config))
            row.update({"seed": seed, **metrics})
            row.update({f"delta_{name}": metrics[name] - baseline[name] for name in METRICS})
            rows.append(row)
            variances, mean_weights, image_count = read_weight_stats(Path(entry["stats_jsonl"]))
            variance_mean, variance_std = mean_std(variances)
            weight_rows.append({
                "seed": seed,
                **dict(zip(CONFIG_FIELDS, config)),
                "images": image_count,
                "wl_variance_mean": variance_mean,
                "wl_variance_std": variance_std,
                "mean_layer_weights": json.dumps(mean_weights.tolist()),
            })
    if not rows:
        raise ValueError("Manifest contains no attention-OT results")

    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in CONFIG_FIELDS)].append(row)
    for config in sorted(grouped):
        group = grouped[config]
        aggregate = dict(zip(CONFIG_FIELDS, config))
        aggregate["seeds"] = len(group)
        for name in METRICS:
            for prefix in ("", "delta_"):
                mean, std = mean_std([row[f"{prefix}{name}"] for row in group])
                aggregate[f"{prefix}{name}_mean"] = mean
                aggregate[f"{prefix}{name}_std"] = std
        weights = [
            row for row in weight_rows
            if tuple(row[name] for name in CONFIG_FIELDS) == config
        ]
        for name in ("wl_variance_mean", "wl_variance_std"):
            mean, std = mean_std([row[name] for row in weights])
            aggregate[f"{name}_across_seeds_mean"] = mean
            aggregate[f"{name}_across_seeds_std"] = std
        aggregate["mean_layer_weights"] = json.dumps(
            np.mean([json.loads(row["mean_layer_weights"]) for row in weights], axis=0).tolist()
        )
        summary.append(aggregate)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with args.weight_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)

    lines = [
        "# Paired Unpooled Layer-Aligned Attention-OT Sweep",
        "",
        "| gamma | Layer temp. | Attention power | Uniform mix | Seeds | Delta F1 | Delta CHAIRs | Delta CHAIRi |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['gamma']:g} | {row['layer_temperature']:g} | "
            f"{row['attention_power']:g} | {row['uniform_mix']:g} | "
            f"{row['seeds']} | {row['delta_F1_mean']:+.4f} +/- {row['delta_F1_std']:.4f} | "
            f"{row['delta_CHAIRs_mean']:+.4f} +/- {row['delta_CHAIRs_std']:.4f} | "
            f"{row['delta_CHAIRi_mean']:+.4f} +/- {row['delta_CHAIRi_std']:.4f} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.weight_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
