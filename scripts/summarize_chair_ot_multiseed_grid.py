#!/usr/bin/env python3
"""Summarize a paired multi-seed VISTA/OT top-k and visual-token grid."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--per-seed-csv", required=True, type=Path)
    parser.add_argument("--aggregate-csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--vsv-lambda", required=True, type=float)
    return parser.parse_args()


def load_metrics(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)["overall_metrics"]
    return {name: float(payload[name]) for name in METRICS}


def load_fixed_ids(path):
    with path.open(encoding="utf-8") as handle:
        return [
            int(line.strip())
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def load_result_ids(path):
    with path.open(encoding="utf-8") as handle:
        return [int(json.loads(line)["image_id"]) for line in handle if line.strip()]


def mean_std(values):
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def read_rows(manifest):
    raw_rows = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            result_path = Path(entry["result_jsonl"])
            ids_path = Path(entry["ids_file"])
            expected_ids = load_fixed_ids(ids_path)
            actual_ids = load_result_ids(result_path)
            if actual_ids != expected_ids:
                raise ValueError(
                    f"Image-ID mismatch for {result_path}: expected the ordered "
                    f"IDs in {ids_path}"
                )
            row = {
                "method": entry["method"],
                "seed": int(entry["seed"]),
                "topk": int(entry["topk"]),
                "visual_tokens": int(entry["visual_tokens"]),
                "result_jsonl": str(result_path),
                "chair_json": entry["chair_json"],
            }
            row.update(load_metrics(Path(entry["chair_json"])))
            raw_rows.append(row)

    baselines = {
        row["seed"]: row for row in raw_rows if row["method"] == "vista"
    }
    seeds = {row["seed"] for row in raw_rows}
    if set(baselines) != seeds:
        missing = sorted(seeds - set(baselines))
        raise ValueError(f"Missing VISTA baseline for seeds: {missing}")

    for row in raw_rows:
        baseline = baselines[row["seed"]]
        for name in METRICS:
            row[f"delta_{name}"] = row[name] - baseline[name]
    return sorted(
        raw_rows,
        key=lambda row: (row["method"] != "vista", row["topk"], row["visual_tokens"], row["seed"]),
    )


def write_per_seed(path, rows):
    fields = (
        "method",
        "seed",
        "topk",
        "visual_tokens",
        *METRICS,
        *(f"delta_{name}" for name in METRICS),
        "result_jsonl",
        "chair_json",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["method"], row["topk"], row["visual_tokens"])
        grouped[key].append(row)

    aggregates = []
    for (method, topk, visual_tokens), group in grouped.items():
        aggregate = {
            "method": method,
            "topk": topk,
            "visual_tokens": visual_tokens,
            "seeds": len(group),
        }
        for name in METRICS:
            mean, std = mean_std([row[name] for row in group])
            delta_mean, delta_std = mean_std(
                [row[f"delta_{name}"] for row in group]
            )
            aggregate[f"{name}_mean"] = mean
            aggregate[f"{name}_std"] = std
            aggregate[f"delta_{name}_mean"] = delta_mean
            aggregate[f"delta_{name}_std"] = delta_std
        aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda row: (row["method"] != "vista", row["topk"], row["visual_tokens"]),
    )


def write_aggregate(path, rows):
    fields = ["method", "topk", "visual_tokens", "seeds"]
    for name in METRICS:
        fields.extend(
            (
                f"{name}_mean",
                f"{name}_std",
                f"delta_{name}_mean",
                f"delta_{name}_std",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, gamma, vsv_lambda):
    baseline = next(row for row in rows if row["method"] == "vista")
    ot_rows = [row for row in rows if row["method"] == "ot"]
    best_f1 = max(ot_rows, key=lambda row: row["F1_mean"])
    best_chairs = min(ot_rows, key=lambda row: row["CHAIRs_mean"])
    best_chairi = min(ot_rows, key=lambda row: row["CHAIRi_mean"])

    lines = [
        "# VISTA-OT Multi-seed Hyperparameter Grid",
        "",
        f"Fixed VSV lambda: {vsv_lambda:g}",
        f"Fixed SLA gamma (`--logits-alpha`): {gamma:g}",
        "Marginals: uniform visual tokens + average visual dustbin; "
        "top-k text-logit probabilities without a text dustbin.",
        "",
        "## Reference and best configurations",
        "",
        f"- VISTA baseline mean F1: {baseline['F1_mean']:.4f}",
        f"- Highest mean F1: topk={best_f1['topk']}, visual_tokens={best_f1['visual_tokens']}, "
        f"F1={best_f1['F1_mean']:.4f}, delta={best_f1['delta_F1_mean']:+.4f}",
        f"- Lowest mean CHAIRs: topk={best_chairs['topk']}, visual_tokens={best_chairs['visual_tokens']}, "
        f"CHAIRs={best_chairs['CHAIRs_mean']:.4f}, delta={best_chairs['delta_CHAIRs_mean']:+.4f}",
        f"- Lowest mean CHAIRi: topk={best_chairi['topk']}, visual_tokens={best_chairi['visual_tokens']}, "
        f"CHAIRi={best_chairi['CHAIRi_mean']:.4f}, delta={best_chairi['delta_CHAIRi_mean']:+.4f}",
        "",
        "## Aggregate results",
        "",
        "| Top-k | Visual tokens | CHAIRs | Delta CHAIRs | CHAIRi | Delta CHAIRi | F1 | Delta F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(ot_rows, key=lambda item: item["F1_mean"], reverse=True):
        lines.append(
            f"| {row['topk']} | {row['visual_tokens']} | {row['CHAIRs_mean']:.4f} +/- {row['CHAIRs_std']:.4f} | "
            f"{row['delta_CHAIRs_mean']:+.4f} | {row['CHAIRi_mean']:.4f} +/- {row['CHAIRi_std']:.4f} | "
            f"{row['delta_CHAIRi_mean']:+.4f} | {row['F1_mean']:.4f} +/- {row['F1_std']:.4f} | "
            f"{row['delta_F1_mean']:+.4f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_rows(args.manifest)
    aggregates = aggregate_rows(rows)
    write_per_seed(args.per_seed_csv, rows)
    write_aggregate(args.aggregate_csv, aggregates)
    write_markdown(args.markdown, aggregates, args.gamma, args.vsv_lambda)
    print(f"Wrote {args.per_seed_csv}")
    print(f"Wrote {args.aggregate_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
