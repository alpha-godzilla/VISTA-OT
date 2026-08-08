#!/usr/bin/env python3
"""Summarize paired VISTA and VISTA-OT CHAIR results across random seeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    return parser.parse_args()


def read_metrics(path):
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)["overall_metrics"]
    return {name: float(metrics[name]) for name in METRICS}


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    rows = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            vista_metrics = read_metrics(Path(entry["vista_chair_json"]))
            ot_metrics = read_metrics(Path(entry["ot_chair_json"]))
            row = {"seed": int(entry["seed"])}
            for method, metrics in (("vista", vista_metrics), ("ot", ot_metrics)):
                for name in METRICS:
                    row[f"{method}_{name}"] = metrics[name]
            for name in METRICS:
                row[f"delta_{name}"] = ot_metrics[name] - vista_metrics[name]
            rows.append(row)
    rows.sort(key=lambda row: row["seed"])

    fields = ["seed"]
    for method in ("vista", "ot"):
        fields.extend(f"{method}_{name}" for name in METRICS)
    fields.extend(f"delta_{name}" for name in METRICS)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def metric_summary(prefix, name):
        return mean_std([row[f"{prefix}_{name}"] for row in rows])

    lines = [
        "# VISTA vs VISTA-OT Seed Comparison",
        "",
        "VISTA-OT: topk=32, visual_tokens=81, Sinkhorn iterations=3, epsilon=0.05.",
        "Fixed VSV lambda=0.17 and SLA gamma (`--logits-alpha`)=0.3.",
        "Each row is paired: VISTA and VISTA-OT share the same fixed val2014 image IDs.",
        "",
        "## Per Seed",
        "",
        "| Seed | VISTA F1 | OT F1 | OT-VISTA F1 | VISTA CHAIRs | OT CHAIRs | OT-VISTA CHAIRs |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['vista_F1']:.4f} | {row['ot_F1']:.4f} | "
            f"{row['delta_F1']:+.4f} | {row['vista_CHAIRs']:.4f} | "
            f"{row['ot_CHAIRs']:.4f} | {row['delta_CHAIRs']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mean +/- Sample Standard Deviation",
            "",
            "| Method | CHAIRs | CHAIRi | Recall | Precision | F1 | Len |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("vista", "ot"):
        values = [
            f"{metric_summary(method, name)[0]:.4f} +/- {metric_summary(method, name)[1]:.4f}"
            for name in METRICS
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")

    delta_values = [
        mean_std([row[f"delta_{name}"] for row in rows]) for name in METRICS
    ]
    lines.extend(
        [
            "",
            "OT minus VISTA mean delta:",
            "",
            "| CHAIRs | CHAIRi | Recall | Precision | F1 | Len |",
            "|---:|---:|---:|---:|---:|---:|",
            "| " + " | ".join(
                f"{mean:+.4f} +/- {std:.4f}" for mean, std in delta_values
            ) + " |",
            "",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
