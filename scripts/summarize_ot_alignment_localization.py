#!/usr/bin/env python3
"""Summarize a one-factor-at-a-time OT alignment localization ablation."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")
DIAGNOSTICS = (
    "mean_uot_iterations", "mean_uot_dual_residual",
    "mean_uniform_uot_iterations", "mean_uniform_uot_dual_residual",
    "mean_candidate_promotion_gate", "mean_candidate_suppression_gate",
    "mean_local_transport_mass", "mean_uniform_transport_mass",
    "mean_attention_retention_abs_deviation",
    "mean_uniform_retention_abs_deviation",
    "mean_attention_uniform_retention_gap",
    "mean_timestep_promotion_strength", "mean_timestep_suppression_strength",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path):
    with path.open(encoding="utf-8") as handle:
        overall = json.load(handle)["overall_metrics"]
    return {name: float(overall[name]) for name in METRICS}


def read_diagnostics(path):
    values = defaultdict(list)
    if path is not None and path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for name in DIAGNOSTICS:
                    if name in record:
                        values[name].append(float(record[name]))
    return {name: float(np.mean(values[name])) if values[name] else None for name in DIAGNOSTICS}


def show(value, digits=4, signed=False):
    if value is None or value == "":
        return "n/a"
    return format(float(value), f"{'+' if signed else ''}.{digits}f")


def main():
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))
    if not entries:
        raise ValueError("Manifest is empty")

    rows = []
    for entry in entries:
        stats = None if entry["stats_jsonl"] in {"", "na"} else Path(entry["stats_jsonl"])
        rows.append({
            "method": entry["method"], "setting": entry["setting"],
            "seed": int(entry["seed"]), "logits_alpha": float(entry["logits_alpha"]),
            "marginal_relaxation": None if entry["marginal_relaxation"] == "na" else float(entry["marginal_relaxation"]),
            **read_metrics(Path(entry["chair_json"])), **read_diagnostics(stats),
        })

    raw = next((row for row in rows if row["method"] == "raw_direction_aware"), None)
    vista = next((row for row in rows if row["method"] == "vista"), None)
    if raw is None or vista is None:
        raise ValueError("Both raw_direction_aware and vista rows are required")
    for row in rows:
        for metric in METRICS:
            row[f"delta_raw_{metric}"] = row[metric] - raw[metric]
            row[f"delta_vista_{metric}"] = row[metric] - vista[metric]

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OT alignment localization ablation", "",
        "All settings use the same seed, image-ID manifest, alpha, rho, decoding, and OT hyperparameters. Each successive OT row adds exactly one alignment component to the preceding row; this makes deltas relative to raw direction-aware UOT causal within this one-seed development experiment.",
        "", "## Absolute results", "",
        "| Method | CHAIRs | CHAIRi | Recall | Precision | F1 | Len |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | " + " | ".join(show(row[m]) for m in METRICS) + " |"
        )
    lines.extend([
        "", "## Delta relative to raw direction-aware UOT", "",
        "| Method | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta Precision | Delta F1 | Delta Len |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['method']} | " + " | ".join(
                show(row[f"delta_raw_{metric}"], signed=True) for metric in METRICS
            ) + " |"
        )
    lines.extend([
        "", "## Diagnostics", "",
        "| Method | Promote | Suppress | Local mass | Uniform mass | Attention deviation | Uniform deviation | q+ | q- | UOT iters | Residual |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['method']} | " + " | ".join(show(row[name], 6) for name in (
                "mean_candidate_promotion_gate", "mean_candidate_suppression_gate",
                "mean_local_transport_mass", "mean_uniform_transport_mass",
                "mean_attention_retention_abs_deviation", "mean_uniform_retention_abs_deviation",
                "mean_timestep_promotion_strength", "mean_timestep_suppression_strength",
                "mean_uot_iterations", "mean_uot_dual_residual",
            )) + " |"
        )
    lines.extend([
        "", "Interpret component effects by comparing adjacent rows, not only against VISTA: `raw → shared → final_norm → mass_centered → timestep_gate`. A degradation appearing at one transition identifies the first harmful component combination under this fixed configuration.",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
