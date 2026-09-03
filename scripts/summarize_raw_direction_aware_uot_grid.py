#!/usr/bin/env python3
"""Summarize the development-only raw direction-aware UOT grid."""

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
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)["overall_metrics"]
    return {name: float(values[name]) for name in METRICS}


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


def nearest_f1(row, references):
    return min(references, key=lambda ref: (
        abs(ref["F1"] - row["F1"]), -ref["F1"], ref["logits_alpha"],
    ))


def pareto(row, rows):
    """Nondominated under lower CHAIRs/CHAIRi and higher F1."""
    for other in rows:
        no_worse = (
            other["CHAIRs"] <= row["CHAIRs"]
            and other["CHAIRi"] <= row["CHAIRi"]
            and other["F1"] >= row["F1"]
        )
        strictly_better = (
            other["CHAIRs"] < row["CHAIRs"]
            or other["CHAIRi"] < row["CHAIRi"]
            or other["F1"] > row["F1"]
        )
        if no_worse and strictly_better:
            return False
    return True


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
    vista = [row for row in rows if row["method"] == "vista"]
    ot_rows = [row for row in rows if row["method"] == "raw_direction_aware_uot"]
    if not vista or not ot_rows:
        raise ValueError("Manifest needs both VISTA and raw direction-aware UOT rows")
    for row in rows:
        reference = nearest_f1(row, vista)
        row["nearest_vista_alpha"] = reference["logits_alpha"]
        for metric in METRICS:
            row[f"delta_nearest_vista_{metric}"] = row[metric] - reference[metric]
    for row in ot_rows:
        row["three_metric_pareto"] = pareto(row, ot_rows)
        row["joint_target"] = bool(
            row["delta_nearest_vista_CHAIRs"] <= 0
            and row["delta_nearest_vista_CHAIRi"] <= 0
            and row["delta_nearest_vista_F1"] >= -0.002
        )
    for row in vista:
        row["three_metric_pareto"] = False
        row["joint_target"] = False
    rows.sort(key=lambda row: (row["method"], row["logits_alpha"], row["marginal_relaxation"] or -1))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Raw direction-aware UOT development grid", "",
        "This grid is label-free during generation. CHAIR labels are used only for post-hoc development evaluation on seed 1994; select a final configuration only after validating it on held-out seeds.",
        "", "## Raw direction-aware UOT", "",
        "| Alpha | Rho | CHAIRs | CHAIRi | Recall | Precision | F1 | Len | Nearest-F1 VISTA alpha | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta F1 | Pareto | Joint target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in ot_rows:
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            f"{show(row['CHAIRs'])} | {show(row['CHAIRi'])} | {show(row['Recall'])} | "
            f"{show(row['Precision'])} | {show(row['F1'])} | {show(row['Len'])} | "
            f"{row['nearest_vista_alpha']:g} | "
            f"{show(row['delta_nearest_vista_CHAIRs'], signed=True)} | "
            f"{show(row['delta_nearest_vista_CHAIRi'], signed=True)} | "
            f"{show(row['delta_nearest_vista_Recall'], signed=True)} | "
            f"{show(row['delta_nearest_vista_F1'], signed=True)} | "
            f"{'yes' if row['three_metric_pareto'] else 'no'} | "
            f"{'yes' if row['joint_target'] else 'no'} |"
        )
    frontier = [row for row in ot_rows if row["three_metric_pareto"]]
    lines.extend([
        "", "## Three-metric Pareto frontier", "",
        "| Alpha | Rho | CHAIRs | CHAIRi | Recall | Precision | F1 | Len |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in frontier:
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            + " | ".join(show(row[name]) for name in METRICS) + " |"
        )
    lines.extend([
        "", "## Solver diagnostics", "",
        "| Alpha | Rho | Attention iters | Attention residual | Uniform iters | Uniform residual | Promotion gate | Suppression gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ot_rows:
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            + " | ".join(show(row[name], 6) for name in DIAGNOSTICS) + " |"
        )
    lines.extend([
        "", "For alpha values without a same-alpha VISTA run, `nearest-F1 VISTA` is a calibration reference only; it is not an equal-intervention-strength causal control.",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
