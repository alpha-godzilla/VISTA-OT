#!/usr/bin/env python3
"""Summarize mass-centered UOT without assuming equal alpha means equal strength."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")
DIAGNOSTICS = (
    "mean_uot_iterations",
    "mean_uot_dual_residual",
    "mean_uniform_uot_iterations",
    "mean_uniform_uot_dual_residual",
    "mean_candidate_promotion_gate",
    "mean_candidate_suppression_gate",
    "mean_local_transport_mass",
    "mean_uniform_transport_mass",
    "mean_attention_retention_abs_deviation",
    "mean_uniform_retention_abs_deviation",
    "mean_attention_uniform_retention_gap",
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
    return {
        name: float(np.mean(values[name])) if values[name] else None
        for name in DIAGNOSTICS
    }


def is_pareto(row, rows, dimensions):
    """Return whether row is nondominated for min/max metric dimensions."""
    for other in rows:
        no_worse = all(
            other[name] <= row[name] if direction == "min"
            else other[name] >= row[name]
            for name, direction in dimensions
        )
        strictly_better = any(
            other[name] < row[name] if direction == "min"
            else other[name] > row[name]
            for name, direction in dimensions
        )
        if no_worse and strictly_better:
            return False
    return True


def nearest_f1_vista(row, vista_rows):
    return min(
        vista_rows,
        key=lambda vista: (
            abs(vista["F1"] - row["F1"]),
            -vista["F1"],
            vista["logits_alpha"],
        ),
    )


def show(value, digits=4, signed=False):
    if value is None or value == "":
        return "n/a"
    format_spec = f"{'+' if signed else ''}.{digits}f"
    return format(float(value), format_spec)


def main():
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle, delimiter="\t"))
    if not entries:
        raise ValueError("Manifest is empty")

    rows = []
    for entry in entries:
        stats_path = (
            None if entry["stats_jsonl"] in {"", "na"}
            else Path(entry["stats_jsonl"])
        )
        rows.append({
            "method": entry["method"],
            "setting": entry["setting"],
            "seed": int(entry["seed"]),
            "logits_alpha": float(entry["logits_alpha"]),
            "marginal_relaxation": (
                None if entry["marginal_relaxation"] in {"", "na"}
                else float(entry["marginal_relaxation"])
            ),
            "gate_mode": entry["gate_mode"],
            **read_metrics(Path(entry["chair_json"])),
            **read_diagnostics(stats_path),
        })

    vista_rows = [row for row in rows if row["method"] == "vista"]
    if not vista_rows:
        raise ValueError("At least one evaluated VISTA row is required")
    raw_by_key = {
        (row["logits_alpha"], row["marginal_relaxation"]): row
        for row in rows if row["gate_mode"] == "raw"
    }
    for row in rows:
        vista = nearest_f1_vista(row, vista_rows)
        row["nearest_vista_alpha"] = vista["logits_alpha"]
        row["nearest_vista_f1_gap"] = row["F1"] - vista["F1"]
        for metric in METRICS:
            row[f"delta_nearest_vista_{metric}"] = (
                row[metric] - vista[metric]
            )
        raw = raw_by_key.get(
            (row["logits_alpha"], row["marginal_relaxation"])
        ) if row["gate_mode"] == "centered" else None
        for metric in METRICS:
            row[f"delta_raw_{metric}"] = (
                row[metric] - raw[metric] if raw is not None else None
            )
        row["joint_nearest_vista_f1_002"] = bool(
            row["gate_mode"] == "centered"
            and row["CHAIRs"] <= vista["CHAIRs"]
            and row["CHAIRi"] <= vista["CHAIRi"]
            and row["F1"] >= vista["F1"] - 0.002
        )

    for row in rows:
        row["chair_f1_global_pareto"] = is_pareto(
            row, rows, (("CHAIRs", "min"), ("F1", "max")),
        )
        row["chairi_f1_global_pareto"] = is_pareto(
            row, rows, (("CHAIRi", "min"), ("F1", "max")),
        )
        row["three_metric_global_pareto"] = is_pareto(
            row, rows,
            (("CHAIRs", "min"), ("CHAIRi", "min"), ("F1", "max")),
        )

    rows.sort(key=lambda row: (
        row["method"], row["logits_alpha"],
        row["marginal_relaxation"]
        if row["marginal_relaxation"] is not None else -1,
    ))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    centered = [row for row in rows if row["gate_mode"] == "centered"]
    pareto = [row for row in rows if row["three_metric_global_pareto"]]
    pareto.sort(key=lambda row: (-row["F1"], row["CHAIRs"], row["CHAIRi"]))
    paired = [
        row for row in centered
        if row["delta_raw_F1"] is not None
    ]
    lines = [
        "# Mass-centered direction-aware UOT ablation", "",
        "This is a single development-seed sweep. Generation is label-free; CHAIR ground truth is read only by post-hoc evaluation. Because centering changes effective intervention strength, the main comparison uses the nearest-F1 VISTA point and a global Pareto frontier rather than assuming equal `logits_alpha` is matched strength.",
        "", "## Centered UOT grid", "",
        "| Alpha | Rho | CHAIRs | CHAIRi | Recall | Precision | F1 | Nearest VISTA alpha | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta F1 | Joint target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in centered:
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            f"{show(row['CHAIRs'])} | {show(row['CHAIRi'])} | "
            f"{show(row['Recall'])} | {show(row['Precision'])} | {show(row['F1'])} | "
            f"{row['nearest_vista_alpha']:g} | "
            f"{show(row['delta_nearest_vista_CHAIRs'], signed=True)} | "
            f"{show(row['delta_nearest_vista_CHAIRi'], signed=True)} | "
            f"{show(row['delta_nearest_vista_Recall'], signed=True)} | "
            f"{show(row['delta_nearest_vista_F1'], signed=True)} | "
            f"{'yes' if row['joint_nearest_vista_f1_002'] else 'no'} |"
        )

    lines.extend([
        "", "## Global CHAIRs/CHAIRi/F1 Pareto frontier", "",
        "| Method | Setting | CHAIRs | CHAIRi | Recall | Precision | F1 |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ])
    for row in pareto:
        lines.append(
            f"| {row['method']} | {row['setting']} | {show(row['CHAIRs'])} | "
            f"{show(row['CHAIRi'])} | {show(row['Recall'])} | "
            f"{show(row['Precision'])} | {show(row['F1'])} |"
        )

    lines.extend([
        "", "## Centered versus raw at identical alpha and rho", "",
        "| Alpha | Rho | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta Precision | Delta F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if paired:
        for row in paired:
            lines.append(
                f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
                f"{show(row['delta_raw_CHAIRs'], signed=True)} | "
                f"{show(row['delta_raw_CHAIRi'], signed=True)} | "
                f"{show(row['delta_raw_Recall'], signed=True)} | "
                f"{show(row['delta_raw_Precision'], signed=True)} | "
                f"{show(row['delta_raw_F1'], signed=True)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend([
        "", "## Centered-gate diagnostics", "",
        "| Alpha | Rho | Promotion | Suppression | Attention deviation | Uniform deviation | Attention-uniform gap | UOT iters | Residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in centered:
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            f"{show(row['mean_candidate_promotion_gate'], 6)} | "
            f"{show(row['mean_candidate_suppression_gate'], 6)} | "
            f"{show(row['mean_attention_retention_abs_deviation'], 6)} | "
            f"{show(row['mean_uniform_retention_abs_deviation'], 6)} | "
            f"{show(row['mean_attention_uniform_retention_gap'], 6)} | "
            f"{show(row['mean_uot_iterations'], 3)} | "
            f"{show(row['mean_uot_dual_residual'], 6)} |"
        )

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
