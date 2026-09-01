#!/usr/bin/env python3
"""Summarize the single-development-seed direction-aware UOT refinement."""

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
    collected = defaultdict(list)
    if not path or not path.exists():
        return {name: None for name in DIAGNOSTICS}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for name in DIAGNOSTICS:
                if name in record:
                    collected[name].append(float(record[name]))
    return {
        name: float(np.mean(collected[name])) if collected[name] else None
        for name in DIAGNOSTICS
    }


def is_pareto(row, candidates):
    """Minimize CHAIRs and maximize F1 within one method family."""
    for other in candidates:
        no_worse = (
            other["CHAIRs"] <= row["CHAIRs"]
            and other["F1"] >= row["F1"]
        )
        strictly_better = (
            other["CHAIRs"] < row["CHAIRs"]
            or other["F1"] > row["F1"]
        )
        if no_worse and strictly_better:
            return False
    return True


def main():
    args = parse_args()
    entries = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            entries.append(entry)
    if not entries:
        raise ValueError("Manifest is empty")

    baselines = {}
    for entry in entries:
        if entry["method"] != "vista":
            continue
        key = (int(entry["seed"]), float(entry["logits_alpha"]))
        if key in baselines:
            raise ValueError(f"Duplicate VISTA baseline for {key}")
        baselines[key] = read_metrics(Path(entry["chair_json"]))

    rows = []
    for entry in entries:
        seed = int(entry["seed"])
        alpha = float(entry["logits_alpha"])
        key = (seed, alpha)
        if key not in baselines:
            raise ValueError(f"Missing same-alpha VISTA baseline for {key}")
        values = read_metrics(Path(entry["chair_json"]))
        baseline = baselines[key]
        diagnostics_path = Path(entry["stats_jsonl"]) if entry["stats_jsonl"] else None
        rows.append({
            "method": entry["method"],
            "setting": entry["setting"],
            "seed": seed,
            "logits_alpha": alpha,
            "marginal_relaxation": (
                float(entry["marginal_relaxation"])
                if entry["marginal_relaxation"] else None
            ),
            "layer_weight_reference": entry["layer_weight_reference"],
            **values,
            **{
                f"delta_{name}": values[name] - baseline[name]
                for name in METRICS
            },
            **read_diagnostics(diagnostics_path),
        })

    candidates = [
        row for row in rows if row["method"] == "direction_aware_uot"
        and row["layer_weight_reference"] == "independent"
    ]
    for row in rows:
        row["chair_f1_pareto"] = (
            is_pareto(row, candidates) if row in candidates else False
        )
        row["joint_target"] = bool(
            row in candidates
            and row["delta_CHAIRs"] <= 0
            and row["delta_CHAIRi"] <= 0
            and row["delta_F1"] >= -0.002
        )

    rows.sort(key=lambda row: (
        row["method"], row["logits_alpha"],
        row["marginal_relaxation"] if row["marginal_relaxation"] is not None else -1,
        row["layer_weight_reference"],
    ))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Direction-aware UOT refinement on development seed", "",
        "All deltas use VISTA with the same `logits_alpha`. The seed is a development seed; selecting a configuration from this table makes the final method development-set tuned.",
        "", "## Independent-uniform direction-aware UOT", "",
        "| Alpha | Rho | CHAIRs | CHAIRi | Recall | Precision | F1 | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta F1 | Pareto | Joint target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            f"{row['CHAIRs']:.4f} | {row['CHAIRi']:.4f} | "
            f"{row['Recall']:.4f} | {row['Precision']:.4f} | {row['F1']:.4f} | "
            f"{row['delta_CHAIRs']:+.4f} | {row['delta_CHAIRi']:+.4f} | "
            f"{row['delta_Recall']:+.4f} | {row['delta_F1']:+.4f} | "
            f"{'yes' if row['chair_f1_pareto'] else 'no'} | "
            f"{'yes' if row['joint_target'] else 'no'} |"
        )
    shared = [
        row for row in rows if row["method"] == "direction_aware_uot"
        and row["layer_weight_reference"] == "shared"
    ]
    if shared:
        lines.extend([
            "", "## Shared-layer-weight compatibility ablation", "",
            "| Alpha | Rho | CHAIRs | CHAIRi | Recall | Precision | F1 | Delta CHAIRs | Delta F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in shared:
            lines.append(
                f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
                f"{row['CHAIRs']:.4f} | {row['CHAIRi']:.4f} | "
                f"{row['Recall']:.4f} | {row['Precision']:.4f} | {row['F1']:.4f} | "
                f"{row['delta_CHAIRs']:+.4f} | {row['delta_F1']:+.4f} |"
            )
    lines.extend([
        "", "## Solver diagnostics", "",
        "| Alpha | Rho | Attention iters | Attention residual | Uniform iters | Uniform residual | Promotion gate | Suppression gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in candidates:
        def show(name):
            value = row[name]
            return "n/a" if value is None else f"{value:.6g}"
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            f"{show('mean_uot_iterations')} | {show('mean_uot_dual_residual')} | "
            f"{show('mean_uniform_uot_iterations')} | {show('mean_uniform_uot_dual_residual')} | "
            f"{show('mean_candidate_promotion_gate')} | {show('mean_candidate_suppression_gate')} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
