#!/usr/bin/env python3
"""Summarize label-free dynamic head-aware UOT development runs."""

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
    "mean_head_effective_count", "mean_head_visual_mass",
    "mean_head_max_weight",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def metrics(path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)["overall_metrics"]
    return {name: float(values[name]) for name in METRICS}


def diagnostics(path):
    values = defaultdict(list)
    if path is not None and path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
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
    rows = []
    for entry in entries:
        stats = None if entry["stats_jsonl"] == "na" else Path(entry["stats_jsonl"])
        rows.append({
            "method": entry["method"], "setting": entry["setting"],
            "seed": int(entry["seed"]), "logits_alpha": float(entry["logits_alpha"]),
            "marginal_relaxation": None if entry["marginal_relaxation"] == "na" else float(entry["marginal_relaxation"]),
            "head_temperature": entry["head_temperature"],
            "head_uniform_mix": entry["head_uniform_mix"],
            "head_topk": entry["head_topk"],
            "head_mass_weight": entry["head_mass_weight"],
            **metrics(Path(entry["chair_json"])), **diagnostics(stats),
        })
    vista = {row["logits_alpha"]: row for row in rows if row["method"] == "vista"}
    raw = {
        (row["logits_alpha"], row["marginal_relaxation"]): row
        for row in rows if row["method"] == "raw_direction_aware_uot"
    }
    if not vista or not raw:
        raise ValueError("Manifest requires reused VISTA and raw-UOT reference rows")
    for row in rows:
        reference = raw.get((row["logits_alpha"], row["marginal_relaxation"]))
        vista_reference = vista.get(row["logits_alpha"])
        for metric in METRICS:
            row[f"delta_raw_{metric}"] = (
                row[metric] - reference[metric] if reference else None
            )
            row[f"delta_same_alpha_vista_{metric}"] = (
                row[metric] - vista_reference[metric]
                if vista_reference else None
            )
    rows.sort(key=lambda row: (row["method"], row["logits_alpha"], row["marginal_relaxation"] or -1, row["head_temperature"]))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    candidates = [row for row in rows if row["method"] in {"head_mass", "head_uot", "head_uot_uniform"}]
    lines = [
        "# Head-aware raw direction-aware UOT development grid", "",
        "Generation is label-free. The reused raw-UOT/VISTA rows and all head-aware rows share the same fixed 500-image seed-1994 manifest. Head IDs are never selected with CHAIR labels; only post-hoc model selection uses this development table.",
        "", "## Head-aware candidates relative to identical raw-UOT", "",
        "| Method | Alpha | Rho | Head temperature | Head uniform mix | CHAIRs | CHAIRi | Recall | Precision | F1 | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta F1 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['method']} | {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            f"{row['head_temperature']} | {row['head_uniform_mix']} | "
            + " | ".join(show(row[m]) for m in ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1"))
            + " | "
            + " | ".join(show(row[f"delta_raw_{m}"], signed=True) for m in ("CHAIRs", "CHAIRi", "Recall", "F1"))
            + " |"
        )
    lines.extend([
        "", "## Head diagnostics", "",
        "| Method | Alpha | Rho | Effective head count | Max head weight | Weighted visual mass | UOT residual |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in candidates:
        lines.append(
            f"| {row['method']} | {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            f"{show(row['mean_head_effective_count'], 3)} | "
            f"{show(row['mean_head_max_weight'], 3)} | "
            f"{show(row['mean_head_visual_mass'], 4)} | "
            f"{show(row['mean_uot_dual_residual'], 6)} |"
        )
    lines.extend([
        "", "Select a single candidate after this table using the predeclared rule: retain F1 within 0.005 of its raw-UOT control; then minimize CHAIRs, break ties by CHAIRi and then Recall. The selected configuration must be frozen before held-out seeds 2024 and 3407.",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
