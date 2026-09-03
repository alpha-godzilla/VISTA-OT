#!/usr/bin/env python3
"""Summarize the paired bidirectional-timestep-gate development ablation."""

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
    "mean_timestep_promotion_strength",
    "mean_timestep_suppression_strength",
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


def nearest_f1(row, references):
    return min(
        references,
        key=lambda ref: (
            abs(ref["F1"] - row["F1"]),
            -ref["F1"],
            ref["logits_alpha"],
        ),
    )


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
            "method": entry["method"],
            "setting": entry["setting"],
            "seed": int(entry["seed"]),
            "logits_alpha": float(entry["logits_alpha"]),
            "marginal_relaxation": (
                None if entry["marginal_relaxation"] in {"", "na"}
                else float(entry["marginal_relaxation"])
            ),
            "timestep_gate": entry["timestep_gate"].lower() == "true",
            **read_metrics(Path(entry["chair_json"])),
            **read_diagnostics(stats),
        })

    vista = [row for row in rows if row["method"] == "vista"]
    if not vista:
        raise ValueError("At least one evaluated VISTA reference is required")
    token_only = {
        (row["seed"], row["logits_alpha"], row["marginal_relaxation"]): row
        for row in rows if row["method"] == "aligned_centered"
    }
    for row in rows:
        reference = nearest_f1(row, vista)
        row["nearest_vista_alpha"] = reference["logits_alpha"]
        for metric in METRICS:
            row[f"delta_nearest_vista_{metric}"] = row[metric] - reference[metric]

        paired = None
        if row["method"] == "aligned_centered_tgate":
            paired = token_only.get((
                row["seed"], row["logits_alpha"], row["marginal_relaxation"],
            ))
            if paired is None:
                raise ValueError(
                    "Every timestep-gated row needs an exact seed/alpha/rho token-only pair"
                )
        for metric in METRICS:
            row[f"delta_token_only_{metric}"] = (
                row[metric] - paired[metric] if paired is not None else None
            )

    rows.sort(key=lambda row: (row["method"], row["logits_alpha"]))
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ot_rows = [row for row in rows if row["method"] != "vista"]
    gated = [row for row in rows if row["method"] == "aligned_centered_tgate"]
    lines = [
        "# Bidirectional timestep-gate paired ablation", "",
        "This is a label-free generation sweep on the single development seed 1994. CHAIR ground truth is used only after generation for evaluation and must not be treated as held-out evidence.",
        "", "## Absolute results", "",
        "| Method | Alpha | Rho | CHAIRs | CHAIRi | Recall | Precision | F1 | Len |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['logits_alpha']:g} | "
            f"{show(row['marginal_relaxation'])} | {show(row['CHAIRs'])} | "
            f"{show(row['CHAIRi'])} | {show(row['Recall'])} | "
            f"{show(row['Precision'])} | {show(row['F1'])} | {show(row['Len'])} |"
        )

    lines.extend([
        "", "## Timestep gate versus token gate only (exact alpha/rho pair)", "",
        "| Alpha | Rho | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta Precision | Delta F1 | Delta Len |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in gated:
        lines.append(
            f"| {row['logits_alpha']:g} | {row['marginal_relaxation']:g} | "
            + " | ".join(
                show(row[f"delta_token_only_{metric}"], signed=True)
                for metric in METRICS
            ) + " |"
        )

    lines.extend([
        "", "## Nearest-F1 VISTA comparison", "",
        "| Method | Alpha | VISTA alpha | Delta CHAIRs | Delta CHAIRi | Delta Recall | Delta Precision | Delta F1 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ot_rows:
        lines.append(
            f"| {row['method']} | {row['logits_alpha']:g} | "
            f"{row['nearest_vista_alpha']:g} | "
            + " | ".join(
                show(row[f"delta_nearest_vista_{metric}"], signed=True)
                for metric in ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1")
            ) + " |"
        )

    lines.extend([
        "", "## Gate and solver diagnostics", "",
        "| Method | Alpha | Token promote | Token suppress | q+ | q- | Attention deviation | Uniform deviation | Retention gap | UOT iters | Residual |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ot_rows:
        lines.append(
            f"| {row['method']} | {row['logits_alpha']:g} | "
            f"{show(row['mean_candidate_promotion_gate'], 6)} | "
            f"{show(row['mean_candidate_suppression_gate'], 6)} | "
            f"{show(row['mean_timestep_promotion_strength'], 6)} | "
            f"{show(row['mean_timestep_suppression_strength'], 6)} | "
            f"{show(row['mean_attention_retention_abs_deviation'], 6)} | "
            f"{show(row['mean_uniform_retention_abs_deviation'], 6)} | "
            f"{show(row['mean_attention_uniform_retention_gap'], 6)} | "
            f"{show(row['mean_uot_iterations'], 2)} | "
            f"{show(row['mean_uot_dual_residual'], 6)} |"
        )

    if gated and all(
        max(
            row["mean_timestep_promotion_strength"] or 0.0,
            row["mean_timestep_suppression_strength"] or 0.0,
        ) < 0.05
        for row in gated
    ):
        lines.extend([
            "", "**Diagnostic warning:** mean `q+` and `q-` are below 0.05 for every gated setting. This is evidence that the second gate may be over-attenuating an already centered token-level intervention; inspect per-step traces before expanding the sweep.",
        ])

    lines.extend([
        "", "The reported gate means are separate expectations. Their product is intentionally not reported as an effective coefficient because, in general, E[qg] is not E[q]E[g].",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
