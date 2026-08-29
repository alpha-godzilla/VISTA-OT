#!/usr/bin/env python3
"""Select and report an ours/VISTA pair matched by calibration-seed F1."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vista-manifest", type=Path, required=True)
    parser.add_argument("--ours-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration-seed", type=int, default=1994)
    parser.add_argument("--f1-tolerance", type=float, default=0.005)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path):
    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)["overall_metrics"]
    return {name: float(values[name]) for name in METRICS}


def load_vista(path):
    configs = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["method"] != "vista":
                continue
            configs[float(row["logits_alpha"])][int(row["seed"])] = read_metrics(
                row["chair_json"]
            )
    return configs


def load_ours(paths):
    configs = defaultdict(dict)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["method"] != "local_verifier":
                    continue
                setting = row["setting"]
                seed = int(row["seed"])
                metrics = read_metrics(row["chair_json"])
                metrics["_gate_passed"] = row.get(
                    "gate_passed", "true",
                ).lower() == "true"
                metrics["_calibration_precision"] = float(
                    row.get("calibration_precision") or "nan"
                )
                metrics["_calibration_tpr"] = float(
                    row.get("calibration_tpr") or "nan"
                )
                existing = configs[setting].get(seed)
                if existing is not None and existing != metrics:
                    raise ValueError(
                        f"Conflicting ours metrics for setting={setting}, seed={seed}"
                    )
                configs[setting][seed] = metrics
    return configs


def mean(values):
    return statistics.fmean(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def build_rows(vista, ours, calibration_seed):
    rows = []
    for ours_setting, ours_seeds in sorted(ours.items()):
        if calibration_seed not in ours_seeds:
            raise ValueError(f"Ours setting {ours_setting} lacks calibration seed")
        for vista_alpha, vista_seeds in sorted(vista.items()):
            if calibration_seed not in vista_seeds:
                raise ValueError(f"VISTA alpha {vista_alpha:g} lacks calibration seed")
            common = sorted(set(ours_seeds) & set(vista_seeds))
            heldout = [seed for seed in common if seed != calibration_seed]
            if not heldout:
                raise ValueError("Matched-F1 comparison needs at least one held-out seed")
            dev_ours = ours_seeds[calibration_seed]
            dev_vista = vista_seeds[calibration_seed]
            row = {
                "ours_setting": ours_setting,
                "vista_logits_alpha": vista_alpha,
                "calibration_seed": calibration_seed,
                "heldout_seeds": ",".join(str(seed) for seed in heldout),
                "dev_ours_F1": dev_ours["F1"],
                "dev_vista_F1": dev_vista["F1"],
                "dev_delta_F1": dev_ours["F1"] - dev_vista["F1"],
                "dev_abs_F1_gap": abs(dev_ours["F1"] - dev_vista["F1"]),
                "ours_gate_passed": dev_ours.get("_gate_passed", True),
                "ours_calibration_precision": dev_ours.get(
                    "_calibration_precision", float("nan"),
                ),
                "ours_calibration_tpr": dev_ours.get(
                    "_calibration_tpr", float("nan"),
                ),
            }
            for metric in METRICS:
                row[f"dev_ours_{metric}"] = dev_ours[metric]
                row[f"dev_vista_{metric}"] = dev_vista[metric]
                row[f"dev_delta_{metric}"] = dev_ours[metric] - dev_vista[metric]
                ours_values = [ours_seeds[seed][metric] for seed in heldout]
                vista_values = [vista_seeds[seed][metric] for seed in heldout]
                deltas = [ours_seeds[seed][metric] - vista_seeds[seed][metric] for seed in heldout]
                row[f"heldout_ours_{metric}_mean"] = mean(ours_values)
                row[f"heldout_vista_{metric}_mean"] = mean(vista_values)
                row[f"heldout_delta_{metric}_mean"] = mean(deltas)
                row[f"heldout_delta_{metric}_std"] = sample_std(deltas)
            row["heldout_abs_F1_gap"] = abs(row["heldout_delta_F1_mean"])
            rows.append(row)
    return rows


def select_pair(rows, tolerance):
    eligible = [row for row in rows if row["dev_abs_F1_gap"] <= tolerance]
    pool = eligible or rows
    # Matching quality is the primary criterion. CHAIRs only breaks near-ties,
    # so the comparison does not select a large F1 mismatch for a nicer score.
    selected = min(
        pool,
        key=lambda row: (
            row["dev_abs_F1_gap"],
            row["dev_delta_CHAIRs"],
            -row["dev_delta_Recall"],
        ),
    )
    return selected, bool(eligible)


def main():
    args = parse_args()
    if args.f1_tolerance < 0:
        raise ValueError("f1-tolerance must be non-negative")
    vista = load_vista(args.vista_manifest)
    ours = load_ours(args.ours_manifests)
    if not vista or not ours:
        raise ValueError("Both VISTA and ours manifests must contain evaluated rows")
    rows = build_rows(vista, ours, args.calibration_seed)
    selected, within_tolerance = select_pair(rows, args.f1_tolerance)
    for row in rows:
        row["selected"] = row is selected
        row["dev_within_tolerance"] = row["dev_abs_F1_gap"] <= args.f1_tolerance
    rows.sort(key=lambda row: (row["dev_abs_F1_gap"], row["dev_delta_CHAIRs"]))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    warning = (
        "The selected pair satisfies the preset calibration-seed F1 tolerance."
        if within_tolerance else
        "No pair satisfies the preset F1 tolerance; the nearest pair is reported and must not be described as equal-F1."
    )
    lines = [
        "# Ours vs VISTA at matched F1",
        "",
        f"Selection uses only calibration seed `{args.calibration_seed}` with F1 tolerance `{args.f1_tolerance:.4f}`. Held-out seeds are reported without retuning.",
        "",
        warning,
        "",
        f"Selected ours setting: **{selected['ours_setting']}**",
        f"Selected VISTA logits_alpha: **{selected['vista_logits_alpha']:g}**",
        f"Selected ours gate passed original go/no-go: **{selected['ours_gate_passed']}**",
        f"Selected ours calibration precision / end-to-end TPR: **{selected['ours_calibration_precision']:.4f} / {selected['ours_calibration_tpr']:.4f}**",
        f"Calibration F1 (ours / VISTA / gap): **{selected['dev_ours_F1']:.4f} / {selected['dev_vista_F1']:.4f} / {selected['dev_delta_F1']:+.4f}**",
        "",
        "| Held-out metric | Ours | VISTA | Paired delta (ours - VISTA) |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {selected[f'heldout_ours_{metric}_mean']:.4f} | "
            f"{selected[f'heldout_vista_{metric}_mean']:.4f} | "
            f"{selected[f'heldout_delta_{metric}_mean']:+.4f} +/- "
            f"{selected[f'heldout_delta_{metric}_std']:.4f} |"
        )
    lines.extend([
        "",
        "The CSV contains every ours/VISTA pairing, including both calibration and held-out F1 gaps.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
