#!/usr/bin/env python3
"""Calibrate an OT candidate gate on one seed and append accepted objects."""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chair_ans import synonyms_txt  # Evaluation-only import; never used by GPU inference.
from ot_candidate_verifier import append_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--work-manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1994, 2024, 3407])
    parser.add_argument("--calibration-seed", type=int, default=1994)
    parser.add_argument("--vista-method", default="vista")
    parser.add_argument("--vista-setting", default="original")
    parser.add_argument("--ot-method", default="recall_recovery")
    parser.add_argument("--ot-setting", default="rho0.25_k32")
    parser.add_argument("--precision-floor", type=float, default=0.95)
    parser.add_argument("--minimum-tpr", type=float, default=0.30)
    parser.add_argument("--max-additions", type=int, default=2)
    parser.add_argument(
        "--allow-failed-gate", action="store_true",
        help="Apply the safest calibrated gate even if the go/no-go TPR fails.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_caption_map(path):
    rows = read_jsonl(path)
    return {int(row["image_id"]): row for row in rows}


def load_chair_map(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(row["image_id"]): row for row in payload["sentences"]}


def source_pairs(args):
    wanted = set(args.seeds)
    pairs = {seed: {} for seed in args.seeds}
    with args.source_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seed = int(row["seed"])
            if seed not in wanted:
                continue
            if row["method"] == args.vista_method and row["setting"] == args.vista_setting:
                pairs[seed]["vista"] = row
            if row["method"] == args.ot_method and row["setting"] == args.ot_setting:
                pairs[seed]["ot"] = row
    for seed, pair in pairs.items():
        if set(pair) != {"vista", "ot"}:
            raise ValueError(f"Missing VISTA/OT source pair for seed={seed}")
    return pairs


def alias_map():
    aliases = {}
    for line in synonyms_txt.splitlines():
        items = [item.strip().lower() for item in line.split(",") if item.strip()]
        if not items:
            continue
        canonical = items[0]
        for item in items:
            aliases[item] = canonical
    return aliases


def canonical_candidate(phrase, aliases):
    text = re.sub(r"[^a-z0-9 ]+", " ", phrase.lower())
    padded = f" {' '.join(text.split())} "
    matches = [
        (len(alias.split()), alias, canonical)
        for alias, canonical in aliases.items()
        if f" {alias} " in padded
    ]
    return max(matches)[2] if matches else None


def label_scores(scores, pairs, aliases):
    chair_cache = {}
    labeled = []
    for row in scores:
        seed = int(row["seed"])
        if seed not in chair_cache:
            chair_cache[seed] = {
                "vista": load_chair_map(pairs[seed]["vista"]["chair_json"]),
                "ot": load_chair_map(pairs[seed]["ot"]["chair_json"]),
            }
        image_id = int(row["image_id"])
        canonical = canonical_candidate(row["phrase"], aliases)
        item = dict(row)
        item["chair_object"] = canonical
        item["evaluation_relevant"] = canonical is not None
        if canonical is None:
            item["label"] = None
        else:
            vista_chair = chair_cache[seed]["vista"][image_id]
            ot_chair = chair_cache[seed]["ot"][image_id]
            gt = set(vista_chair["mscoco_gt_words"])
            ot_objects = set(ot_chair["mscoco_generated_words"])
            # Redundant proposals are negatives for recall recovery: accepting
            # them adds length but cannot recover a missing ground-truth object.
            item["label"] = int(canonical in gt and canonical not in ot_objects)
        labeled.append(item)
    return labeled


def quantile_values(values, direction):
    values = sorted(float(x) for x in values)
    if not values:
        return []
    indices = {round(q * (len(values) - 1)) for q in [i / 10 for i in range(11)]}
    selected = sorted({values[index] for index in indices})
    if direction == "min":
        return [float("-inf")] + selected
    return selected + [float("inf")]


def accepted(row, region, gate):
    features = row["regions"][str(region)]
    return (
        row["visual_attention_mass"] >= gate["minimum_attention_mass"]
        and features["positive_cost"] <= gate["maximum_positive_cost"]
        and features["median_margin"] >= gate["minimum_median_margin"]
        and features["positive_layer_fraction"] >= gate["minimum_layer_fraction"]
    )


def gate_metrics(rows, region, gate, positive_denominator=None):
    extracted_positives = sum(row["label"] == 1 for row in rows)
    positives = extracted_positives if positive_denominator is None else positive_denominator
    negatives = sum(row["label"] == 0 for row in rows)
    chosen = [row for row in rows if accepted(row, region, gate)]
    tp = sum(row["label"] == 1 for row in chosen)
    fp = sum(row["label"] == 0 for row in chosen)
    precision = tp / (tp + fp) if tp + fp else 1.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "accepted": tp + fp,
        "precision": precision,
        "tpr": tp / positives if positives else 0.0,
        "fpr": fp / negatives if negatives else 0.0,
        "positives": positives,
        "extracted_positives": extracted_positives,
        "negatives": negatives,
    }


def calibrate(rows, precision_floor, minimum_tpr, positive_denominator=None):
    relevant = [row for row in rows if row["label"] is not None]
    if not relevant or not any(row["label"] == 1 for row in relevant):
        raise ValueError("Calibration seed contains no evaluation-relevant true candidates")
    regions = sorted({int(key) for row in relevant for key in row["regions"]})
    best = None
    best_any = None
    for region in regions:
        masses = quantile_values([row["visual_attention_mass"] for row in relevant], "min")
        costs = quantile_values([row["regions"][str(region)]["positive_cost"] for row in relevant], "max")
        margins = quantile_values([row["regions"][str(region)]["median_margin"] for row in relevant], "min")
        fractions = quantile_values([row["regions"][str(region)]["positive_layer_fraction"] for row in relevant], "min")
        for mass in masses:
            for cost in costs:
                for margin in margins:
                    for fraction in fractions:
                        gate = {
                            "region_topk": region,
                            "minimum_attention_mass": mass,
                            "maximum_positive_cost": cost,
                            "minimum_median_margin": margin,
                            "minimum_layer_fraction": fraction,
                        }
                        metrics = gate_metrics(
                            relevant, region, gate,
                            positive_denominator=positive_denominator,
                        )
                        if metrics["accepted"] == 0:
                            continue
                        fallback_rank = (
                            metrics["precision"], metrics["tpr"],
                            -metrics["false_positives"], -metrics["accepted"],
                        )
                        if best_any is None or fallback_rank > best_any[0]:
                            best_any = (fallback_rank, gate, metrics)
                        if metrics["precision"] < precision_floor:
                            continue
                        # Primary objective is recovery at the required precision;
                        # ties prefer higher precision and fewer false accepts.
                        rank = (
                            metrics["tpr"], metrics["precision"],
                            -metrics["false_positives"], -metrics["accepted"],
                        )
                        if best is None or rank > best[0]:
                            best = (rank, gate, metrics)
    selected = best if best is not None else best_any
    if selected is None:
        raise RuntimeError("No candidate was accepted by any calibrated gate")
    gate, metrics = selected[1], selected[2]
    gate["calibration_metrics"] = metrics
    gate["precision_floor"] = precision_floor
    gate["minimum_tpr"] = minimum_tpr
    gate["precision_floor_met"] = metrics["precision"] >= precision_floor
    gate["passed"] = (
        gate["precision_floor_met"] and metrics["tpr"] >= minimum_tpr
    )
    return gate


def oracle_true_extra_total(pair):
    vista = load_chair_map(pair["vista"]["chair_json"])
    ot = load_chair_map(pair["ot"]["chair_json"])
    if set(vista) != set(ot):
        raise ValueError("VISTA/OT CHAIR image sets differ")
    total = 0
    for image_id in vista:
        gt = set(vista[image_id]["mscoco_gt_words"])
        vista_objects = set(vista[image_id]["mscoco_generated_words"])
        ot_objects = set(ot[image_id]["mscoco_generated_words"])
        total += len((vista_objects - ot_objects) & gt)
    return total


def finite_gate(gate):
    result = {}
    for key, value in gate.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = "-inf" if value < 0 else "inf"
        else:
            result[key] = value
    return result


def append_outputs(args, scores, pairs, gate):
    region = gate["region_topk"]
    chosen = defaultdict(list)
    for row in scores:
        if accepted(row, region, gate):
            chosen[(int(row["seed"]), int(row["image_id"]))].append(row)

    outputs = {}
    for seed in args.seeds:
        base_rows = read_jsonl(pairs[seed]["ot"]["result_jsonl"])
        output_path = args.output_dir / f"seed{seed}_ot_local_verifier.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for base in base_rows:
                key = (seed, int(base["image_id"]))
                accepted_rows = sorted(
                    chosen.get(key, []),
                    key=lambda row: (
                        row["regions"][str(region)]["median_margin"],
                        row["visual_attention_mass"],
                    ),
                    reverse=True,
                )[:args.max_additions]
                caption = append_candidates(base["caption"], accepted_rows)
                handle.write(json.dumps({
                    "image_id": int(base["image_id"]),
                    "caption": caption,
                    "ot_caption": base["caption"],
                    "accepted_candidates": [
                        {
                            "phrase": row["phrase"],
                            "head": row["head"],
                            "work_id": row["work_id"],
                        }
                        for row in accepted_rows
                    ],
                }) + "\n")
        outputs[seed] = output_path
    return outputs


def write_manifest(args, pairs, outputs, gate):
    setting = f"r{gate['region_topk']}_p{args.precision_floor:g}"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "setting", "seed", "gpu", "ids_file", "result_jsonl", "chair_json"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for seed in args.seeds:
            for method, setting_name, source in (
                ("vista", args.vista_setting, pairs[seed]["vista"]),
                ("ot_stage1", args.ot_setting, pairs[seed]["ot"]),
            ):
                writer.writerow({
                    "method": method, "setting": setting_name, "seed": seed,
                    "gpu": -1, "ids_file": source["ids_file"],
                    "result_jsonl": source["result_jsonl"],
                    "chair_json": source["chair_json"],
                })
            output = outputs[seed]
            writer.writerow({
                "method": "local_verifier", "setting": setting, "seed": seed,
                "gpu": -1, "ids_file": pairs[seed]["ot"]["ids_file"],
                "result_jsonl": output,
                "chair_json": output.with_name(output.stem + "_chair.json"),
            })


def report(args, labeled, gate):
    per_seed = {}
    pairs = source_pairs(args)
    for seed in args.seeds:
        rows = [row for row in labeled if int(row["seed"]) == seed and row["label"] is not None]
        oracle_positives = oracle_true_extra_total(pairs[seed])
        per_seed[seed] = gate_metrics(
            rows, gate["region_topk"], gate,
            positive_denominator=oracle_positives,
        )
        per_seed[seed]["all_generic_candidates"] = sum(
            int(row["seed"]) == seed for row in labeled
        )
        per_seed[seed]["evaluation_relevant_candidates"] = len(rows)
    payload = {"gate": finite_gate(gate), "per_seed_candidate_metrics": per_seed}
    args.report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metrics = gate["calibration_metrics"]
    lines = [
        "# Candidate-conditioned local OT verifier",
        "",
        "CHAIR/COCO labels are used only here for calibration and reporting; the GPU scorer and candidate extractor do not import them.",
        "",
        f"Calibration seed: **{args.calibration_seed}**",
        f"Selected visual region top-k: **{gate['region_topk']}**",
        f"Gate passed: **{gate['passed']}**",
        f"Calibration precision / TPR / FPR: **{metrics['precision']:.4f} / {metrics['tpr']:.4f} / {metrics['fpr']:.4f}**",
        "",
        "TPR uses all oracle VISTA-only true extras as its denominator, so it includes candidate-extraction misses.",
        "",
        "| Seed | Generic candidates | Eval-relevant | Extracted true / oracle | Accepted | TP | FP | Precision | End-to-end TPR | FPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in args.seeds:
        row = per_seed[seed]
        lines.append(
            f"| {seed} | {row['all_generic_candidates']} | "
            f"{row['evaluation_relevant_candidates']} | "
            f"{row['extracted_positives']} / {row['positives']} | {row['accepted']} | "
            f"{row['true_positives']} | {row['false_positives']} | "
            f"{row['precision']:.4f} | {row['tpr']:.4f} | {row['fpr']:.4f} |"
        )
    args.report_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.calibration_seed not in args.seeds:
        raise ValueError("calibration-seed must be included in seeds")
    if not 0 < args.precision_floor <= 1 or not 0 <= args.minimum_tpr <= 1:
        raise ValueError("precision-floor and minimum-tpr must be probabilities")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_work_ids = {row["work_id"] for row in read_jsonl(args.work_manifest)}
    by_work_id = {}
    for path in args.scores:
        for row in read_jsonl(path):
            if row["work_id"] in current_work_ids:
                by_work_id[row["work_id"]] = row
    missing = current_work_ids - set(by_work_id)
    if missing:
        raise ValueError(f"Missing scores for {len(missing)} current work items")
    scores = list(by_work_id.values())
    pairs = source_pairs(args)
    labeled = label_scores(scores, pairs, alias_map())
    calibration_rows = [
        row for row in labeled if int(row["seed"]) == args.calibration_seed
    ]
    calibration_oracle_positives = oracle_true_extra_total(
        pairs[args.calibration_seed]
    )
    gate = calibrate(
        calibration_rows, args.precision_floor, args.minimum_tpr,
        positive_denominator=calibration_oracle_positives,
    )
    report(args, labeled, gate)
    if not gate["passed"] and not args.allow_failed_gate:
        raise RuntimeError(
            "Verifier did not meet the go/no-go operating point. Review "
            f"{args.report_markdown}; pass --allow-failed-gate only for diagnosis."
        )
    outputs = append_outputs(args, scores, pairs, gate)
    write_manifest(args, pairs, outputs, gate)
    print(f"Wrote {args.report_markdown}")
    print(f"Wrote {args.output_manifest}")


if __name__ == "__main__":
    main()
