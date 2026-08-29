#!/usr/bin/env python3
"""Tune a scalar UOT verifier and calibrate append-only CHAIRs risk.

The calibration seed is deterministically split by image.  UOT configuration,
feature normalization, and linear score are learned on the tune half.  Only a
one-dimensional acceptance threshold is selected on the disjoint calibration
half using the conformal-risk correction ``(errors + 1) / (n + 1)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ot_candidate_verifier import append_candidates
from scripts.calibrate_apply_ot_candidate_verifier import (
    alias_map,
    canonical_candidate,
    load_chair_map,
    read_jsonl,
    source_pairs,
)


FEATURE_NAMES = (
    "log_attention_mass",
    "negative_cost",
    "log_transport_mass",
    "hard_negative_margin",
    "negative_layer_cost_std",
    "positive_layer_fraction",
)
CONTRAST_FEATURE_NAMES = (
    "counterfactual_cost_gap",
    "clean_to_counterfactual_log_mass",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--work-manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--mode", choices=("uot", "contrast"), default="uot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1994, 2024, 3407])
    parser.add_argument("--calibration-seed", type=int, default=1994)
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    parser.add_argument("--split-salt", type=int, default=2024)
    parser.add_argument("--risk-budget", type=float, default=0.01)
    parser.add_argument("--tune-precision-floor", type=float, default=0.9)
    parser.add_argument("--max-additions", type=int, default=1)
    parser.add_argument("--logistic-l2", type=float, default=0.1)
    parser.add_argument("--vista-method", default="vista")
    parser.add_argument("--vista-setting", default="original")
    parser.add_argument("--ot-method", default="recall_recovery")
    parser.add_argument("--ot-setting", default="rho0.25_k32")
    return parser.parse_args()


def split_is_tune(image_id: int, fraction: float, salt: int) -> bool:
    digest = hashlib.sha256(f"{salt}:{image_id}".encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def score_files(paths, current_work_ids):
    by_id = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("work_id") in current_work_ids:
                by_id[row["work_id"]] = row
    missing = current_work_ids - set(by_id)
    if missing:
        raise ValueError(f"Missing UOT scores for {len(missing)} work items")
    return list(by_id.values())


def annotate_rows(rows, pairs):
    aliases = alias_map()
    chairs = {}
    oracle = {}
    for seed, pair in pairs.items():
        vista = load_chair_map(Path(pair["vista"]["chair_json"]))
        ot = load_chair_map(Path(pair["ot"]["chair_json"]))
        chairs[seed] = {"vista": vista, "ot": ot}
        oracle[seed] = sum(
            len(
                (set(vista[image_id]["mscoco_generated_words"])
                 - set(ot[image_id]["mscoco_generated_words"]))
                & set(vista[image_id]["mscoco_gt_words"])
            )
            for image_id in vista
        )
    annotated = []
    for original in rows:
        row = dict(original)
        seed = int(row["seed"])
        image_id = int(row["image_id"])
        canonical = canonical_candidate(str(row["phrase"]), aliases)
        ot_item = chairs[seed]["ot"][image_id]
        gt = set(ot_item["mscoco_gt_words"])
        generated = set(ot_item["mscoco_generated_words"])
        row["chair_object"] = canonical
        row["evaluation_relevant"] = canonical is not None
        row["target"] = bool(canonical in gt and canonical not in generated)
        row["false_object"] = bool(canonical is not None and canonical not in gt)
        row["base_clean"] = not bool(ot_item["mscoco_hallucinated_words"])
        annotated.append(row)
    return annotated, oracle


def available_configs(rows, mode):
    configs = set()
    for row in rows:
        clean = row.get("uot", {}).get("relaxations", {})
        if mode == "contrast" and "uot_counterfactual" not in row:
            continue
        for relaxation, regions in clean.items():
            for region in regions:
                configs.add((relaxation, region))
    if not configs:
        raise ValueError(f"No {mode} UOT configurations found in score files")
    return sorted(configs, key=lambda item: (float(item[0]), int(item[1])))


def feature_vector(row, relaxation, region, mode):
    clean = row["uot"]["relaxations"][relaxation][region]
    values = [
        math.log(max(float(row["visual_attention_mass"]), 1e-12)),
        -float(clean["normalized_cost"]),
        math.log(max(float(clean["transport_mass"]), 1e-12)),
        float(clean["normalized_margin"]),
        -float(clean["layer_cost_std"]),
        float(clean["positive_layer_fraction"]),
    ]
    if mode == "contrast":
        noisy = row["uot_counterfactual"]["relaxations"][relaxation][region]
        values.extend([
            float(noisy["normalized_cost"]) - float(clean["normalized_cost"]),
            math.log(max(float(clean["transport_mass"]), 1e-12))
            - math.log(max(float(noisy["transport_mass"]), 1e-12)),
        ])
    return np.asarray(values, dtype=np.float64)


def fit_logistic(features, labels, l2):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(set(y.tolist())) < 2:
        raise ValueError("Tune split needs both positive and negative candidates")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    design = np.concatenate([np.ones((len(x), 1)), standardized], axis=1)
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    sample_weight = np.where(y > 0, 0.5 / positives, 0.5 / negatives) * len(y)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * l2
    regularizer[0, 0] = 0.0
    for _ in range(50):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weight * (probability - y)) / len(y)
        gradient += regularizer @ weights
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design / len(y) + regularizer
        try:
            step = np.linalg.solve(hessian + 1e-8 * np.eye(len(weights)), gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"mean": mean, "scale": scale, "weights": weights}


def predict(model, feature):
    standardized = (feature - model["mean"]) / model["scale"]
    design = np.concatenate([[1.0], standardized])
    return float(design @ model["weights"])


def score_rows(rows, relaxation, region, mode, model):
    return {
        row["work_id"]: predict(
            model, feature_vector(row, relaxation, region, mode),
        )
        for row in rows
    }


def operating_stats(rows, scores, threshold, image_ids):
    grouped = defaultdict(list)
    for row in rows:
        if int(row["image_id"]) in image_ids:
            grouped[int(row["image_id"])].append(row)
    chosen = []
    for image_id in image_ids:
        candidates = grouped.get(image_id, [])
        if not candidates:
            continue
        best = max(candidates, key=lambda row: scores[row["work_id"]])
        if scores[best["work_id"]] >= threshold:
            chosen.append(best)
    tp = sum(bool(row["target"]) for row in chosen)
    relevant_fp = sum(
        bool(row["evaluation_relevant"]) and not bool(row["target"])
        for row in chosen
    )
    added_hallucinations = sum(
        bool(row["base_clean"]) and bool(row["false_object"])
        for row in chosen
    )
    relevant_accepted = tp + relevant_fp
    return {
        "accepted": len(chosen),
        "true_recoveries": tp,
        "relevant_false_accepts": relevant_fp,
        "precision": tp / relevant_accepted if relevant_accepted else 1.0,
        "added_hallucinations": added_hallucinations,
        "empirical_added_chairs": added_hallucinations / max(1, len(image_ids)),
    }


def threshold_grid(rows, scores, image_ids):
    values = sorted({
        scores[row["work_id"]]
        for row in rows if int(row["image_id"]) in image_ids
    }, reverse=True)
    return [float("inf"), *values]


def tune_configuration(rows, tune_ids, mode, risk_budget, precision_floor, l2):
    relevant = [
        row for row in rows
        if int(row["image_id"]) in tune_ids and row["evaluation_relevant"]
    ]
    best = None
    for relaxation, region in available_configs(rows, mode):
        model = fit_logistic(
            [feature_vector(row, relaxation, region, mode) for row in relevant],
            [int(row["target"]) for row in relevant],
            l2,
        )
        scores = score_rows(rows, relaxation, region, mode, model)
        best_threshold = None
        for threshold in threshold_grid(rows, scores, tune_ids):
            stats = operating_stats(rows, scores, threshold, tune_ids)
            if stats["empirical_added_chairs"] > risk_budget:
                continue
            if stats["precision"] < precision_floor:
                continue
            rank = (
                stats["true_recoveries"], stats["precision"],
                -stats["added_hallucinations"], stats["accepted"],
            )
            if best_threshold is None or rank > best_threshold[0]:
                best_threshold = (rank, threshold, stats)
        if best_threshold is None:
            continue
        config_rank = (*best_threshold[0], -float(relaxation), -int(region))
        if best is None or config_rank > best[0]:
            best = (
                config_rank, relaxation, region, model,
                best_threshold[1], best_threshold[2], scores,
            )
    if best is None:
        raise RuntimeError("No UOT configuration meets the tune constraints")
    return {
        "relaxation": best[1], "region": best[2], "model": best[3],
        "tune_threshold": best[4], "tune_stats": best[5], "scores": best[6],
    }


def calibrate_threshold(rows, scores, calibration_ids, risk_budget):
    best = None
    n = len(calibration_ids)
    for threshold in threshold_grid(rows, scores, calibration_ids):
        stats = operating_stats(rows, scores, threshold, calibration_ids)
        # Bounded monotone loss correction from conformal risk control.
        crc_bound = (stats["added_hallucinations"] + 1.0) / (n + 1.0)
        stats["crc_expected_risk_bound"] = crc_bound
        if crc_bound > risk_budget:
            continue
        rank = (
            stats["true_recoveries"], stats["precision"],
            -stats["added_hallucinations"], stats["accepted"],
        )
        if best is None or rank > best[0]:
            best = (rank, threshold, stats)
    if best is None:
        raise RuntimeError(
            "Risk budget is below the finite-sample CRC resolution; increase "
            "calibration size or risk budget"
        )
    return best[1], best[2]


def apply_outputs(args, rows, scores, threshold, pairs):
    chosen = defaultdict(list)
    for row in rows:
        if scores[row["work_id"]] >= threshold:
            chosen[(int(row["seed"]), int(row["image_id"]))].append(row)
    outputs = {}
    for seed in args.seeds:
        output = args.output_dir / f"seed{seed}_ot_{args.output_tag}.jsonl"
        base_rows = read_jsonl(Path(pairs[seed]["ot"]["result_jsonl"]))
        with output.open("w", encoding="utf-8") as handle:
            for base in base_rows:
                key = (seed, int(base["image_id"]))
                selected = sorted(
                    chosen.get(key, []),
                    key=lambda row: scores[row["work_id"]], reverse=True,
                )[:args.max_additions]
                caption = append_candidates(base["caption"], selected)
                handle.write(json.dumps({
                    "image_id": int(base["image_id"]),
                    "caption": caption,
                    "ot_caption": base["caption"],
                    "accepted_candidates": [
                        {
                            "phrase": row["phrase"], "head": row["head"],
                            "work_id": row["work_id"],
                            "score": scores[row["work_id"]],
                        }
                        for row in selected
                    ],
                }) + "\n")
        outputs[seed] = output
    return outputs


def write_manifest(args, pairs, outputs, selected, calibration_stats):
    setting = (
        f"tau{selected['relaxation']}_k{selected['region']}"
        f"_risk{args.risk_budget:g}_m{args.max_additions}"
    )
    method = "uot_contrast_crc" if args.mode == "contrast" else "uot_crc"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method", "setting", "seed", "gpu", "ids_file", "result_jsonl",
            "chair_json", "gate_passed", "crc_expected_risk_bound",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for seed in args.seeds:
            for source_method, source_setting, source in (
                ("vista", args.vista_setting, pairs[seed]["vista"]),
                ("ot_stage1", args.ot_setting, pairs[seed]["ot"]),
            ):
                writer.writerow({
                    "method": source_method, "setting": source_setting,
                    "seed": seed, "gpu": -1, "ids_file": source["ids_file"],
                    "result_jsonl": source["result_jsonl"],
                    "chair_json": source["chair_json"], "gate_passed": "",
                    "crc_expected_risk_bound": "",
                })
            output = outputs[seed]
            writer.writerow({
                "method": method, "setting": setting, "seed": seed, "gpu": -1,
                "ids_file": pairs[seed]["ot"]["ids_file"],
                "result_jsonl": output,
                "chair_json": output.with_name(output.stem + "_chair.json"),
                "gate_passed": str(
                    calibration_stats["true_recoveries"] > 0
                    and calibration_stats["crc_expected_risk_bound"]
                    <= args.risk_budget
                ).lower(),
                "crc_expected_risk_bound": calibration_stats[
                    "crc_expected_risk_bound"
                ],
            })
    return method, setting


def report(args, rows, oracle, tune_ids, calibration_ids, excluded_overlap,
           selected, threshold, calibration_stats, pairs, method, setting):
    per_seed = {}
    for seed in args.seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        seed_ids = set(load_chair_map(Path(pairs[seed]["ot"]["chair_json"])))
        stats = operating_stats(seed_rows, selected["scores"], threshold, seed_ids)
        per_seed[seed] = {
            "generic_candidates": len(seed_rows),
            "evaluation_relevant_candidates": sum(
                bool(row["evaluation_relevant"]) for row in seed_rows
            ),
            "extracted_true": sum(bool(row["target"]) for row in seed_rows),
            "oracle_true_extras": oracle[seed],
            **stats,
        }
    feature_names = list(FEATURE_NAMES)
    if args.mode == "contrast":
        feature_names.extend(CONTRAST_FEATURE_NAMES)
    model_json = {
        "feature_names": feature_names,
        "mean": selected["model"]["mean"].tolist(),
        "scale": selected["model"]["scale"].tolist(),
        "weights": selected["model"]["weights"].tolist(),
    }
    payload = {
        "method": method, "setting": setting, "mode": args.mode,
        "calibration_seed": args.calibration_seed,
        "tune_images": len(tune_ids), "calibration_images": len(calibration_ids),
        "development_images_excluded_for_heldout_overlap": excluded_overlap,
        "risk_budget": args.risk_budget,
        "selected_relaxation": selected["relaxation"],
        "selected_region": selected["region"],
        "selected_threshold": threshold,
        "score_model": model_json,
        "tune_stats": selected["tune_stats"],
        "calibration_stats": calibration_stats,
        "per_seed": per_seed,
    }
    args.report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {method}: scalar UOT + CHAIRs risk control", "",
        "The calibration seed is split by image; tune and calibration images are disjoint.",
        "The reported CRC bound controls expected added sentence-level CHAIR risk under exchangeability; it is not a deterministic held-out guarantee.",
        "",
        f"- Setting: `{setting}`",
        f"- Tune / calibration images: **{len(tune_ids)} / {len(calibration_ids)}**",
        f"- Development images excluded for held-out overlap: **{excluded_overlap}**",
        f"- Threshold: **{threshold:.6g}**",
        f"- Calibration added hallucinations: **{calibration_stats['added_hallucinations']}**",
        f"- CRC expected-risk bound / budget: **{calibration_stats['crc_expected_risk_bound']:.4f} / {args.risk_budget:.4f}**",
        "",
        "| Seed | Generic | Eval-relevant | Extracted true / oracle | Accepted | TP | Added CHAIRs risk | Precision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in args.seeds:
        item = per_seed[seed]
        lines.append(
            f"| {seed} | {item['generic_candidates']} | "
            f"{item['evaluation_relevant_candidates']} | "
            f"{item['extracted_true']} / {item['oracle_true_extras']} | "
            f"{item['accepted']} | {item['true_recoveries']} | "
            f"{item['empirical_added_chairs']:.4f} | {item['precision']:.4f} |"
        )
    args.report_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.calibration_seed not in args.seeds:
        raise ValueError("calibration-seed must be included in seeds")
    if not 0 < args.tune_fraction < 1:
        raise ValueError("tune-fraction must be in (0, 1)")
    if not 0 < args.risk_budget <= 1:
        raise ValueError("risk-budget must be in (0, 1]")
    if not 0 < args.tune_precision_floor <= 1:
        raise ValueError("tune-precision-floor must be in (0, 1]")
    if args.max_additions != 1:
        raise ValueError("Risk-controlled v2 intentionally requires max-additions=1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_work_ids = {row["work_id"] for row in read_jsonl(args.work_manifest)}
    raw_scores = score_files(args.scores, current_work_ids)
    pairs = source_pairs(args)
    rows, oracle = annotate_rows(raw_scores, pairs)
    image_ids_by_seed = {
        seed: set(load_chair_map(Path(pair["ot"]["chair_json"])))
        for seed, pair in pairs.items()
    }
    calibration_seed_ids = image_ids_by_seed[args.calibration_seed]
    heldout_ids = set().union(*(
        image_ids for seed, image_ids in image_ids_by_seed.items()
        if seed != args.calibration_seed
    ))
    overlap = calibration_seed_ids & heldout_ids
    development_ids = calibration_seed_ids - overlap
    tune_ids = {
        image_id for image_id in development_ids
        if split_is_tune(image_id, args.tune_fraction, args.split_salt)
    }
    calibration_ids = development_ids - tune_ids
    if not tune_ids or not calibration_ids:
        raise RuntimeError("Deterministic split produced an empty partition")
    calibration_seed_rows = [
        row for row in rows if int(row["seed"]) == args.calibration_seed
    ]
    selected = tune_configuration(
        calibration_seed_rows, tune_ids, args.mode, args.risk_budget,
        args.tune_precision_floor, args.logistic_l2,
    )
    threshold, calibration_stats = calibrate_threshold(
        calibration_seed_rows, selected["scores"], calibration_ids,
        args.risk_budget,
    )
    # The score map learned on the calibration seed is applied unchanged to
    # every seed.  Recompute scores for held-out rows, never refit the model.
    selected["scores"] = score_rows(
        rows, selected["relaxation"], selected["region"], args.mode,
        selected["model"],
    )
    outputs = apply_outputs(args, rows, selected["scores"], threshold, pairs)
    method, setting = write_manifest(
        args, pairs, outputs, selected, calibration_stats,
    )
    report(
        args, rows, oracle, tune_ids, calibration_ids, len(overlap), selected,
        threshold, calibration_stats, pairs, method, setting,
    )
    print(f"Wrote {args.report_markdown}")
    print(f"Wrote {args.output_manifest}")


if __name__ == "__main__":
    main()
