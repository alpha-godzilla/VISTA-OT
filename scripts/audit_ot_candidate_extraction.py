#!/usr/bin/env python3
"""Measure proposal recall before spending GPU time on UOT verification."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.calibrate_apply_ot_candidate_verifier import (
    alias_map,
    canonical_candidate,
    load_chair_map,
    read_jsonl,
    source_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--work-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1994, 2024, 3407])
    parser.add_argument("--vista-method", default="vista")
    parser.add_argument("--vista-setting", default="original")
    parser.add_argument("--ot-method", default="recall_recovery")
    parser.add_argument("--ot-setting", default="rho0.25_k32")
    return parser.parse_args()


def extraction_metrics(work_rows, pairs):
    aliases = alias_map()
    proposed = defaultdict(set)
    generic = defaultdict(int)
    relevant = defaultdict(int)
    for row in work_rows:
        seed = int(row["seed"])
        image_id = int(row["image_id"])
        generic[seed] += 1
        canonical = canonical_candidate(str(row["phrase"]), aliases)
        if canonical is not None:
            relevant[seed] += 1
            proposed[(seed, image_id)].add(canonical)

    result = {}
    for seed, pair in pairs.items():
        vista = load_chair_map(Path(pair["vista"]["chair_json"]))
        ot = load_chair_map(Path(pair["ot"]["chair_json"]))
        oracle = extracted = 0
        for image_id in vista:
            gt = set(vista[image_id]["mscoco_gt_words"])
            true_extras = (
                set(vista[image_id]["mscoco_generated_words"])
                - set(ot[image_id]["mscoco_generated_words"])
            ) & gt
            oracle += len(true_extras)
            extracted += len(true_extras & proposed[(seed, image_id)])
        result[seed] = {
            "generic_candidates": generic[seed],
            "evaluation_relevant_candidates": relevant[seed],
            "extracted_true": extracted,
            "oracle_true_extras": oracle,
            "extraction_recall": extracted / oracle if oracle else 0.0,
        }
    return result


def main():
    args = parse_args()
    pairs = source_pairs(args)
    metrics = extraction_metrics(read_jsonl(args.work_manifest), pairs)
    payload = {"per_seed": metrics}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V2 candidate extraction audit", "",
        "This audit runs before GPU scoring. Labels are used only for offline diagnosis.",
        "",
        "| Seed | Generic | Eval-relevant | Extracted true / oracle | Extraction recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in args.seeds:
        row = metrics[seed]
        lines.append(
            f"| {seed} | {row['generic_candidates']} | "
            f"{row['evaluation_relevant_candidates']} | "
            f"{row['extracted_true']} / {row['oracle_true_extras']} | "
            f"{row['extraction_recall']:.4f} |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
