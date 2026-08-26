import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chair_ot_two_stage import build_fusion_query, load_proposals
from scripts.analyze_chair_proposal_overlap import main as overlap_main


class TwoStagePromptTests(unittest.TestCase):
    def test_dual_prompt_assigns_distinct_roles_to_proposals(self):
        query = build_fusion_query(
            "dual_fusion",
            "A person with a bicycle.",
            "A person outdoors.",
        )
        self.assertIn("High-coverage draft: A person with a bicycle.", query)
        self.assertIn("Conservative draft: A person outdoors.", query)
        self.assertIn("only when the image supports them", query)

    def test_self_prompt_does_not_include_vista_proposal(self):
        query = build_fusion_query("ot_self", "VISTA SECRET", "OT DRAFT")
        self.assertNotIn("VISTA SECRET", query)
        self.assertIn("OT DRAFT", query)

    def test_load_proposals_rejects_duplicate_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "caps.jsonl"
            path.write_text(
                json.dumps({"image_id": 1, "caption": "one"}) + "\n" +
                json.dumps({"image_id": 1, "caption": "two"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate image_id"):
                load_proposals(path)


class ProposalOverlapTests(unittest.TestCase):
    def test_reports_true_and_false_vista_only_objects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vista_chair = root / "vista.json"
            ot_chair = root / "ot.json"
            vista_chair.write_text(json.dumps({"sentences": [{
                "image_id": 1,
                "mscoco_gt_words": ["person", "bicycle"],
                "mscoco_generated_words": ["person", "bicycle", "dog"],
            }]}), encoding="utf-8")
            ot_chair.write_text(json.dumps({"sentences": [{
                "image_id": 1,
                "mscoco_gt_words": ["person", "bicycle"],
                "mscoco_generated_words": ["person"],
            }]}), encoding="utf-8")
            manifest = root / "manifest.tsv"
            entries = [
                {"method": "vista", "setting": "original", "seed": "2024",
                 "gpu": "-1", "ids_file": "ids", "result_jsonl": "vista.jsonl",
                 "chair_json": str(vista_chair)},
                {"method": "ot_stage1", "setting": "rho0.25_k32", "seed": "2024",
                 "gpu": "-1", "ids_file": "ids", "result_jsonl": "ot.jsonl",
                 "chair_json": str(ot_chair)},
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(entries[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(entries)
            output_csv = root / "overlap.csv"
            output_md = root / "overlap.md"
            argv = ["analyze", "--manifest", str(manifest), "--csv", str(output_csv),
                    "--markdown", str(output_md)]
            with patch.object(sys, "argv", argv):
                overlap_main()
            with output_csv.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(int(row["true_extra_objects"]), 1)
            self.assertEqual(int(row["false_extra_objects"]), 1)
            self.assertAlmostEqual(float(row["recoverable_recall"]), 0.5)
            self.assertAlmostEqual(float(row["proposal_precision"]), 0.5)


if __name__ == "__main__":
    unittest.main()
