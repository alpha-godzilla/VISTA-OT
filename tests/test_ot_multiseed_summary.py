import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_chair_ot_multiseed_grid import aggregate_rows, read_rows


METRICS = {
    "CHAIRs": 0.2,
    "CHAIRi": 0.1,
    "Recall": 0.6,
    "Precision": 0.8,
    "F1": 0.7,
    "Len": 1.5,
}


class MultiSeedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_entry(self, method, seed, topk, visual_tokens, delta=0.0):
        ids_path = self.root / f"ids_{seed}.txt"
        ids_path.write_text("11\n22\n", encoding="utf-8")
        result_path = self.root / f"{method}_{seed}_{topk}_{visual_tokens}.jsonl"
        result_path.write_text(
            "".join(
                json.dumps({"image_id": image_id, "caption": "test"}) + "\n"
                for image_id in (11, 22)
            ),
            encoding="utf-8",
        )
        chair_path = result_path.with_name(result_path.stem + "_chair.json")
        metrics = {name: value + delta for name, value in METRICS.items()}
        chair_path.write_text(
            json.dumps({"overall_metrics": metrics}), encoding="utf-8"
        )
        return {
            "method": method,
            "seed": seed,
            "topk": topk,
            "visual_tokens": visual_tokens,
            "gpu": 0,
            "ids_file": str(ids_path),
            "result_jsonl": str(result_path),
            "chair_json": str(chair_path),
        }

    def write_manifest(self, entries):
        path = self.root / "manifest.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=entries[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(entries)
        return path

    def test_aggregate_computes_paired_delta(self):
        entries = []
        for seed in (1, 2):
            entries.append(self.write_entry("vista", seed, 0, 0))
            entries.append(self.write_entry("ot", seed, 4, 16, delta=-0.01))
        rows = read_rows(self.write_manifest(entries))
        aggregate = aggregate_rows(rows)
        ot = next(row for row in aggregate if row["method"] == "ot")

        self.assertEqual(ot["seeds"], 2)
        self.assertAlmostEqual(ot["delta_F1_mean"], -0.01)
        self.assertAlmostEqual(ot["F1_mean"], 0.69)

    def test_mismatched_image_order_is_rejected(self):
        entry = self.write_entry("vista", 1, 0, 0)
        Path(entry["result_jsonl"]).write_text(
            json.dumps({"image_id": 22, "caption": "test"}) + "\n"
            + json.dumps({"image_id": 11, "caption": "test"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Image-ID mismatch"):
            read_rows(self.write_manifest([entry]))


if __name__ == "__main__":
    unittest.main()
