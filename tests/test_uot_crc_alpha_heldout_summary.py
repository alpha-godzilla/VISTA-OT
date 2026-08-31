import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HeldoutAlphaSummaryTest(unittest.TestCase):
    def test_excludes_calibration_seed_and_reports_absolute_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.tsv"
            fields = ("method", "setting", "seed", "chair_json")
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for seed in (1994, 2024, 3407):
                    for method, f1 in (("vista", .70), ("ot_stage1", .68), ("uot_crc", .695)):
                        path = root / f"{method}_{seed}.json"
                        path.write_text(json.dumps({"overall_metrics": {
                            "CHAIRs": .16 if method == "vista" else .14,
                            "CHAIRi": .06 if method == "vista" else .045,
                            "Recall": .57 if method == "vista" else .56,
                            "Precision": .89 if method == "vista" else .905,
                            "F1": f1, "Len": 2.0,
                        }}), encoding="utf-8")
                        writer.writerow({
                            "method": method, "setting": "x", "seed": seed,
                            "chair_json": path,
                        })
            by_seed = root / "by_seed.csv"
            summary = root / "summary.csv"
            markdown = root / "summary.md"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/summarize_uot_crc_alpha_heldout.py"),
                "--entry", f"0.3={manifest}", "--heldout-seeds", "2024", "3407",
                "--calibration-seed", "1994", "--by-seed-csv", str(by_seed),
                "--summary-csv", str(summary), "--markdown", str(markdown),
            ], check=True)
            with by_seed.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({int(row["seed"]) for row in rows}, {2024, 3407})
            uot = [row for row in rows if row["method"] == "uot_crc"]
            self.assertEqual(len(uot), 2)
            self.assertAlmostEqual(float(uot[0]["delta_F1"]), -0.005)
            text = markdown.read_text(encoding="utf-8")
            self.assertIn("2024, 3407", text)
            self.assertIn("0.6950", text)


if __name__ == "__main__":
    unittest.main()
