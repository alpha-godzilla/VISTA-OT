import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RawDirectionAwareGridSummaryTests(unittest.TestCase):
    def test_nearest_f1_reference_and_pareto_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fields = (
                "method setting seed logits_alpha marginal_relaxation gpu "
                "ids_file result_jsonl chair_json stats_jsonl"
            ).split()
            specs = [
                ("vista", 0.15, "na", 0.16, 0.06, 0.72),
                ("vista", 0.35, "na", 0.14, 0.07, 0.68),
                ("raw_direction_aware_uot", 0.25, 0.5, 0.13, 0.055, 0.701),
                ("raw_direction_aware_uot", 0.35, 0.7, 0.12, 0.060, 0.699),
            ]
            entries = []
            for index, (method, alpha, rho, chairs, chairi, f1) in enumerate(specs):
                chair = root / f"chair_{index}.json"
                chair.write_text(json.dumps({"overall_metrics": {
                    "CHAIRs": chairs, "CHAIRi": chairi, "Recall": 0.57,
                    "Precision": 0.90, "F1": f1, "Len": 2.0,
                }}), encoding="utf-8")
                stats = "na"
                if method != "vista":
                    stats_path = root / f"stats_{index}.jsonl"
                    stats_path.write_text(json.dumps({
                        "mean_uot_iterations": 55,
                        "mean_uot_dual_residual": 0.0007,
                    }) + "\n", encoding="utf-8")
                    stats = str(stats_path)
                entries.append(dict(zip(fields, [
                    method, "setting", 1994, alpha, rho, 0, "ids", "out",
                    str(chair), stats,
                ])))
            manifest = root / "manifest.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(entries)
            output, markdown = root / "summary.csv", root / "summary.md"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/summarize_raw_direction_aware_uot_grid.py"),
                "--manifest", str(manifest), "--summary-csv", str(output),
                "--markdown", str(markdown),
            ], check=True)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            row = next(row for row in rows if row["logits_alpha"] == "0.25")
            self.assertEqual(float(row["nearest_vista_alpha"]), 0.15)
            self.assertAlmostEqual(float(row["delta_nearest_vista_F1"]), -0.019)
            self.assertEqual(row["three_metric_pareto"], "True")
            report = markdown.read_text(encoding="utf-8")
            self.assertIn("Three-metric Pareto frontier", report)
            self.assertIn("not an equal-intervention-strength", report)


if __name__ == "__main__":
    unittest.main()
