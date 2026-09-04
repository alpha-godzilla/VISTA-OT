import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HeadAwareUOTSummaryTests(unittest.TestCase):
    def test_reports_delta_to_identical_raw_control(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fields = (
                "method setting seed logits_alpha marginal_relaxation "
                "head_temperature head_uniform_mix head_topk head_mass_weight "
                "gpu ids_file result_jsonl chair_json stats_jsonl"
            ).split()
            specs = [
                ("vista", "na", 0.15, "na", 0.16, 0.72),
                ("raw_direction_aware_uot", "na", 0.35, 0.6, 0.14, 0.70),
                ("head_mass", "0.5", 0.35, 0.6, 0.13, 0.702),
                ("head_uot_uniform", "0.1", 0.35, 0.6, 0.12, 0.699),
            ]
            entries = []
            for index, (method, temp, alpha, rho, chairs, f1) in enumerate(specs):
                chair = root / f"chair_{index}.json"
                chair.write_text(json.dumps({"overall_metrics": {
                    "CHAIRs": chairs, "CHAIRi": 0.05, "Recall": 0.57,
                    "Precision": 0.90, "F1": f1, "Len": 2.0,
                }}), encoding="utf-8")
                stats = "na"
                if method != "vista":
                    stats_path = root / f"stats_{index}.jsonl"
                    stats_path.write_text(json.dumps({
                        "mean_head_effective_count": 2.0,
                        "mean_head_max_weight": 0.7,
                    }) + "\n", encoding="utf-8")
                    stats = str(stats_path)
                entries.append(dict(zip(fields, [
                    method, "x", 1994, alpha, rho, temp,
                    "0.05" if method == "head_uot_uniform" else "0",
                    "4", "0.1", "0", "ids", "out", str(chair), stats,
                ])))
            manifest = root / "manifest.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(entries)
            output, markdown = root / "summary.csv", root / "summary.md"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/summarize_head_aware_uot_grid.py"),
                "--manifest", str(manifest), "--summary-csv", str(output),
                "--markdown", str(markdown),
            ], check=True)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            mass = next(row for row in rows if row["method"] == "head_mass")
            self.assertAlmostEqual(float(mass["delta_raw_CHAIRs"]), -0.01)
            self.assertAlmostEqual(float(mass["delta_raw_F1"]), 0.002)
            self.assertIn("predeclared rule", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
