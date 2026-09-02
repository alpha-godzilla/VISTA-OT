import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


class MassCenteredUOTSummaryTests(unittest.TestCase):
    def test_nearest_f1_paired_raw_and_global_pareto(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fields = [
                "method", "setting", "seed", "logits_alpha",
                "marginal_relaxation", "gate_mode", "gpu", "ids_file",
                "result_jsonl", "chair_json", "stats_jsonl",
            ]
            specifications = [
                ("vista", "alpha0.15", 0.15, "na", "baseline", 0.16, 0.06, 0.72),
                ("vista", "alpha0.35", 0.35, "na", "baseline", 0.14, 0.07, 0.68),
                ("direction_aware_uot_raw", "alpha0.35_rho0.7", 0.35, 0.7, "raw", 0.135, 0.065, 0.70),
                ("direction_aware_uot_centered", "alpha0.35_rho0.7", 0.35, 0.7, "centered", 0.13, 0.055, 0.701),
            ]
            entries = []
            for index, spec in enumerate(specifications):
                method, setting, alpha, rho, mode, chairs, chairi, f1 = spec
                chair = root / f"chair_{index}.json"
                chair.write_text(json.dumps({"overall_metrics": {
                    "CHAIRs": chairs, "CHAIRi": chairi, "Recall": 0.56,
                    "Precision": 0.91, "F1": f1, "Len": 2.0,
                }}), encoding="utf-8")
                stats = "na"
                if mode != "baseline":
                    stats_path = root / f"stats_{index}.jsonl"
                    stats_path.write_text(json.dumps({
                        "mean_candidate_promotion_gate": 0.2,
                        "mean_candidate_suppression_gate": 0.3,
                        "mean_attention_retention_abs_deviation": 0.04,
                        "mean_uniform_retention_abs_deviation": 0.03,
                        "mean_attention_uniform_retention_gap": 0.02,
                        "mean_uot_iterations": 50,
                        "mean_uot_dual_residual": 0.0005,
                    }) + "\n", encoding="utf-8")
                    stats = str(stats_path)
                entries.append({
                    "method": method, "setting": setting, "seed": 1994,
                    "logits_alpha": alpha, "marginal_relaxation": rho,
                    "gate_mode": mode, "gpu": 0, "ids_file": "ids",
                    "result_jsonl": "result", "chair_json": str(chair),
                    "stats_jsonl": stats,
                })
            manifest = root / "manifest.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(entries)

            output = root / "summary.csv"
            markdown = root / "summary.md"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/summarize_mass_centered_uot_ablation.py"),
                "--manifest", str(manifest), "--summary-csv", str(output),
                "--markdown", str(markdown),
            ], check=True)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            centered = next(row for row in rows if row["gate_mode"] == "centered")
            self.assertEqual(float(centered["nearest_vista_alpha"]), 0.15)
            self.assertAlmostEqual(float(centered["delta_raw_CHAIRs"]), -0.005)
            self.assertAlmostEqual(float(centered["delta_raw_F1"]), 0.001)
            self.assertEqual(centered["three_metric_global_pareto"], "True")
            report = markdown.read_text(encoding="utf-8")
            self.assertIn("Global CHAIRs/CHAIRi/F1 Pareto frontier", report)
            self.assertIn("Attention-uniform gap", report)


if __name__ == "__main__":
    unittest.main()
