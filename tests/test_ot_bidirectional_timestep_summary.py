import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BidirectionalTimestepSummaryTests(unittest.TestCase):
    def test_exact_pair_nearest_vista_and_small_gate_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fields = [
                "method", "setting", "seed", "logits_alpha",
                "marginal_relaxation", "timestep_gate", "gpu", "ids_file",
                "result_jsonl", "chair_json", "stats_jsonl",
            ]
            specs = [
                ("vista", 0.15, "na", False, 0.16, 0.70),
                ("vista", 0.35, "na", False, 0.14, 0.66),
                ("aligned_centered", 0.35, 0.7, False, 0.145, 0.695),
                ("aligned_centered_tgate", 0.35, 0.7, True, 0.142, 0.698),
            ]
            entries = []
            for index, (method, alpha, rho, tgate, chairs, f1) in enumerate(specs):
                chair = root / f"chair_{index}.json"
                chair.write_text(json.dumps({"overall_metrics": {
                    "CHAIRs": chairs, "CHAIRi": 0.05, "Recall": 0.56,
                    "Precision": 0.90, "F1": f1, "Len": 2.0,
                }}), encoding="utf-8")
                stats = "na"
                if method != "vista":
                    stats_path = root / f"stats_{index}.jsonl"
                    stats_path.write_text(json.dumps({
                        "mean_candidate_promotion_gate": 0.02,
                        "mean_candidate_suppression_gate": 0.03,
                        "mean_timestep_promotion_strength": 0.01 if tgate else 1.0,
                        "mean_timestep_suppression_strength": 0.02 if tgate else 1.0,
                        "mean_attention_retention_abs_deviation": 0.005,
                        "mean_uniform_retention_abs_deviation": 0.003,
                        "mean_attention_uniform_retention_gap": 0.002,
                        "mean_uot_iterations": 45,
                        "mean_uot_dual_residual": 0.0004,
                    }) + "\n", encoding="utf-8")
                    stats = str(stats_path)
                entries.append({
                    "method": method, "setting": f"alpha{alpha}", "seed": 1994,
                    "logits_alpha": alpha, "marginal_relaxation": rho,
                    "timestep_gate": str(tgate).lower(), "gpu": 0,
                    "ids_file": "ids", "result_jsonl": "result",
                    "chair_json": str(chair), "stats_jsonl": stats,
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
                str(ROOT / "scripts/summarize_ot_bidirectional_timestep_ablation.py"),
                "--manifest", str(manifest), "--summary-csv", str(output),
                "--markdown", str(markdown),
            ], check=True)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            gated = next(row for row in rows if row["method"] == "aligned_centered_tgate")
            self.assertAlmostEqual(float(gated["delta_token_only_CHAIRs"]), -0.003)
            self.assertAlmostEqual(float(gated["delta_token_only_F1"]), 0.003)
            self.assertEqual(float(gated["nearest_vista_alpha"]), 0.15)
            self.assertEqual(float(gated["mean_timestep_promotion_strength"]), 0.01)
            report = markdown.read_text(encoding="utf-8")
            self.assertIn("exact alpha/rho pair", report)
            self.assertIn("over-attenuating", report)
            self.assertIn("E[qg]", report)


if __name__ == "__main__":
    unittest.main()
