import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DirectionAwareUOTRefineSummaryTests(unittest.TestCase):
    def test_same_alpha_baseline_and_solver_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.tsv"
            fields = [
                "method", "setting", "seed", "logits_alpha",
                "marginal_relaxation", "layer_weight_reference", "gpu",
                "ids_file", "result_jsonl", "chair_json", "stats_jsonl",
            ]
            entries = []
            for alpha, vista_f1, ours_f1 in ((0.2, 0.70, 0.71), (0.3, 0.68, 0.69)):
                for method, f1, chairs in (
                    ("vista", vista_f1, 0.16),
                    ("direction_aware_uot", ours_f1, 0.14),
                ):
                    stem = f"{method}_{alpha}"
                    chair = root / f"{stem}_chair.json"
                    chair.write_text(json.dumps({"overall_metrics": {
                        "CHAIRs": chairs, "CHAIRi": 0.05, "Recall": 0.57,
                        "Precision": 0.90, "F1": f1, "Len": 1.5,
                    }}), encoding="utf-8")
                    stats = ""
                    if method != "vista":
                        stats_path = root / f"{stem}_stats.jsonl"
                        stats_path.write_text(json.dumps({
                            "mean_uot_iterations": 35.0,
                            "mean_uot_dual_residual": 0.0005,
                            "mean_uniform_uot_iterations": 40.0,
                            "mean_uniform_uot_dual_residual": 0.0007,
                            "mean_candidate_promotion_gate": 0.4,
                            "mean_candidate_suppression_gate": 0.6,
                        }) + "\n", encoding="utf-8")
                        stats = str(stats_path)
                    entries.append({
                        "method": method,
                        "setting": f"alpha{alpha}",
                        "seed": 1994,
                        "logits_alpha": alpha,
                        "marginal_relaxation": "" if method == "vista" else 0.5,
                        "layer_weight_reference": "baseline" if method == "vista" else "independent",
                        "gpu": 0, "ids_file": "ids", "result_jsonl": "result",
                        "chair_json": str(chair), "stats_jsonl": stats,
                    })
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(entries)

            output = root / "summary.csv"
            markdown = root / "summary.md"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/summarize_direction_aware_uot_refine.py"),
                "--manifest", str(manifest),
                "--summary-csv", str(output),
                "--markdown", str(markdown),
            ], check=True)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            ours = [row for row in rows if row["method"] == "direction_aware_uot"]
            self.assertEqual(len(ours), 2)
            self.assertAlmostEqual(float(ours[0]["delta_F1"]), 0.01)
            self.assertAlmostEqual(float(ours[1]["delta_F1"]), 0.01)
            self.assertAlmostEqual(float(ours[0]["mean_uot_iterations"]), 35.0)
            self.assertIn("Solver diagnostics", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
