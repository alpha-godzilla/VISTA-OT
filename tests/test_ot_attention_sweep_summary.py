import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.summarize_chair_ot_attention_grid import main


METRICS = {
    "CHAIRs": 0.2,
    "CHAIRi": 0.1,
    "Recall": 0.6,
    "Precision": 0.8,
    "F1": 0.7,
    "Len": 1.5,
}


class AttentionSweepSummaryTests(unittest.TestCase):
    def test_summary_uses_alpha_matched_baseline_and_config_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = []
            for method, logits_alpha, layer_temperature, power, mix, delta in (
                ("vista", "0.5", "baseline", "baseline", "baseline", 0.0),
                ("ot", "0.5", "0.2", "0.5", "0.02", -0.01),
                ("ot", "0.5", "0.2", "1.0", "0.0", 0.02),
            ):
                stem = f"{method}_{logits_alpha}_{layer_temperature}_{power}_{mix}"
                chair_path = root / f"{stem}_chair.json"
                chair_path.write_text(
                    json.dumps({
                        "overall_metrics": {
                            name: value + delta for name, value in METRICS.items()
                        }
                    }),
                    encoding="utf-8",
                )
                stats_path = root / f"{stem}_stats.jsonl"
                if method == "ot":
                    stats_path.write_text(
                        json.dumps({"mean_layer_weights": [0.2] * 5}) + "\n",
                        encoding="utf-8",
                    )
                entries.append({
                    "method": method,
                    "seed": "2024",
                    "logits_alpha": logits_alpha,
                    "layer_temperature": layer_temperature,
                    "attention_power": power,
                    "uniform_mix": mix,
                    "gpu": "0",
                    "ids_file": "ids.txt",
                    "result_jsonl": f"{stem}.jsonl",
                    "chair_json": str(chair_path),
                    "stats_jsonl": str(stats_path),
                })
            manifest = root / "manifest.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=entries[0], delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(entries)

            summary = root / "summary.csv"
            weights = root / "weights.csv"
            markdown = root / "summary.md"
            argv = [
                "summarize_chair_ot_attention_grid.py",
                "--manifest", str(manifest),
                "--summary-csv", str(summary),
                "--weight-csv", str(weights),
                "--markdown", str(markdown),
            ]
            with patch.object(sys, "argv", argv):
                main()

            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["logits_alpha"], "0.5")
            self.assertAlmostEqual(float(rows[0]["delta_F1_mean"]), -0.01)
            self.assertIn("Attention power", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
