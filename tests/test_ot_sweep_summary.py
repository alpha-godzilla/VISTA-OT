import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_chair_ot_sweep import read_rows, write_csv, write_markdown


class OTSweepSummaryTests(unittest.TestCase):
    def test_summary_reads_and_sorts_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chair_a = root / "a.json"
            chair_b = root / "b.json"
            metrics = {
                "CHAIRs": 0.1,
                "CHAIRi": 0.2,
                "Recall": 0.3,
                "Precision": 0.4,
                "F1": 0.5,
                "Len": 0.6,
            }
            chair_a.write_text(
                json.dumps({"overall_metrics": metrics}),
                encoding="utf-8",
            )
            chair_b.write_text(
                json.dumps({"overall_metrics": metrics}),
                encoding="utf-8",
            )
            manifest = root / "manifest.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "topk",
                        "visual_tokens",
                        "gpu",
                        "result_jsonl",
                        "chair_json",
                    ),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "topk": 16,
                        "visual_tokens": 64,
                        "gpu": 1,
                        "result_jsonl": "b.jsonl",
                        "chair_json": chair_b,
                    }
                )
                writer.writerow(
                    {
                        "topk": 4,
                        "visual_tokens": 16,
                        "gpu": 0,
                        "result_jsonl": "a.jsonl",
                        "chair_json": chair_a,
                    }
                )

            rows = read_rows(manifest)
            self.assertEqual(
                [(row["topk"], row["visual_tokens"]) for row in rows],
                [(4, 16), (16, 64)],
            )

            csv_path = root / "summary.csv"
            markdown_path = root / "summary.md"
            write_csv(csv_path, rows)
            write_markdown(markdown_path, rows, gamma=0.3, vsv_lambda=0.17)
            self.assertIn(
                "Fixed gamma (`--logits-alpha`): 0.3",
                markdown_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
