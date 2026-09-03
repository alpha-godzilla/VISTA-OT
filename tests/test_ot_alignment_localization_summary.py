import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AlignmentLocalizationSummaryTests(unittest.TestCase):
    def test_deltas_are_relative_to_raw_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fields = (
                "method setting seed logits_alpha marginal_relaxation gpu "
                "ids_file result_jsonl chair_json stats_jsonl"
            ).split()
            methods = [
                ("vista", 0.16, 0.71),
                ("raw_direction_aware", 0.14, 0.70),
                ("shared_candidate", 0.15, 0.69),
                ("shared_candidate_final_norm", 0.17, 0.68),
                ("aligned_centered", 0.20, 0.67),
                ("aligned_centered_tgate", 0.21, 0.66),
            ]
            entries = []
            for index, (method, chairs, f1) in enumerate(methods):
                chair = root / f"chair_{index}.json"
                chair.write_text(json.dumps({"overall_metrics": {
                    "CHAIRs": chairs, "CHAIRi": 0.05, "Recall": 0.58,
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
                    method, "fixed", 1994, 0.35,
                    "na" if method == "vista" else 0.7, 0, "ids", "out",
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
                str(ROOT / "scripts/summarize_ot_alignment_localization.py"),
                "--manifest", str(manifest), "--summary-csv", str(output),
                "--markdown", str(markdown),
            ], check=True)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            centered = next(row for row in rows if row["method"] == "aligned_centered")
            self.assertAlmostEqual(float(centered["delta_raw_CHAIRs"]), 0.06)
            self.assertAlmostEqual(float(centered["delta_raw_F1"]), -0.03)
            self.assertIn("raw → shared → final_norm", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
