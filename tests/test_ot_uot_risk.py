import unittest

from scripts.calibrate_apply_ot_uot_risk import (
    calibrate_threshold,
    operating_stats,
    split_is_tune,
)


class UOTRiskCalibrationTests(unittest.TestCase):
    @staticmethod
    def row(work_id, image_id, target=False, false=False, clean=True):
        return {
            "work_id": work_id,
            "image_id": image_id,
            "target": target,
            "false_object": false,
            "base_clean": clean,
            "evaluation_relevant": True,
        }

    def test_image_split_is_deterministic_and_disjoint(self):
        first = {image_id for image_id in range(100) if split_is_tune(image_id, 0.5, 7)}
        second = {image_id for image_id in range(100) if split_is_tune(image_id, 0.5, 7)}
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertLess(len(first), 100)

    def test_top_one_append_has_sentence_level_added_chairs_loss(self):
        rows = [
            self.row("true", 1, target=True),
            self.row("false-lower", 1, false=True),
            self.row("false", 2, false=True),
        ]
        scores = {"true": 3.0, "false-lower": 2.0, "false": 1.0}
        stats = operating_stats(rows, scores, threshold=0.0, image_ids={1, 2})
        self.assertEqual(stats["accepted"], 2)
        self.assertEqual(stats["true_recoveries"], 1)
        self.assertEqual(stats["added_hallucinations"], 1)
        self.assertAlmostEqual(stats["empirical_added_chairs"], 0.5)

    def test_crc_threshold_respects_finite_sample_correction(self):
        rows = [
            self.row("true", 1, target=True),
            self.row("false", 2, false=True),
        ]
        scores = {"true": 2.0, "false": 1.0}
        threshold, stats = calibrate_threshold(
            rows, scores, calibration_ids=set(range(1, 100)), risk_budget=0.02,
        )
        self.assertGreater(threshold, 1.0)
        self.assertEqual(stats["true_recoveries"], 1)
        self.assertEqual(stats["added_hallucinations"], 0)
        self.assertAlmostEqual(stats["crc_expected_risk_bound"], 0.01)


if __name__ == "__main__":
    unittest.main()
