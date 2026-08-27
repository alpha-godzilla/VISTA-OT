import unittest
from unittest.mock import patch

import torch

from ot_candidate_verifier import (
    append_candidates,
    candidate_ot_features,
    extract_noun_phrases,
    vista_only_candidates,
    word_balanced_target_marginal,
)
from scripts.calibrate_apply_ot_candidate_verifier import calibrate
from scripts.summarize_ot_local_verifier_matched_f1 import (
    build_rows,
    select_pair,
)


class CandidateExtractionTests(unittest.TestCase):
    def test_extracts_generic_compound_nouns_without_dataset_vocabulary(self):
        tagged = [
            ("A", "DT"), ("small", "JJ"), ("traffic", "NN"),
            ("light", "NN"), ("near", "IN"), ("cars", "NNS"), (".", "."),
        ]
        with patch("ot_candidate_verifier._tagged_words", return_value=tagged):
            candidates = extract_noun_phrases("ignored")
        self.assertEqual(candidates[0].phrase, "small traffic light")
        self.assertEqual(candidates[0].head, "light")
        self.assertEqual(candidates[1].head, "car")
        self.assertTrue(candidates[1].plural)

    def test_removes_noun_already_covered_by_ot(self):
        vista_tagged = [("man", "NN"), ("with", "IN"), ("bicycle", "NN")]
        ot_tagged = [("person", "NN")]
        with patch(
            "ot_candidate_verifier._tagged_words",
            side_effect=[vista_tagged, ot_tagged],
        ), patch(
            "ot_candidate_verifier._noun_equivalents",
            side_effect=lambda word: {"person", "man"} if word in {"person", "man"} else {word},
        ):
            candidates = vista_only_candidates("vista", "ot")
        self.assertEqual([candidate.head for candidate in candidates], ["bicycle"])


class CandidateOTFeatureTests(unittest.TestCase):
    def test_preserves_absolute_attention_mass_before_normalization(self):
        torch.manual_seed(0)
        visual = torch.randn(3, 12, 8)
        attention = torch.full((3, 2, 12), 0.01)
        tokens = torch.randn(2, 8)
        result = candidate_ot_features(
            visual, attention, tokens, ["▁traffic", "▁light"],
            region_topks=[4], sinkhorn_iters=10,
        )
        self.assertAlmostEqual(result["visual_attention_mass"], 0.12, places=5)
        region = result["regions"]["4"]
        self.assertEqual(len(region["layer_margins"]), 3)
        self.assertEqual(region["region_topk_effective"], 4)

    def test_word_balanced_marginal_does_not_overweight_split_word(self):
        marginal = word_balanced_target_marginal(
            ["▁traffic", "▁bi", "cycle"], torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(marginal, torch.tensor([0.5, 0.25, 0.25])))

    def test_append_only_preserves_original_caption_prefix(self):
        original = "A person stands beside a car."
        result = append_candidates(original, [{"phrase": "bicycle", "plural": False}])
        self.assertTrue(result.startswith(original))
        self.assertEqual(result, original + " Also visible is a bicycle.")


class GateCalibrationTests(unittest.TestCase):
    @staticmethod
    def row(label, mass, cost, margin, fraction):
        return {
            "label": label,
            "visual_attention_mass": mass,
            "regions": {"16": {
                "positive_cost": cost,
                "median_margin": margin,
                "positive_layer_fraction": fraction,
            }},
        }

    def test_calibration_respects_precision_floor(self):
        rows = [
            self.row(1, 0.4, 0.2, 0.3, 1.0),
            self.row(1, 0.3, 0.3, 0.2, 0.8),
            self.row(0, 0.1, 0.8, -0.2, 0.2),
            self.row(0, 0.2, 0.7, -0.1, 0.3),
        ]
        gate = calibrate(rows, precision_floor=0.95, minimum_tpr=0.3)
        self.assertTrue(gate["passed"])
        self.assertGreaterEqual(gate["calibration_metrics"]["precision"], 0.95)
        self.assertGreaterEqual(gate["calibration_metrics"]["tpr"], 0.3)


class MatchedF1SelectionTests(unittest.TestCase):
    @staticmethod
    def metrics(f1, chairs):
        return {
            "CHAIRs": chairs, "CHAIRi": chairs / 2, "Recall": 0.5,
            "Precision": 0.9, "F1": f1, "Len": 2.0,
        }

    def test_pair_selection_uses_calibration_seed_not_heldout_scores(self):
        vista = {
            0.2: {
                1994: self.metrics(0.701, 0.16),
                2024: self.metrics(0.80, 0.16),
            },
            0.3: {
                1994: self.metrics(0.74, 0.15),
                2024: self.metrics(0.70, 0.15),
            },
        }
        ours = {
            "gate_a": {
                1994: self.metrics(0.700, 0.13),
                # Deliberately bad held-out F1: this must not alter selection.
                2024: self.metrics(0.60, 0.13),
            },
            "gate_b": {
                1994: self.metrics(0.730, 0.12),
                2024: self.metrics(0.70, 0.12),
            },
        }
        rows = build_rows(vista, ours, calibration_seed=1994)
        selected, within = select_pair(rows, tolerance=0.005)
        self.assertTrue(within)
        self.assertEqual(selected["ours_setting"], "gate_a")
        self.assertEqual(selected["vista_logits_alpha"], 0.2)


if __name__ == "__main__":
    unittest.main()
