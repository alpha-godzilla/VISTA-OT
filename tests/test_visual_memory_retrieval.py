import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import torch

from visual_memory_retrieval import METRIC_NAMES, VALID_NAMES, merge_trace_directory, retrieval_group_metrics


class RetrievalGroupMetricsTest(unittest.TestCase):
    def test_exact_decomposition_and_count_identity(self):
        torch.manual_seed(7)
        query = torch.randn(2, 4)
        keys = torch.randn(2, 5, 4)
        values = torch.randn(2, 5, 4)
        visual = torch.tensor([True, False, False, False, False])
        prompt = torch.tensor([False, True, True, False, False])
        generated = torch.tensor([False, False, False, True, True])
        logits = torch.einsum("hd,hkd->hk", query, keys) / math.sqrt(4)
        original = torch.einsum("hk,hkd->hd", torch.softmax(logits, dim=-1), values)
        result = retrieval_group_metrics(query, keys, values, visual, prompt, generated, original)
        self.assertLess(float(result["reconstruction_error"].max()), 1e-5)
        identity = math.log(2 / 1) + result["compat_GV"]
        self.assertTrue(torch.allclose(result["L_GV"], identity, atol=1e-6))
        self.assertLess(float(result["L_GV_identity_error"].max()), 1e-6)

    def test_empty_generated_group_is_finite_and_marked_invalid(self):
        query = torch.ones(1, 2)
        keys = torch.ones(1, 2, 2)
        values = torch.ones(1, 2, 2)
        visual = torch.tensor([True, False])
        prompt = torch.tensor([False, True])
        generated = torch.tensor([False, False])
        result = retrieval_group_metrics(query, keys, values, visual, prompt, generated)
        self.assertFalse(bool(result["has_G"][0]))
        self.assertEqual(float(result["L_GV"][0]), 0.0)
        self.assertTrue(torch.isfinite(result["reconstruction_error"]).all())

    def test_merge_preserves_all_axes_as_flat_records(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            arrays = {name: np.ones((2, 1, 2), dtype=np.float32) for name in METRIC_NAMES}
            arrays.update({name: np.ones((2, 1, 2), dtype=np.bool_) for name in VALID_NAMES})
            arrays.update({name: np.ones((2, 1), dtype=np.int32) for name in ("n_V", "n_P", "n_G")})
            np.savez_compressed(directory / "sample_7.npz", token_ids=np.array([-1, 3]), token_text=np.array(["x", "y"]), **arrays)
            output = merge_trace_directory(directory, directory / "all.npz")
            with np.load(output) as summary:
                self.assertEqual(summary["sample_id"].shape, (4,))
                self.assertEqual(summary["timestep"].tolist(), [0, 0, 1, 1])
                self.assertEqual(summary["head"].tolist(), [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
