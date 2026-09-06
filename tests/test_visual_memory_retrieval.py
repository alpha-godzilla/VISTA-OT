import math
import unittest

import torch

from visual_memory_retrieval import retrieval_group_metrics


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


if __name__ == "__main__":
    unittest.main()
