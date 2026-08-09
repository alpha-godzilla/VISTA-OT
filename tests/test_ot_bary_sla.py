import unittest

import torch

from ot_bary_sla import OTBarySLA, log_sinkhorn


class LogSinkhornTests(unittest.TestCase):
    def test_shape_nonnegative_and_total_mass(self):
        torch.manual_seed(0)
        cost = torch.rand(2, 5, 37, 9)
        source = torch.full((1, 1, 37), 1.0 / 37)
        target = torch.full((1, 1, 9), 1.0 / 9)

        plan = log_sinkhorn(cost, source, target, epsilon=0.05, num_iters=3)

        self.assertEqual(plan.shape, (2, 5, 37, 9))
        self.assertTrue(torch.all(plan >= 0))
        torch.testing.assert_close(
            plan.sum(dim=(-2, -1)),
            torch.ones(2, 5),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_marginals_with_more_iterations(self):
        torch.manual_seed(1)
        cost = torch.rand(2, 3, 7, 5)
        source = torch.full((1, 1, 7), 1.0 / 7)
        target = torch.full((1, 1, 5), 1.0 / 5)

        plan = log_sinkhorn(cost, source, target, epsilon=0.1, num_iters=100)

        torch.testing.assert_close(
            plan.sum(dim=-1),
            source.expand(2, 3, -1),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            plan.sum(dim=-2),
            target.expand(2, 3, -1),
            atol=1e-5,
            rtol=1e-5,
        )


class OTBarySLATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.visual = torch.randn(2, 16, 12)
        self.embedding = torch.randn(32, 12)
        self.early_logits = torch.randn(2, 5, 32)

    def test_output_shapes_and_weight_distribution(self):
        method = OTBarySLA(
            topk=8,
            visual_tokens=9,
            sinkhorn_iters=5,
        )
        method.cache_visual_features(self.visual)
        weights, details = method.compute_layer_weights(
            self.early_logits,
            self.embedding,
            return_details=True,
        )

        self.assertEqual(details["transport_plan"].shape, (2, 5, 10, 8))
        self.assertEqual(details["layer_scores"].shape, (2, 5))
        self.assertEqual(weights.shape, (2, 5))
        self.assertTrue(torch.all(weights >= 0))
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(2),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_force_uniform_reproduces_original_sla(self):
        method = OTBarySLA(
            topk=4,
            visual_tokens=9,
            force_uniform=True,
        )
        method.cache_visual_features(self.visual)
        final_logits = torch.randn(2, 32)
        gamma = 0.3

        mixed, details = method.aggregate(
            self.early_logits,
            final_logits,
            self.embedding,
            gamma,
        )
        expected = (
            (1.0 - gamma) * final_logits
            + gamma * self.early_logits.mean(dim=1)
        )

        torch.testing.assert_close(
            details["layer_weights"],
            torch.full((2, 5), 0.2),
        )
        torch.testing.assert_close(mixed, expected, atol=1e-6, rtol=1e-6)

    def test_aligned_layer_receives_highest_weight(self):
        embedding = torch.zeros(12, 4)
        embedding[0:2, 0] = 1.0
        embedding[2:4, 1] = 1.0
        embedding[4:6, 2] = 1.0
        embedding[6:] = torch.randn(6, 4)
        visual = torch.zeros(1, 4, 4)
        visual[..., 0] = 1.0

        logits = torch.full((1, 3, 12), -20.0)
        logits[0, 0, 2:4] = 10.0
        logits[0, 1, 0:2] = 10.0
        logits[0, 2, 4:6] = 10.0

        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            epsilon=0.05,
            sinkhorn_iters=20,
        )
        method.cache_visual_features(visual)
        weights, details = method.compute_layer_weights(
            logits,
            embedding,
            return_details=True,
        )

        self.assertEqual(details["layer_scores"].argmax(dim=-1).item(), 1)
        self.assertEqual(weights.argmax(dim=-1).item(), 1)

    def test_special_tokens_are_excluded(self):
        logits = self.early_logits.clone()
        logits[..., 0] = 1000
        logits[..., 1] = 999
        method = OTBarySLA(
            topk=4,
            visual_tokens=9,
            special_token_ids=[0, 1],
        )
        method.cache_visual_features(self.visual)
        _, details = method.compute_layer_weights(
            logits,
            self.embedding,
            return_details=True,
        )
        ids = details["candidate_ids"]
        self.assertFalse(torch.any(ids == 0))
        self.assertFalse(torch.any(ids == 1))

    def test_text_marginal_uses_selected_logit_probabilities(self):
        method = OTBarySLA(topk=4, visual_tokens=9)
        method.cache_visual_features(self.visual)
        _, details = method.compute_layer_weights(
            self.early_logits,
            self.embedding,
            return_details=True,
        )
        selected_logits = torch.gather(
            self.early_logits,
            dim=-1,
            index=details["candidate_ids"],
        )
        expected = torch.softmax(selected_logits.float(), dim=-1)
        torch.testing.assert_close(details["target_marginal"], expected)
        self.assertEqual(details["target_marginal"].shape, (2, 5, 4))

    def test_visual_marginal_keeps_uniform_avg_dustbin_weight(self):
        method = OTBarySLA(topk=4, visual_tokens=9)
        method.cache_visual_features(self.visual)
        _, details = method.compute_layer_weights(
            self.early_logits,
            self.embedding,
            return_details=True,
        )
        torch.testing.assert_close(
            details["source_marginal"],
            torch.full((1, 1, 10), 0.1),
        )

    def test_layer_score_excludes_only_visual_dustbin_row(self):
        method = OTBarySLA(topk=4, visual_tokens=9)
        method.cache_visual_features(self.visual)
        _, details = method.compute_layer_weights(
            self.early_logits,
            self.embedding,
            return_details=True,
        )
        expected = (
            details["transport_plan"][..., :-1, :]
            * details["similarity"][..., :-1, :]
        ).sum(dim=(-2, -1))
        torch.testing.assert_close(details["layer_scores"], expected)

    def test_mixed_precision_inputs_are_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                method = OTBarySLA(topk=4, visual_tokens=9)
                method.cache_visual_features(self.visual)
                weights = method.compute_layer_weights(
                    self.early_logits.to(dtype),
                    self.embedding.to(dtype),
                )
                self.assertEqual(weights.dtype, dtype)
                self.assertTrue(torch.isfinite(weights.float()).all())

    def test_visual_cache_does_not_duplicate_short_inputs(self):
        visual = torch.randn(2, 4, 12)
        method = OTBarySLA(topk=4, visual_tokens=9)
        method.cache_visual_features(visual)
        weights, details = method.compute_layer_weights(
            self.early_logits,
            self.embedding,
            return_details=True,
        )
        self.assertEqual(details["transport_plan"].shape[-2], 5)
        self.assertEqual(weights.shape, (2, 5))

    def test_diagnostics_are_aggregated_without_per_step_output(self):
        method = OTBarySLA(topk=4, visual_tokens=9, log_stats=True)
        method.cache_visual_features(self.visual)
        method.compute_layer_weights(self.early_logits, self.embedding)
        method.compute_layer_weights(self.early_logits, self.embedding)
        diagnostics = method.get_diagnostics()

        self.assertEqual(diagnostics["steps"], 2)
        self.assertEqual(len(diagnostics["mean_layer_weights"]), 5)
        self.assertIn("mean_local_transport_mass", diagnostics)
        self.assertIn("mean_dustbin_to_token_mass", diagnostics)
        self.assertNotIn("mean_token_to_dustbin_mass", diagnostics)


if __name__ == "__main__":
    unittest.main()
