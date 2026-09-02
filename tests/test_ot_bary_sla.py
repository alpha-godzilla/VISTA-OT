import unittest

import torch

from ot_bary_sla import OTBarySLA, log_sinkhorn, log_unbalanced_sinkhorn


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

    def test_tolerance_bounds_both_marginal_residuals(self):
        torch.manual_seed(11)
        cost = torch.rand(1, 2, 32, 8)
        source = torch.softmax(torch.randn(1, 2, 32) * 2.0, dim=-1)
        target = torch.softmax(torch.randn(1, 2, 8), dim=-1)
        plan = log_sinkhorn(
            cost, source, target,
            epsilon=0.05, num_iters=200, tolerance=1e-3,
        )
        self.assertLessEqual(
            (plan.sum(dim=-1) - source).abs().max().item(), 1e-3,
        )
        self.assertLessEqual(
            (plan.sum(dim=-2) - target).abs().max().item(), 1e-3,
        )

    def test_unbalanced_transport_mass_decreases_with_cost(self):
        source = torch.full((1, 1, 4), 0.25)
        target = torch.full((1, 1, 2), 0.5)
        low = log_unbalanced_sinkhorn(
            torch.zeros(1, 1, 4, 2), source, target,
            epsilon=0.05, marginal_relaxation=0.5,
            num_iters=100, tolerance=1e-5,
        )
        high = log_unbalanced_sinkhorn(
            torch.ones(1, 1, 4, 2), source, target,
            epsilon=0.05, marginal_relaxation=0.5,
            num_iters=100, tolerance=1e-5,
        )
        self.assertLess(high.sum().item(), low.sum().item())
        self.assertAlmostEqual(low.sum().item(), 1.0, places=5)
        self.assertLess(high.sum().item(), 1.0)
        self.assertTrue(torch.isfinite(low).all())
        self.assertTrue(torch.isfinite(high).all())

    def test_unbalanced_solver_reports_convergence(self):
        plan, diagnostics = log_unbalanced_sinkhorn(
            torch.full((1, 2, 4, 2), 0.25),
            torch.full((1, 2, 4), 0.25),
            torch.full((1, 2, 2), 0.5),
            epsilon=0.05, marginal_relaxation=0.5,
            num_iters=100, tolerance=1e-3,
            return_diagnostics=True,
        )
        self.assertTrue(torch.isfinite(plan).all())
        self.assertLessEqual(diagnostics["iterations"].item(), 100)
        self.assertLessEqual(diagnostics["dual_residual"].item(), 1e-3)


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

    def test_target_marginal_stays_strictly_positive_after_softmax_underflow(self):
        method = OTBarySLA(topk=4, visual_tokens=4, sinkhorn_iters=5)
        method.cache_visual_features(torch.randn(1, 4, 12))
        logits = torch.full((1, 2, 32), -1000.0)
        logits[..., 0] = 1000.0
        _, details = method.compute_layer_weights(
            logits, self.embedding, return_details=True,
        )
        self.assertTrue(torch.all(details["target_marginal"] > 0))
        torch.testing.assert_close(
            details["target_marginal"].sum(dim=-1),
            torch.ones(1, 2),
        )

    def test_independent_uniform_weights_require_directional_mode(self):
        with self.assertRaisesRegex(ValueError, "direction-aware gating"):
            OTBarySLA(independent_uniform_layer_weights=True)

    def test_mass_centering_requires_directional_mode(self):
        with self.assertRaisesRegex(ValueError, "direction-aware gating"):
            OTBarySLA(mass_centered_direction_gating=True)

    def test_timestep_gate_requires_mass_centered_directional_mode(self):
        with self.assertRaisesRegex(ValueError, "mass-centered"):
            OTBarySLA(bidirectional_timestep_gating=True)

    def test_timestep_gate_requires_shared_candidates_and_final_norm(self):
        common = dict(
            attention_visual_marginal=True,
            unbalanced=True,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
            mass_centered_direction_gating=True,
        )
        with self.assertRaisesRegex(ValueError, "shared candidate"):
            OTBarySLA(**common, bidirectional_timestep_gating=True)
        with self.assertRaisesRegex(ValueError, "final-norm"):
            OTBarySLA(
                **common,
                shared_candidate_set=True,
                bidirectional_timestep_gating=True,
            )

    def test_shared_candidate_set_is_identical_across_layers(self):
        method = OTBarySLA(topk=3, shared_candidate_set=True)
        logits = torch.tensor([[
            [8.0, 1.0, 0.0, -1.0, -2.0],
            [-2.0, 7.0, 1.0, 0.0, -1.0],
        ]])
        candidate_ids = method._candidate_ids(logits)
        self.assertEqual(candidate_ids.shape, (1, 2, 3))
        torch.testing.assert_close(candidate_ids[:, 0], candidate_ids[:, 1])
        self.assertIn(0, candidate_ids[0, 0].tolist())
        self.assertIn(1, candidate_ids[0, 0].tolist())

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

    def test_direction_aware_mix_separates_promotion_and_suppression(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=2,
            attention_visual_marginal=True,
            unbalanced=True,
            marginal_relaxation=0.5,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
        )
        candidate_ids = torch.tensor([[[0, 1], [0, 2]]])
        target = torch.full((1, 2, 2), 0.5)
        # Token 0 is fully retained. Tokens 1 and 2 are weak under current
        # attention but fully retained by the whole-image uniform reference.
        attention_plan = torch.tensor(
            [[[[0.5, 0.1]], [[0.5, 0.1]]]], dtype=torch.float32,
        )
        uniform_plan = torch.tensor(
            [[[[0.5, 0.5]], [[0.5, 0.5]]]], dtype=torch.float32,
        )
        details = {
            "candidate_ids": candidate_ids,
            "target_marginal": target,
            "transport_plan": attention_plan,
            "uniform_transport_plan": uniform_plan,
        }
        final = torch.zeros(1, 5)
        augmented = torch.tensor([[2.0, -2.0, -2.0, 3.0, -3.0]])
        mixed, promote, suppress = method._direction_aware_mix(
            final, augmented, torch.tensor([[0.5, 0.5]]), details, 0.3,
        )

        self.assertAlmostEqual(promote[0, 0].item(), 1.0)
        self.assertAlmostEqual(promote[0, 1].item(), 0.2)
        self.assertAlmostEqual(suppress[0, 1].item(), 0.0)
        # A globally supported peripheral candidate is protected from the
        # negative early-layer delta.
        self.assertAlmostEqual(mixed[0, 1].item(), 0.0)
        # Tokens outside the evaluated candidate union are exact final-logit
        # fallbacks, irrespective of the augmented-logit direction.
        self.assertAlmostEqual(mixed[0, 3].item(), 0.0)
        self.assertAlmostEqual(mixed[0, 4].item(), 0.0)
        self.assertLessEqual(
            (mixed - final).abs().max().item(),
            0.3 * (augmented - final).abs().max().item(),
        )

    def test_direction_aware_mix_suppresses_globally_unsupported_candidate(self):
        method = OTBarySLA(
            topk=1,
            visual_tokens=1,
            attention_visual_marginal=True,
            unbalanced=True,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
        )
        details = {
            "candidate_ids": torch.tensor([[[1]]]),
            "target_marginal": torch.ones(1, 1, 1),
            "transport_plan": torch.zeros(1, 1, 1, 1),
            "uniform_transport_plan": torch.zeros(1, 1, 1, 1),
        }
        final = torch.zeros(1, 3)
        augmented = torch.tensor([[4.0, -2.0, 4.0]])
        mixed, promote, suppress = method._direction_aware_mix(
            final, augmented, torch.ones(1, 1), details, 0.3,
        )
        self.assertEqual(promote[0, 1].item(), 0.0)
        self.assertEqual(suppress[0, 1].item(), 1.0)
        self.assertAlmostEqual(mixed[0, 1].item(), -0.6, places=6)

    def test_mass_centered_direction_gate_uses_relative_retention(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=1,
            attention_visual_marginal=True,
            unbalanced=True,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
            mass_centered_direction_gating=True,
        )
        # Retentions are [0.8, 0.2] under a uniform target, hence transported
        # mass is 0.5. The normalized positive/negative deviations are 0.6.
        details = {
            "candidate_ids": torch.tensor([[[1, 2]]]),
            "target_marginal": torch.tensor([[[0.5, 0.5]]]),
            "transport_plan": torch.tensor([[[[0.4, 0.1]]]]),
            "uniform_transport_plan": torch.tensor([[[[0.4, 0.1]]]]),
        }
        final = torch.zeros(1, 4)
        augmented = torch.tensor([[0.0, 2.0, -2.0, 0.0]])
        mixed, promote, suppress = method._direction_aware_mix(
            final, augmented, torch.ones(1, 1), details, 0.5,
        )

        self.assertAlmostEqual(promote[0, 1].item(), 0.6, places=6)
        self.assertAlmostEqual(suppress[0, 2].item(), 0.6, places=6)
        self.assertAlmostEqual(mixed[0, 1].item(), 0.6, places=6)
        self.assertAlmostEqual(mixed[0, 2].item(), -0.6, places=6)
        self.assertAlmostEqual(
            details["attention_retention_abs_deviation"].item(), 0.3,
            places=6,
        )

    def test_mass_centered_gate_preserves_unbounded_retention_before_clipping_evidence(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=1,
            attention_visual_marginal=True,
            unbalanced=True,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
            mass_centered_direction_gating=True,
        )
        details = {
            "candidate_ids": torch.tensor([[[1, 2]]]),
            "target_marginal": torch.tensor([[[0.5, 0.5]]]),
            # Total mass is 0.8, while the first retention is 0.75/0.5=1.5.
            "transport_plan": torch.tensor([[[[0.75, 0.05]]]]),
            "uniform_transport_plan": torch.tensor([[[[0.75, 0.05]]]]),
        }
        final = torch.zeros(1, 4)
        augmented = torch.tensor([[0.0, 2.0, -2.0, 0.0]])
        _, promote, suppress = method._direction_aware_mix(
            final, augmented, torch.ones(1, 1), details, 0.5,
        )

        self.assertAlmostEqual(
            details["attention_retention"][0, 0, 0].item(), 1.5,
            places=6,
        )
        self.assertAlmostEqual(
            (
                details["target_marginal"]
                * details["attention_retention"]
            ).sum().item(),
            0.8,
            places=6,
        )
        self.assertEqual(promote[0, 1].item(), 1.0)
        self.assertTrue(torch.all((promote >= 0) & (promote <= 1)))
        self.assertTrue(torch.all((suppress >= 0) & (suppress <= 1)))

    def test_mass_centered_direction_gate_removes_global_shrinkage(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=1,
            attention_visual_marginal=True,
            unbalanced=True,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
            mass_centered_direction_gating=True,
        )
        # Both candidates retain the same 0.4 fraction. This is pure global
        # UOT mass shrinkage, so centered promotion and suppression are zero.
        details = {
            "candidate_ids": torch.tensor([[[1, 2]]]),
            "target_marginal": torch.tensor([[[0.5, 0.5]]]),
            "transport_plan": torch.tensor([[[[0.2, 0.2]]]]),
            "uniform_transport_plan": torch.tensor([[[[0.2, 0.2]]]]),
        }
        final = torch.tensor([[0.0, 0.5, -0.5, 0.0]])
        augmented = torch.tensor([[0.0, 2.0, -2.0, 0.0]])
        mixed, promote, suppress = method._direction_aware_mix(
            final, augmented, torch.ones(1, 1), details, 1.0,
        )

        torch.testing.assert_close(promote, torch.zeros_like(promote))
        torch.testing.assert_close(suppress, torch.zeros_like(suppress))
        torch.testing.assert_close(mixed, final)

    def test_bidirectional_timestep_gate_keeps_positive_and_negative_evidence_separate(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=1,
            attention_visual_marginal=True,
            unbalanced=True,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
            mass_centered_direction_gating=True,
            bidirectional_timestep_gating=True,
            shared_candidate_set=True,
            final_norm_alignment=True,
        )
        # The first candidate has positive attention evidence; the second has
        # uniform whole-image absence evidence. Final logits favor the first,
        # so q+ must be much larger than q-.
        details = {
            "candidate_ids": torch.tensor([[[1, 2]]]),
            "target_marginal": torch.tensor([[[0.5, 0.5]]]),
            "transport_plan": torch.tensor([[[[0.4, 0.1]]]]),
            "uniform_transport_plan": torch.tensor([[[[0.4, 0.1]]]]),
        }
        final = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
        augmented = torch.tensor([[0.0, 4.0, -2.0, 0.0]])
        mixed, promote, suppress = method._direction_aware_mix(
            final, augmented, torch.ones(1, 1), details, 0.5,
        )

        candidate_probability = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
        expected_q_plus = 0.6 * candidate_probability[0]
        expected_q_minus = 0.6 * candidate_probability[1]
        self.assertAlmostEqual(
            details["timestep_promotion_strength"].item(),
            expected_q_plus.item(), places=6,
        )
        self.assertAlmostEqual(
            details["timestep_suppression_strength"].item(),
            expected_q_minus.item(), places=6,
        )
        self.assertGreater(
            details["timestep_promotion_strength"].item(),
            details["timestep_suppression_strength"].item(),
        )
        self.assertAlmostEqual(
            mixed[0, 1].item(), 2.0 + 0.5 * expected_q_plus.item() * 0.6 * 2.0,
            places=6,
        )
        self.assertAlmostEqual(
            mixed[0, 2].item(), -0.5 * expected_q_minus.item() * 0.6 * 2.0,
            places=6,
        )
        self.assertEqual(promote[0, 3].item(), 0.0)
        self.assertEqual(suppress[0, 3].item(), 0.0)

    def test_independent_uniform_weights_remove_attention_layer_bias(self):
        common = dict(
            topk=1,
            visual_tokens=1,
            attention_visual_marginal=True,
            unbalanced=True,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
        )
        shared = OTBarySLA(**common)
        independent = OTBarySLA(
            **common, independent_uniform_layer_weights=True,
        )
        details = {
            "candidate_ids": torch.tensor([[[1], [1]]]),
            "target_marginal": torch.ones(1, 2, 1),
            "transport_plan": torch.zeros(1, 2, 1, 1),
            # Layer zero supports the token; layer one does not.
            "uniform_transport_plan": torch.tensor(
                [[[[1.0]], [[0.0]]]], dtype=torch.float32,
            ),
            "uniform_layer_weights": torch.tensor([[0.1, 0.9]]),
        }
        final = torch.zeros(1, 3)
        augmented = torch.tensor([[0.0, -2.0, 0.0]])
        attention_weights = torch.tensor([[0.9, 0.1]])
        mixed_shared, _, suppress_shared = shared._direction_aware_mix(
            final, augmented, attention_weights, details, 0.3,
        )
        mixed_independent, _, suppress_independent = (
            independent._direction_aware_mix(
                final, augmented, attention_weights, details, 0.3,
            )
        )
        self.assertAlmostEqual(suppress_shared[0, 1].item(), 0.1, places=6)
        self.assertAlmostEqual(
            suppress_independent[0, 1].item(), 0.9, places=6,
        )
        self.assertGreater(
            abs(mixed_independent[0, 1].item()),
            abs(mixed_shared[0, 1].item()),
        )

    def test_direction_aware_aggregate_runs_end_to_end(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            epsilon=0.1,
            sinkhorn_iters=50,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.02,
            unbalanced=True,
            marginal_relaxation=0.5,
            mass_aware_layer_weights=True,
            direction_aware_gating=True,
            independent_uniform_layer_weights=True,
            mass_centered_direction_gating=True,
            bidirectional_timestep_gating=True,
            shared_candidate_set=True,
            final_norm_alignment=True,
            log_stats=True,
        )
        method.cache_visual_features(torch.randn(1, 4, 12))
        method.cache_visual_attention_positions(
            torch.tensor([[False, True, True, True, True, False]])
        )
        hidden = torch.randn(1, 6, 12)
        method.cache_layer_visual_features((hidden, hidden.clone()))
        attention = torch.ones(1, 2, 1, 6)
        final = torch.randn(1, 32)
        mixed, details = method.aggregate(
            self.early_logits[:1, :2], final, self.embedding,
            logits_alpha=0.3,
            attentions=(attention, attention.clone()),
            attention_layer_indices=(0, 1),
            output_embedding_weight=self.embedding,
        )
        self.assertEqual(mixed.shape, final.shape)
        self.assertTrue(torch.isfinite(mixed).all())
        self.assertEqual(details["promotion_gate"].shape, final.shape)
        self.assertEqual(details["suppression_gate"].shape, final.shape)
        self.assertIn("uniform_transport_plan", details)
        candidates = details["candidate_ids"].flatten().unique()
        outside = torch.ones(32, dtype=torch.bool)
        outside[candidates] = False
        torch.testing.assert_close(mixed[0, outside], final[0, outside])
        diagnostics = method.get_diagnostics()
        self.assertIn("mean_candidate_promotion_gate", diagnostics)
        self.assertIn("mean_uniform_transport_mass", diagnostics)
        self.assertIn("mean_uniform_layer_weights", diagnostics)
        self.assertIn("mean_uot_iterations", diagnostics)
        self.assertIn("mean_uniform_uot_dual_residual", diagnostics)
        self.assertIn(
            "mean_attention_retention_abs_deviation", diagnostics,
        )
        self.assertIn("mean_uniform_retention_abs_deviation", diagnostics)
        self.assertIn("mean_attention_uniform_retention_gap", diagnostics)
        self.assertIn("mean_timestep_promotion_strength", diagnostics)
        self.assertIn("mean_timestep_suppression_strength", diagnostics)

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

    def test_layer_cost_is_mean_local_transport_cost(self):
        method = OTBarySLA(topk=4, visual_tokens=9)
        method.cache_visual_features(self.visual)
        _, details = method.compute_layer_weights(
            self.early_logits,
            self.embedding,
            return_details=True,
        )
        expected = (
            details["transport_plan"][..., :-1, :]
            * (1.0 - details["similarity"][..., :-1, :])
        ).sum(dim=(-2, -1)) / details["local_transport_mass"]
        torch.testing.assert_close(details["layer_costs"], expected)
        torch.testing.assert_close(details["layer_scores"], -expected)

    def test_temperature_controls_inverse_cost_weight_sharpness(self):
        aligned = OTBarySLA(
            topk=2, visual_tokens=4, epsilon=0.05, sinkhorn_iters=20,
            layer_temperature=0.05,
        )
        smooth = OTBarySLA(
            topk=2, visual_tokens=4, epsilon=0.05, sinkhorn_iters=20,
            layer_temperature=1.0,
        )
        embedding = torch.eye(4).repeat(3, 1)
        visual = torch.zeros(1, 4, 4)
        visual[..., 0] = 1.0
        logits = torch.full((1, 2, 12), -20.0)
        logits[0, 0, 2:4] = 10.0
        logits[0, 1, 0:2] = 10.0
        for method in (aligned, smooth):
            method.cache_visual_features(visual)
        sharp_weights, sharp_details = aligned.compute_layer_weights(logits, embedding, True)
        smooth_weights = smooth.compute_layer_weights(logits, embedding)
        self.assertEqual(sharp_details["layer_costs"].argmin(dim=-1).item(), 1)
        self.assertEqual(sharp_weights.argmax(dim=-1).item(), 1)
        self.assertGreater(sharp_weights[0, 1], smooth_weights[0, 1])

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

    def test_attention_visual_marginal_has_no_dustbin_and_tracks_attention(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            epsilon=0.1,
            sinkhorn_iters=50,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.0,
        )
        visual = torch.randn(1, 4, 12)
        method.cache_visual_features(visual)
        positions = torch.tensor([[False, True, True, True, True, False]])
        method.cache_visual_attention_positions(positions)
        layer_hidden = torch.randn(1, 6, 12)
        method.cache_layer_visual_features((layer_hidden, layer_hidden.clone()))
        # Two requested layers, two heads, one query, and six key positions.
        attention = torch.zeros(1, 2, 1, 6)
        attention[..., 1] = 0.7
        attention[..., 2] = 0.2
        attention[..., 3] = 0.1
        attentions = (attention, attention.clone())
        weights, details = method.compute_layer_weights(
            self.early_logits[:1, :2],
            self.embedding,
            return_details=True,
            attentions=attentions,
            attention_layer_indices=(0, 1),
            output_embedding_weight=self.embedding,
        )

        self.assertEqual(details["transport_plan"].shape, (1, 2, 4, 2))
        self.assertEqual(details["source_marginal"].shape, (1, 2, 4))
        torch.testing.assert_close(
            details["source_marginal"][0, 0],
            torch.tensor([0.7, 0.2, 0.1, 0.0]),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            details["transport_plan"].sum(dim=(-2, -1)),
            torch.ones(1, 2),
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertTrue(torch.isfinite(weights).all())

    def test_attention_trace_records_effective_patch_mass(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            epsilon=0.1,
            sinkhorn_iters=50,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.0,
            trace_attention=True,
        )
        visual = torch.randn(1, 4, 12)
        method.cache_visual_features(visual)
        positions = torch.tensor([[False, True, True, True, True, False]])
        method.cache_visual_attention_positions(positions)
        hidden = torch.randn(1, 6, 12)
        method.cache_layer_visual_features((hidden, hidden.clone()))
        attention = torch.zeros(1, 2, 1, 6)
        attention[..., 1] = 0.7
        attention[..., 2] = 0.2
        attention[..., 3] = 0.1
        method.aggregate(
            self.early_logits[:1, :2],
            torch.randn(1, 32),
            self.embedding,
            logits_alpha=0.3,
            attentions=(attention, attention.clone()),
            attention_layer_indices=(0, 1),
            output_embedding_weight=self.embedding,
        )

        trace = method.get_diagnostics()["attention_trace"]
        self.assertEqual(len(trace), 1)
        self.assertEqual(len(trace[0]["effective_source_marginal"][0]), 4)
        self.assertAlmostEqual(
            sum(trace[0]["effective_source_marginal"][0]), 1.0, places=5,
        )

    def test_coverage_aware_marginal_reweights_previously_used_patches(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.0,
            attention_coverage_beta=0.5,
            attention_coverage_epsilon=0.1,
        )
        method.cache_visual_features(torch.randn(1, 4, 12))
        method.cache_visual_attention_positions(
            torch.tensor([[False, True, True, True, True, False]])
        )
        attention = torch.zeros(1, 1, 1, 6)
        attention[..., 1] = 0.7
        attention[..., 2] = 0.2
        attention[..., 3] = 0.1
        method._attention_coverage = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

        source = method._attention_source_marginal(
            (attention, attention.clone()), (0, 1), batch_size=1, visual_tokens=4,
        )
        self.assertLess(source[0, 0, 0].item(), 0.7)
        self.assertGreater(source[0, 0, 1].item(), 0.2)
        torch.testing.assert_close(
            source.sum(dim=-1), torch.ones(1, 2), atol=1e-6, rtol=1e-6,
        )

    def test_adaptive_alpha_shrinks_for_concentrated_visual_marginal(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            epsilon=0.1,
            sinkhorn_iters=50,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.0,
            adaptive_alpha=True,
            adaptive_alpha_min_ratio=0.25,
        )
        method.cache_visual_features(torch.randn(1, 4, 12))
        method.cache_visual_attention_positions(
            torch.tensor([[False, True, True, True, True, False]])
        )
        hidden = torch.randn(1, 6, 12)
        method.cache_layer_visual_features((hidden, hidden.clone()))
        attention = torch.zeros(1, 1, 1, 6)
        attention[..., 1] = 0.7
        attention[..., 2] = 0.2
        attention[..., 3] = 0.1
        _, details = method.aggregate(
            self.early_logits[:1, :2],
            torch.randn(1, 32),
            self.embedding,
            logits_alpha=0.3,
            attentions=(attention, attention.clone()),
            attention_layer_indices=(0, 1),
            output_embedding_weight=self.embedding,
        )
        self.assertGreaterEqual(details["adaptive_alpha"].item(), 0.3 * 0.25)
        self.assertLess(details["adaptive_alpha"].item(), 0.3)

    def test_recall_reward_only_changes_candidate_union(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            epsilon=0.1,
            sinkhorn_iters=50,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.0,
            recall_reward_lambda=0.5,
            recall_candidate_topk=2,
        )
        method.cache_visual_features(torch.randn(1, 4, 12))
        method.cache_visual_attention_positions(
            torch.tensor([[False, True, True, True, True, False]])
        )
        hidden = torch.randn(1, 6, 12)
        method.cache_layer_visual_features((hidden, hidden.clone()))
        # Make the first patch already covered, so candidate rewards depend on
        # the remaining visual evidence rather than a uniform initial state.
        method._attention_coverage = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
        attention = torch.ones(1, 1, 1, 6)
        _, details = method.aggregate(
            self.early_logits[:1, :2],
            torch.randn(1, 32),
            self.embedding,
            logits_alpha=0.3,
            attentions=(attention, attention.clone()),
            attention_layer_indices=(0, 1),
            output_embedding_weight=self.embedding,
        )

        reward = details["recall_reward"]
        candidates = details["recall_candidate_ids"]
        self.assertTrue(torch.isfinite(reward).all())
        self.assertGreater(reward.max().item(), 0.0)
        candidate_mask = torch.zeros(32, dtype=torch.bool)
        candidate_mask[candidates[0].unique()] = True
        self.assertTrue(torch.all(reward[0, ~candidate_mask] == 0))

    def test_recall_recovery_is_bounded_by_uniform_layer_reference(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            epsilon=0.1,
            sinkhorn_iters=50,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.0,
            adaptive_alpha=True,
            adaptive_alpha_min_ratio=0.2,
            recall_recovery_rho=1.0,
            recall_candidate_topk=2,
        )
        method.cache_visual_features(torch.randn(1, 4, 12))
        method.cache_visual_attention_positions(
            torch.tensor([[False, True, True, True, True, False]])
        )
        hidden = torch.randn(1, 6, 12)
        method.cache_layer_visual_features((hidden, hidden.clone()))
        method._attention_coverage = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
        attention = torch.ones(1, 1, 1, 6)
        recovered, details = method.aggregate(
            self.early_logits[:1, :2],
            torch.randn(1, 32),
            self.embedding,
            logits_alpha=0.3,
            attentions=(attention, attention.clone()),
            attention_layer_indices=(0, 1),
            output_embedding_weight=self.embedding,
        )

        before = details["pre_recovery_logits"]
        reference = details["uniform_reference_logits"]
        self.assertTrue(torch.all(recovered >= before))
        self.assertTrue(torch.all(recovered <= torch.maximum(before, reference) + 1e-6))
        self.assertTrue(torch.all(details["recall_recovery"] >= 0))
        candidates = details["recall_candidate_ids"]
        candidate_mask = torch.zeros(32, dtype=torch.bool)
        candidate_mask[candidates[0].unique()] = True
        torch.testing.assert_close(recovered[0, ~candidate_mask], before[0, ~candidate_mask])

    def test_additive_reward_and_bounded_recovery_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            OTBarySLA(
                recall_reward_lambda=0.1,
                recall_recovery_rho=0.5,
            )

    def test_attention_visual_path_never_pools_layer_tokens(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=4,
            attention_visual_marginal=True,
        )
        visual = torch.randn(1, 16, 12)
        method.cache_visual_features(visual)
        positions = torch.zeros(1, 18, dtype=torch.bool)
        positions[:, 1:17] = True
        method.cache_visual_attention_positions(positions)
        hidden = torch.randn(1, 18, 12)
        method.cache_layer_visual_features((hidden, hidden.clone()))

        self.assertEqual(method._visual_local.shape[1], 16)
        self.assertEqual(method._layer_visual_features.shape[2], 16)

    def test_attention_cost_aligns_layer_hidden_with_lm_head_rows(self):
        method = OTBarySLA(
            topk=2,
            visual_tokens=1,
            epsilon=0.1,
            sinkhorn_iters=50,
            attention_visual_marginal=True,
            attention_power=1.0,
            attention_uniform_mix=0.0,
        )
        method.cache_visual_features(torch.randn(1, 2, 2))
        positions = torch.tensor([[False, True, True]])
        method.cache_visual_attention_positions(positions)
        layer_zero = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        layer_one = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
        method.cache_layer_visual_features((layer_zero, layer_one))

        early_logits = torch.tensor([[[5.0, 5.0, -5.0, -5.0],
                                      [5.0, 5.0, -5.0, -5.0]]])
        # Deliberately oppose input embeddings and lm-head rows. Correctly
        # using the lm-head rows makes layer zero the aligned layer.
        input_embeddings = torch.tensor([[0.0, 1.0], [0.0, 1.0],
                                         [1.0, 0.0], [1.0, 0.0]])
        lm_head_rows = torch.tensor([[1.0, 0.0], [1.0, 0.0],
                                     [0.0, 1.0], [0.0, 1.0]])
        attention = torch.ones(1, 1, 1, 3)
        weights, details = method.compute_layer_weights(
            early_logits,
            input_embeddings,
            return_details=True,
            attentions=(attention, attention.clone()),
            attention_layer_indices=(0, 1),
            output_embedding_weight=lm_head_rows,
        )

        self.assertEqual(details["layer_costs"].argmin(dim=-1).item(), 0)
        self.assertEqual(weights.argmax(dim=-1).item(), 0)


if __name__ == "__main__":
    unittest.main()
