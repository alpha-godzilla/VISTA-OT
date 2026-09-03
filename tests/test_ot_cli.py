import unittest
from types import SimpleNamespace

import myutils


def common_args():
    return SimpleNamespace(
        model="llava-1.5",
        seed=1994,
        vsv=True,
        vsv_lambda=0.17,
        layers=None,
        logits_aug=True,
        logits_layers="26,30",
        logits_alpha=0.3,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=None,
        max_new_tokens=512,
    )


class OTBaryCLICompatibilityTests(unittest.TestCase):
    def test_original_filename_is_unchanged_without_ot_attributes(self):
        args = common_args()
        actual = "_".join(myutils.prepare_common_fileparts(args))
        self.assertEqual(
            actual,
            "seed1994_vsv_lambda_0.17_logaug_loglayer_26,30_"
            "logalpha_0.3_greedy_max_new_tokens_512",
        )

    def test_ot_filename_contains_method_configuration(self):
        args = common_args()
        args.use_ot_bary_sla = True
        args.ot_topk = 8
        args.ot_visual_tokens = 36
        args.ot_sinkhorn_iters = 3
        args.ot_epsilon = 0.05
        args.ot_layer_temperature = 0.1
        args.ot_force_uniform = False

        actual = "_".join(myutils.prepare_common_fileparts(args))

        self.assertIn("otbary_vdust_tlogit_m8_k36_it3_eps0.05_ltemp0.1", actual)

    def test_ot_rejects_non_llava_model(self):
        args = common_args()
        args.use_ot_bary_sla = True
        args.model = "minigpt4"
        with self.assertRaisesRegex(ValueError, "only --model llava-1.5"):
            myutils.validate_ot_bary_sla_arguments(args)

    def test_attention_ot_filename_describes_unpooled_aligned_method(self):
        args = common_args()
        args.use_ot_bary_sla = True
        args.ot_attention_visual_marginal = True
        args.ot_topk = 16
        args.ot_sinkhorn_iters = 50
        args.ot_sinkhorn_tolerance = 1e-3
        args.ot_epsilon = 0.05
        args.ot_layer_temperature = 0.2
        args.ot_attention_power = 0.5
        args.ot_attention_uniform_mix = 0.02
        args.ot_force_uniform = False

        actual = "_".join(myutils.prepare_common_fileparts(args))
        self.assertIn("otattn_nodust_layerhid_lmhead_tlogit", actual)
        self.assertIn("m16_kunpooled_it50_tol0.001", actual)
        self.assertNotIn("vdust", actual)

    def test_mass_centered_direction_gate_has_distinct_filename_tag(self):
        args = common_args()
        args.use_ot_bary_sla = True
        args.ot_attention_visual_marginal = True
        args.ot_topk = 16
        args.ot_sinkhorn_iters = 100
        args.ot_sinkhorn_tolerance = 1e-3
        args.ot_epsilon = 0.05
        args.ot_layer_temperature = 0.06
        args.ot_attention_power = 0.75
        args.ot_attention_uniform_mix = 0.02
        args.ot_force_uniform = False
        args.ot_unbalanced = True
        args.ot_marginal_relaxation = 0.7
        args.ot_mass_aware_layer_weights = True
        args.ot_direction_aware_gating = True
        args.ot_independent_uniform_layer_weights = True
        args.ot_mass_centered_direction_gating = True
        args.ot_bidirectional_timestep_gate = True
        args.ot_shared_candidate_set = True
        args.ot_final_norm_alignment = True

        actual = "_".join(myutils.prepare_common_fileparts(args))
        self.assertIn(
            "uot_mrel0.7_masslayer_dirgate_induni_mc_"
            "sc_fn_tg", actual,
        )
        self.assertLessEqual(len(actual + "_ot_stats.jsonl"), 255)


if __name__ == "__main__":
    unittest.main()
