import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
)
from steering_vector import add_logits_flag, remove_logits_flag


def make_args(use_ot=False, force_uniform=False, attention_visual=False):
    return SimpleNamespace(
        logits_aug=True,
        logits_layers="1,2",
        logits_alpha=0.3,
        use_ot_bary_sla=use_ot,
        ot_topk=4,
        ot_visual_tokens=4,
        ot_sinkhorn_iters=3,
        ot_sinkhorn_tolerance=1e-3,
        ot_epsilon=0.05,
        ot_log_stats=False,
        ot_force_uniform=force_uniform,
        ot_attention_visual_marginal=attention_visual,
        ot_attention_power=0.5,
        ot_attention_uniform_mix=0.02,
    )


class TinyLlavaIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        config = LlavaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        self.model = LlavaLlamaForCausalLM(config).eval()
        self.input_ids = torch.randint(0, 64, (2, 5))

    def original_sla_expected(self):
        with torch.no_grad():
            base = self.model(
                input_ids=self.input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = base.hidden_states[1:]
        early = torch.stack(
            [self.model.lm_head(hidden_states[idx]) for idx in (1, 2)]
        ).mean(dim=0)
        return 0.3 * early + 0.7 * base.logits, base.logits

    def test_disabled_ot_path_exactly_matches_original_sla(self):
        expected, _ = self.original_sla_expected()
        add_logits_flag(self.model, make_args(use_ot=False))
        try:
            with torch.no_grad():
                actual = self.model(
                    input_ids=self.input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                ).logits
        finally:
            diagnostics = remove_logits_flag(self.model)

        self.assertEqual(diagnostics, {})
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_forced_uniform_matches_original_current_token(self):
        expected, base_logits = self.original_sla_expected()
        add_logits_flag(
            self.model,
            make_args(use_ot=True, force_uniform=True),
        )
        self.model.ot_bary_sla.cache_visual_features(
            torch.randn(2, 4, self.model.config.hidden_size)
        )
        try:
            with torch.no_grad():
                actual = self.model(
                    input_ids=self.input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                ).logits
        finally:
            remove_logits_flag(self.model)

        torch.testing.assert_close(
            actual[:, -1, :],
            expected[:, -1, :],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            actual[:, :-1, :],
            base_logits[:, :-1, :],
            atol=0,
            rtol=0,
        )

    def test_ot_state_is_removed_after_generation(self):
        add_logits_flag(self.model, make_args(use_ot=True))
        self.assertTrue(hasattr(self.model, "ot_bary_sla"))
        remove_logits_flag(self.model)
        self.assertFalse(hasattr(self.model, "ot_bary_sla"))
        self.assertFalse(hasattr(self.model, "use_ot_bary_sla"))

    def test_projected_visual_tokens_are_cached_once_at_prefill(self):
        class FakeVisionTower(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.calls = 0

            def forward(self, images):
                self.calls += 1
                batch_size = images.shape[0]
                base = torch.arange(
                    4 * self.hidden_size,
                    device=images.device,
                    dtype=images.dtype,
                ).reshape(1, 4, self.hidden_size)
                return base.repeat(batch_size, 1, 1)

        vision_tower = FakeVisionTower(self.model.config.hidden_size)
        self.model.model.vision_tower = vision_tower
        self.model.model.mm_projector = nn.Identity()
        multimodal_ids = torch.tensor(
            [
                [1, 2, IMAGE_TOKEN_INDEX, 3, 4],
                [1, 5, IMAGE_TOKEN_INDEX, 6, 7],
            ]
        )
        images = torch.randn(2, 3, 2, 2)

        add_logits_flag(self.model, make_args(use_ot=True))
        try:
            with torch.no_grad():
                prefill = self.model(
                    input_ids=multimodal_ids,
                    images=images,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            self.assertEqual(vision_tower.calls, 1)
            self.assertTrue(self.model.ot_bary_sla.has_visual_cache)

            past_length = prefill.past_key_values[-1][-1].shape[-2]
            with torch.no_grad():
                self.model(
                    input_ids=torch.tensor([[8], [9]]),
                    images=images,
                    past_key_values=prefill.past_key_values,
                    attention_mask=torch.ones(2, past_length + 1),
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            self.assertEqual(vision_tower.calls, 1)
        finally:
            remove_logits_flag(self.model)

    def test_greedy_and_nucleus_generation_smoke(self):
        class FakeVisionTower(nn.Module):
            def forward(self, images):
                values = torch.arange(
                    4 * self_hidden_size,
                    device=images.device,
                    dtype=images.dtype,
                )
                return values.reshape(1, 4, self_hidden_size).repeat(
                    images.shape[0],
                    1,
                    1,
                )

        self_hidden_size = self.model.config.hidden_size
        self.model.model.vision_tower = FakeVisionTower()
        self.model.model.mm_projector = nn.Identity()
        input_ids = torch.tensor([[1, 10, IMAGE_TOKEN_INDEX, 11]])
        images = torch.randn(1, 3, 2, 2)

        for do_sample in (False, True):
            with self.subTest(do_sample=do_sample):
                torch.manual_seed(12)
                add_logits_flag(self.model, make_args(use_ot=True))
                try:
                    sampling_args = (
                        {"top_p": 0.9, "temperature": 0.7}
                        if do_sample
                        else {}
                    )
                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            images=images,
                            max_new_tokens=2,
                            do_sample=do_sample,
                            output_hidden_states=True,
                            use_cache=True,
                            pad_token_id=0,
                            **sampling_args,
                        )
                finally:
                    remove_logits_flag(self.model)
                self.assertEqual(output.shape[0], 1)
                self.assertGreater(output.shape[1], input_ids.shape[1])

    def test_attention_visual_ot_caches_positions_and_uses_decoder_attention(self):
        class FakeVisionTower(nn.Module):
            def forward(self, images):
                values = torch.arange(
                    4 * self_hidden_size,
                    device=images.device,
                    dtype=images.dtype,
                )
                return values.reshape(1, 4, self_hidden_size).repeat(
                    images.shape[0], 1, 1,
                )

        self_hidden_size = self.model.config.hidden_size
        self.model.model.vision_tower = FakeVisionTower()
        self.model.model.mm_projector = nn.Identity()
        input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 10, 11]])
        images = torch.randn(1, 3, 2, 2)
        add_logits_flag(
            self.model, make_args(use_ot=True, attention_visual=True),
        )
        try:
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    images=images,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            self.assertIsNotNone(output.attentions)
            self.assertEqual(
                self.model.ot_bary_sla._visual_attention_positions.sum().item(),
                4,
            )
            self.assertEqual(self.model.ot_bary_sla._visual_extended.shape[1], 4)
            self.assertEqual(
                self.model.ot_bary_sla._layer_visual_features.shape,
                (1, 2, 4, self.model.config.hidden_size),
            )
            cached_visual = self.model.ot_bary_sla._layer_visual_features.clone()
            past_length = output.past_key_values[-1][-1].shape[-2]
            with torch.no_grad():
                self.model(
                    input_ids=torch.tensor([[12]]),
                    images=images,
                    past_key_values=output.past_key_values,
                    attention_mask=torch.ones(1, past_length + 1),
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            torch.testing.assert_close(
                self.model.ot_bary_sla._layer_visual_features,
                cached_visual,
            )
        finally:
            remove_logits_flag(self.model)


if __name__ == "__main__":
    unittest.main()
