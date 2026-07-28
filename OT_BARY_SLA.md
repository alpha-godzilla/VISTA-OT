# OT-BarySLA

OT-BarySLA replaces only VISTA SLA's uniform early-layer average with
image-conditioned optimal-transport weights. VSV construction, VSV injection,
the fixed SLA coefficient, decoding, prompts, and model weights are unchanged.

## Method

For the early layers selected by the inclusive `--logits-layers START,END`
range, the original method uses:

```text
augmented_logits = mean(early_logits, layer)
```

OT-BarySLA uses projected LLaVA image tokens and each layer's top-M token
embeddings to construct a small cosine-cost OT problem. It appends one global
visual node and one global textual node, uses uniform marginals, and solves the
extended problem with float32 log-domain Sinkhorn iterations. Only the
local-to-local transport block contributes to the layer score.

The scores become a distribution over early layers:

```text
layer_weights = softmax(layer_scores / layer_temperature)
augmented_logits = sum(layer_weights * early_logits)
final_logits = (1 - gamma) * final_logits + gamma * augmented_logits
```

This weighted logit mixture is the closed-form reverse-KL barycenter. It does
not require an iterative KL-barycenter solver or an additional Transformer
forward pass.

## Defaults

```text
ot_topk              = 8
ot_visual_tokens     = 36
ot_sinkhorn_iters    = 3
ot_epsilon           = 0.05
ot_layer_temperature = 0.1
logits_alpha (gamma) = 0.3
logits_layers        = 26,30
```

The repository's historical `25,30` setting is an inclusive six-layer range.
It remains unchanged for backward compatibility. The OT runner explicitly
uses `26,30`, an inclusive five-layer range.

## Run

Set the paths when they differ from the server defaults:

```bash
export VISTA_LLAVA_MODEL_PATH=/data/sun_yuxi/models/llava-v1.5-7b
export VISTA_COCO_ROOT=/data/sun_yuxi/datasets/coco
export NLTK_DATA=/data/sun_yuxi/nltk_data
```

Run one-image smoke tests:

```bash
SUBSET_SIZE=1 MODE=original bash run_chair_ot_bary.sh
SUBSET_SIZE=1 MODE=uniform  bash run_chair_ot_bary.sh
SUBSET_SIZE=1 MODE=ot       bash run_chair_ot_bary.sh
```

Run the full 500-image OT evaluation:

```bash
MODE=ot SUBSET_SIZE=500 CUDA_VISIBLE_DEVICES=0 bash run_chair_ot_bary.sh
```

Override OT parameters through environment variables:

```bash
OT_TOPK=16 \
OT_VISUAL_TOKENS=36 \
OT_SINKHORN_ITERS=5 \
OT_EPSILON=0.1 \
OT_LAYER_TEMPERATURE=0.2 \
bash run_chair_ot_bary.sh
```

When `OT_LOG_STATS=1` (the OT runner default), detached generation summaries
are written beside the caption JSONL with an `_ot_stats.jsonl` suffix.

## Tests

Use a Python environment containing PyTorch and Transformers:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover Sinkhorn validity, layer-weight normalization, uniform
fallback, aligned-layer preference, special-token filtering, mixed precision,
disabled-path regression, and tiny-LLaVA integration.

## Benchmark

```bash
python scripts/benchmark_ot_bary_sla.py \
  --modes original ot uniform \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeats 3 \
  --output exp_results/ot_bary_benchmark.json
```

The report includes milliseconds per token, tokens per second, peak allocated
GPU memory, and latency overhead relative to original VISTA.
