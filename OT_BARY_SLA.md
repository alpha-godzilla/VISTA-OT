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
layer_weights = softmax(layer_scores)
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
logits_alpha (gamma) = 0.3
logits_layers        = 25,30
```

The repository's historical `25,30` setting is an inclusive six-layer range.
The OT runner retains it so that original SLA and adaptive OT weighting differ
only in their aggregation rule. The paper's five-layer setting can be selected
explicitly with `LOGITS_LAYERS=26,30`.

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

For reproducible comparisons, export the image IDs from an existing 500-row
CHAIR result and reuse the resulting manifest:

```bash
python scripts/export_chair_image_ids.py \
  exp_results/chair_eval/llava-1.5/seed1994_example.jsonl \
  /data/sun_yuxi/datasets/coco/splits/chair_seed1994_500.txt

SUBSET_IDS_FILE=/data/sun_yuxi/datasets/coco/splits/chair_seed1994_500.txt \
SEED=1994 MODE=ot CUDA_VISIBLE_DEVICES=0 \
bash run_chair_ot_bary.sh
```

The manifest must contain one unique integer COCO image ID per line. Its order
is preserved, and random `--subset-size` sampling is disabled when it is set.

Run the default six-GPU OT grid with fixed `gamma=0.3`, `lambda=0.17`,
`topk={4,8,16,32}`, and `visual_tokens={16,36,64}`:

```bash
SUBSET_IDS_FILE=/data/sun_yuxi/datasets/coco/splits/chair_seed1994_500.txt \
bash run_chair_ot_topk_visual_sweep.sh
```

The 12 configurations are assigned round-robin to GPUs `0 1 2 3 4 5`.
Completed 500-caption results are reused. CHAIR metrics and logs are written
under `exp_results/chair_ot_topk_visual_sweep/`. Override the grid with, for
example, `TOPKS="4 8" VISUAL_TOKENS="16 36" GPU_IDS="0 1"`.

Override OT parameters through environment variables:

```bash
OT_TOPK=16 \
OT_VISUAL_TOKENS=36 \
OT_SINKHORN_ITERS=5 \
OT_EPSILON=0.1 \
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
