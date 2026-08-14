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
average node only on the visual side. The visual marginal remains uniform over
the pooled visual tokens and their average dustbin, while the text marginal is
the softmax distribution of that layer's selected top-M logits. There is no
text dustbin. The extended problem is solved with float32 log-domain Sinkhorn
iterations, and only the non-dustbin visual rows contribute to the layer score.
New result filenames include `otbary_vdust_tlogit` so they cannot be confused
with results produced by the previous two-dustbin, uniform-marginal method.

The normalized local OT costs become a distribution over early layers.  The
global visual node is excluded from the cost average, so layers are compared
by local visual-text matching quality rather than how much mass reaches that
node:

```text
layer_cost[l] = sum(P_local[l] * cost_local[l]) / sum(P_local[l])
layer_weights = softmax(-layer_cost / ot_layer_temperature)
augmented_logits = sum(layer_weights * early_logits)
final_logits = (1 - logits_alpha) * final_logits + logits_alpha * augmented_logits
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
logits_alpha         = 0.3
logits_layers        = 25,30
```

## Attention visual marginal (no dustbin)

The `--ot-attention-visual-marginal` variant leaves OT2 intact and instead
uses the current decoder query's attention to the original image-patch
positions as its visual OT marginal. It keeps every original image token and
does not pool either visual hidden states or attention weights. During
multimodal prefill it caches the image-position hidden states at each selected
SLA layer. Its layer-specific cost is

```text
C[l,k,m] = 1 - cosine(layer_visual_hidden[l,k], lm_head.weight[token[l,m]])
```

so the visual representation, current-query attention, candidate logits, and
output token directions are aligned to the same layer. There is no global
visual node or dustbin in this variant.

```text
ot_topk                  = 16
visual tokens            = all original image tokens (unpooled)
ot_epsilon               = 0.05
ot_sinkhorn_iters        = 50 (maximum)
ot_sinkhorn_tolerance    = 0.001
ot_layer_temperature     = 0.2
ot_attention_power       = 0.5
ot_attention_uniform_mix = 0.02
logits_alpha             = 0.5
```

`ot_attention_power=0.5` tempers an excessively peaked raw attention map;
the small `uniform_mix` is only numerical smoothing over real patches, not a
dustbin. Sinkhorn checks both marginal residuals every five iterations and
stops early once their maximum is at most the configured tolerance. A matched
seed-2024 CHAIR baseline/attention-OT run on GPUs 6 and 7 is provided by:

```bash
PYTHON_BIN=/home/ljc/miniconda3/envs/formodelling-gpu/bin/python \
bash run_chair_ot_attention_seed2024.sh
```

For an eight-GPU paired search, use the VISTA-aligned attention-OT preset. It
runs one VISTA baseline per `(seed, logits_alpha)` pair, then searches 30 attention
configurations by default with `vsv_lambda=0.17`, fixed `logits_alpha=0.3`,
and `uniform_mix=0.02`:
`layer_temperature={0.03,0.06,0.1,0.2,0.4,0.8}` and
`attention_power={0.25,0.5,0.75,1.0,1.5}`. All configurations use unpooled
visual tokens, `topk=16`, `epsilon=0.05`, at most 50 Sinkhorn iterations, and
tolerance `1e-3`.

```bash
GPU_IDS="0 1 2 3 4 5 6 7" \
bash run_chair_ot_attention_vista_preset.sh
```

The runner writes a paired `summary.csv`, layer-weight diagnostics, and a
compact Markdown report under `exp_results/chair_ot_attention_grid/`.

After the broad seed-2024 search, the five-seed refinement around the useful
region (`layer_temperature={0.06,0.1}`, `attention_power={0.75,1.0}`) is
available without changing the original VISTA parameters:

```bash
bash run_chair_ot_attention_multiseed_refine.sh
```

To search VISTA's original `logits_alpha` jointly for a matched original-VISTA
baseline and the fixed attention-OT setting (`layer_temperature=0.06`,
`attention_power=0.75`, `uniform_mix=0.02`, `topk=16`, `epsilon=0.05`), run:

```bash
bash run_chair_ot_attention_alpha_multiseed.sh
```

It evaluates `logits_alpha={0.15,0.20,0.25,0.30,0.35}` for both methods on
the same five seed-specific image subsets, so every reported delta is paired.

### Attention coverage diagnosis

To test whether attention-OT concentrates mass on salient objects while
under-covering other ground-truth COCO objects, run a single-image trace:

```bash
bash run_chair_ot_attention_trace.sh COCO_IMAGE_ID
```

The run writes `coverage.md` and `coverage.json` below
`exp_results/attention_ot_trace_COCO_IMAGE_ID/`. For each COCO box, the report
contains its effective attention mass, formed by combining each layer's visual
OT marginal with that step's OT layer weights. `enrichment_vs_uniform < 1`
means the object box receives less mass than a uniform distribution over image
patches would assign. This diagnoses visual-mass concentration, not whether a
caption explicitly names the object; inspect the paired generated caption too.

### Coverage and adaptive-alpha ablation

The coverage-aware marginal modestly downweights patches that have already
received OT mass in earlier generation steps. Adaptive alpha scales the base
`logits_alpha` by the entropy of the current effective visual marginal, so a
very concentrated visual focus receives a smaller OT intervention. To isolate
the two effects, this eight-GPU runner evaluates original VISTA, plain
attention-OT, coverage-only, and adaptive-alpha-only variants on three paired
seeds with the same fixed attention-OT setup:

```bash
bash run_chair_ot_attention_module_ablation.sh
```

It searches `coverage_beta={0.1,0.25,0.5}` and
`adaptive_alpha_min_ratio={0,0.25,0.5}`, while keeping
`logits_alpha=0.3`, `layer_temperature=0.06`, `attention_power=0.75`,
`uniform_mix=0.02`, `topk=16`, and `epsilon=0.05`. Results are written to
`exp_results/chair_ot_attention_module_ablation/summary.{csv,md}`.

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

Run the default six-GPU refinement for the new marginal design with fixed
`logits_alpha=0.3`, `lambda=0.17`, `topk={8,10,12,14,16,18,20,24,32}`, and
`visual_tokens={49,64,81}`:

```bash
SUBSET_IDS_FILE=/data/sun_yuxi/datasets/coco/splits/chair_seed1994_500.txt \
bash run_chair_ot_topk_visual_sweep.sh
```

Legacy two-dustbin results are not reused because the new filename contains
`otbary_vdust_tlogit`; the default refinement therefore runs all 27
configurations. Complete results from this new method are still skipped before
GPU assignment. CHAIR metrics and logs are written under
`exp_results/chair_ot_topk_visual_vdust_tlogit_refine_sweep/`. Override the grid
with, for example,
`TOPKS="12 16 20" VISUAL_TOKENS="49 64" GPU_IDS="0 1"`.

Test larger visual grids with `topk={8,16,32,64}` and
`visual_tokens={100,196,324,576}`:

```bash
SUBSET_IDS_FILE=/data/sun_yuxi/datasets/coco/splits/chair_seed1994_500.txt \
GPU_IDS="0 1 2 3 4 5" \
bash run_chair_ot_large_visual_sweep.sh
```

This launches 16 configurations using the same pending-only scheduler.
Completed results are reused on rerun. Logs, the manifest, and the automatic
CHAIR CSV/Markdown summaries are isolated under
`exp_results/chair_ot_large_visual_vdust_tlogit_sweep/`.

Run the default paired multi-seed grid for the visual-dustbin/text-logit
method with `seeds={1994,2024,3407,42,1234}`, `topk={4,16,32}`, and
`visual_tokens={16,64,81}`:

```bash
GPU_IDS="0 1 2 3 4 5" bash run_chair_ot_multiseed_grid.sh
```

This evaluates 45 OT runs and reuses complete VISTA baselines when available.
For every seed, the baseline and all nine OT configurations share the same
ordered 500-image manifest. The summarizer verifies the image IDs before
reporting paired deltas. Outputs are written to:

```text
exp_results/chair_ot_multiseed_grid_vdust_tlogit/per_seed.csv
exp_results/chair_ot_multiseed_grid_vdust_tlogit/aggregate.csv
exp_results/chair_ot_multiseed_grid_vdust_tlogit/summary.md
```

Override the grid without editing the script, for example:

```bash
SEEDS="1994 2024" TOPKS="8 16" VISUAL_TOKENS="36 64" \
GPU_IDS="0 1 2 3" bash run_chair_ot_multiseed_grid.sh
```

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
