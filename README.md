# VISTA: Visual Information Steering with Token-logit Augmentation

[![arXiv](https://img.shields.io/badge/arXiv-2502.03628-b31b1b.svg)](https://arxiv.org/pdf/2502.03628)

This is the official implementation of the paper "The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering".

<img src="assets/overview.png" width="600"/>

## Overview

VISTA is a training-free inference-time intervention framework that reduces hallucination in Large Vision-Language Models (LVLMs) while promoting genuine information. Our approach reveals and addresses three key patterns in how LVLMs process information:

1. **Gradual Visual Information Loss**: Visually grounded tokens gradually become less favored throughout generation
2. **Early Excitation**: Semantically meaningful tokens achieve peak activation in layers earlier than the final layer
3. **Hidden Genuine Information**: Visually grounded tokens maintain relatively high rankings at inference

VISTA combines two complementary approaches:
- **Visual Steering Vector (VSV)**: Reinforces visual information in activation space
- **Self-Logits Augmentation (SLA)**: Leverages early layer activations to promote semantically meaningful decoding

## Key Features

- Training-free inference-time intervention
- No external supervision required
- Compatible with various decoding strategies
- Applicable across multiple LVLM architectures
- Reduces hallucination by ~40% on evaluated open-ended tasks

## Installation

```bash
# Git LFS is required for the bundled MMHal-Bench assets.
git lfs install

# Clone this V100-configured repository.
git clone https://github.com/alpha-godzilla/vista-v100.git
cd vista-v100
git lfs pull

# Create and activate the virtual environment
conda env create -f environment.yml
```

For the V100 server, install the CUDA 11.8 PyTorch build before the
remaining Python dependencies:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Prepare Data
Download MSCOCO 2014 dataset from [the official website](https://cocodataset.org/#home) and extract it to your data directory.

### V100 server paths

This fork defaults to the following server layout:

```text
/home/sun_yuxi/luo_jiacheng/VISTA
/data/sun_yuxi/models/llava-1.5-7b-hf
/data/sun_yuxi/datasets/coco/{train2014,val2014,annotations}
```

The defaults can be changed without editing the code:

```bash
export VISTA_LLAVA_MODEL_PATH=/path/to/llava-1.5-7b-hf
export VISTA_COCO_ROOT=/path/to/coco
```


## Usage

```bash
# For CHAIR evaluation.
bash run_chair.sh

# Sweep CHAIR gamma/lambda settings on six GPUs and summarize the metrics.
bash run_chair_sweep.sh

# Run the optional OT-BarySLA extension.
bash run_chair_ot_bary.sh

# Refine OT top-k and visual-token counts around the current best point.
bash run_chair_ot_topk_visual_sweep.sh

# Sweep larger OT visual grids (100/196/324/576) and top-k (8/16/32/64).
bash run_chair_ot_large_visual_sweep.sh

# For POPE evaluation (specify split with --pope-type).
bash run_pope.sh

# For mmhal evaluation.
bash run_mmhal.sh
```

Please check the corresponding bash script for how to read results.

### OT-BarySLA

This copy includes an optional, training-free OT-BarySLA implementation for
LLaVA-1.5. It dynamically weights VISTA's selected early-layer logits using a
small OMIT-style optimal-transport problem over projected visual tokens and
top-ranked token embeddings. The original VISTA path remains the default and
is unchanged when `--use-ot-bary-sla` is absent.

See [OT_BARY_SLA.md](OT_BARY_SLA.md) for the method, commands, tests, and
benchmark procedure.

The CHAIR sweep defaults to gamma values `0.1 0.2 0.3 0.4`, lambda
values `0.13 0.14 0.15 0.16 0.17 0.18`, and GPUs `0 1 2 3 4 5`.
Here gamma refers to VISTA's `--logits-alpha`. The 24 runs are distributed
round-robin across the six GPUs. Completed 500-caption JSONL files are reused,
partial files are preserved with a timestamp, and the final CHAIR metrics are
written to:

```text
exp_results/chair_sweep_gamma_lambda/chair_sweep_summary.md
exp_results/chair_sweep_gamma_lambda/chair_sweep_summary.csv
```

The defaults can be overridden with environment variables:

```bash
GPU_IDS="0 1 2 3 4 5" SUBSET_SIZE=500 bash run_chair_sweep.sh
```

### Configuration Options

1. **Model Selection**: Use "--model" to specify the target LVLM (supported: "llava-1.5", "instructblip", "shikra", "minigpt-4")

2. **Visual Steering Vector (VSV )**: Enable with "--vsv" and control strength via "--vsv-lambda"

3. **Self-Logits Augmentation (SLA)**: Enable with "--logits-aug", configure target layers with "--logits-layers" and mixing ratio with "--logits-alpha"


## Best Practices

1. VSV is designed to counteract Gradual Visual Information Loss and is suitable for open-ended generation tasks. Different LVLMs favor different lambda scales, so users should calibrate the scale when using new architectures. The --vsv-lambda parameter provides a flexible way to adjust the model from being more aggressive (more hallucination) to more conservative.
3. The impact of SLA depends on both target layers and the strength of --logits-alpha. A rule of thumb is to use a smaller alpha for larger window sizes and vice versa (see Table 4 in the paper).


## Citation
```bibtex
@misc{li2025hiddenlifetokensreducing,
      title={The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering}, 
      author={Zhuowei Li and Haizhou Shi and Yunhe Gao and Di Liu and Zhenting Wang and Yuxiao Chen and Ting Liu and Long Zhao and Hao Wang and Dimitris N. Metaxas},
      year={2025},
      eprint={2502.03628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.03628}, 
}
```

## Acknowledgement
This project builds upon the following excellent works:
- [PAI](https://github.com/LALBJ/PAI)
- [ICV](https://github.com/shengliu66/ICV)
- [OPERA](https://github.com/shikiw/OPERA)
