# Adaptive VSV dose-law experiment

本分支的研究问题是：VISTA 已经为每张图像提取 image-specific VSV，能否只依赖推理时可见的信息，为每个样本选择 base `lambda`。默认 VISTA 行为不变；新增路径只在显式调用 adaptive sweep 时启用。

## 固定定义

正向上下文为原图像+问题，负向上下文为同一问题但不插入图像。对每层的 prefill 最后位置保存

\[
 r_l=h^{pos}_l-h^{neg}_l,
 \qquad v_l=\text{official VSV}_l.
\]

`v_l` 由当前 `obtain_vsv()` 的 PCA 构造原样产生；实验代码同时记录 `||r_l||`、`cos(r_l,v_l)` 及其层统计，绝不把 raw difference 默默替换成官方向量。

VSV 的真实注入位置是 `Sequential(original_mlp, VSVLayer)` 的 MLP 输出。若 `x_l` 是该位置的 prefill 最后 token，则

\[
c_l=\cos(x_l,v_l),\quad
g_l=1+\max(0,-c_l),\quad
\lambda^{eff}_l=\lambda_{base}g_l
\]

是 legacy gate；`--vsv-sim-gate off` 将 `g_l=1`。默认仍为 `legacy`。

## 四种 law

1. geometry：让 `median_l theta_l(lambda)` 接近 calibration set 的 `theta_ref`，其中
\[
\theta_l=\operatorname{atan2}(a_l\sqrt{1-c_l^2},1+a_lc_l),\quad a_l=\lambda g_l.
\]
用二分求解并显式记录 upper clamp。
2. visual-deficit：以 `relative_raw_diff_norm_mean` 为无监督视觉差异特征，校准 inverse/direct 两个方向。
3. local-sensitivity：只在生成前比较 `lambda=0` 与 `epsilon=0.02` 的 first-next-token logits，记录 logit L2、JS、熵变化和 top-1 是否改变。
4. layer-consistency：记录相邻官方 VSV 的 cosine、负 cosine 比例、方向平滑度；仅作为描述性 reliability feature，不自动解释为因果变量。

每张图只做一次正/负 VSV 提取，然后复用到 lambda grid。控制器特征不读 CHAIR、COCO GT、生成 caption 或后验概率；CHAIR 只在离线 outcome 表中使用。

## 运行顺序

```bash
PYTHON_BIN=/home/sun_yuxi/anaconda3/envs/vista/bin/python \
  bash scripts/run_adaptive_vsv_8gpu.sh
```

默认生成 500 图 master list，前 300 为 calibration，后 200 为 held-out。可用 `TARGET_IDS=...` 指定单个 split，`LAMBDA_GRID=...` 扩展到 0.20/0.25。每个 GPU 独立 shard，merge 脚本拒绝重复、缺失、非法 lambda 和 malformed JSONL。

Law discovery 首先使用 `--logits-aug`/SLA 关闭的输出；最佳 law 再另行在 `logits_alpha=0.3` 的完整 VISTA 配置确认。不要把 calibration 的 CHAIR 结果用于 held-out controller 特征。

## 统计纪律

`lambda=0.17` 是 reference。应报告 calibration 与 held-out 的 image-bootstrap Spearman、MAE/RMSE、lambda 分布和实际 angular-dose 分布，并保留 EMPTY、SINGLETON、CONTIGUOUS、NONCONTIGUOUS safe sets。相关性不构成因果结论，也不允许先筛 head/layer 再做主结论。
