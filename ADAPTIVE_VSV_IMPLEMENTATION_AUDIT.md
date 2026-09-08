# Adaptive VSV implementation audit

This branch preserves the existing VISTA path and adds an opt-in adaptive-dose
measurement path. The official extraction is in `steering_vector.py`:
`get_hiddenstates()` records the final prefill/context position from every
layer, and `obtain_vsv()` applies the repository PCA construction. With the
single demonstration used by `chair_eval.py`, the centered PCA has a zero
component and the returned vector reduces numerically to the raw positive-minus-
negative difference; this is diagnosed, not substituted. `obtain_vsv_with_diagnostics`
returns both quantities and per-layer comparison is emitted by the sweep.

The intervention is `layer.mlp = Sequential(original_mlp, VSVLayer(...))` in
`llm_layers.py`; therefore `x_mlp_last_l` is captured immediately before VSV
addition, not from the block residual stream. The legacy gate is
`1 + max(0, cosine(x, -vsv))`; `--vsv-sim-gate off` sets it to one. The default
remains `legacy`, and the effective dose is always `base_lambda * lambda_sim`.

LLaVA generation remains greedy, batch size one, FP16-compatible, and uses the
existing model loader. The adaptive sweep computes positive/negative states and
the VSV once per image, then reuses it across the configurable lambda grid.
Inference-time features contain no CHAIR/COCO labels. Outcome scoring and
oracle dose construction are offline only.

The official baseline-equivalence and GPU smoke checks must be run in the
remote `vista` environment before a 500-image sweep. This checkout does not
contain model weights or generated results.
