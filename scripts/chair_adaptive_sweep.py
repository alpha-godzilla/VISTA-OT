#!/usr/bin/env python3
"""Inference-only VSV dose sweep, reusing one VSV extraction per image.

This file intentionally does not import CHAIR or COCO annotations.  It emits
captions and inference-time features; offline outcome scoring is separate.
"""
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import myutils
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from steering_vector import obtain_vsv_with_diagnostics
from llm_layers import add_vsv_layers, remove_vsv_layers
from adaptive_vsv import (build_vsv_features, summarize_angles, layer_consistency,
                          steering_angle, solve_lambda_for_angle, law_dose)

DEFAULT_GRID = (0.00, 0.05, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18)

def args():
    p=argparse.ArgumentParser(); p.add_argument("--image-id-file",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--data-path",type=Path,default=Path("/data/sun_yuxi/datasets/coco/val2014")); p.add_argument("--model",default="llava-1.5")
    p.add_argument("--lambda-grid",default=",".join(map(str,DEFAULT_GRID))); p.add_argument("--vsv-lambda",type=float,default=.17)
    p.add_argument("--vsv-sim-gate",choices=("legacy","off"),default="legacy"); p.add_argument("--layers",default=None)
    p.add_argument("--controller",choices=("fixed","geometry","visual_deficit","sensitivity","consistency"),default="fixed")
    p.add_argument("--geometry-target",type=float,default=None,help="Calibration target angle in radians for geometry controller")
    p.add_argument("--law-scale",type=float,default=.17); p.add_argument("--law-feature",default="relative_raw_diff_norm_mean")
    p.add_argument("--max-new-tokens",type=int,default=128); p.add_argument("--seed",type=int,default=1994); p.add_argument("--resume",action="store_true")
    p.add_argument("--num-shards",type=int,default=1); p.add_argument("--shard-id",type=int,default=0); p.add_argument("--local-sensitivity-epsilon",type=float,default=.02)
    return p.parse_args()

def ids(path, n, shard):
    all_ids=[int(x.strip()) for x in path.read_text().splitlines() if x.strip()]
    return all_ids[shard::n]

def js(p, q):
    p=p.float().softmax(-1); q=q.float().softmax(-1); m=(p+q)/2
    return float((.5*(p*(p.log()-m.log())).sum()+.5*(q*(q.log()-m.log())).sum()).cpu())

def main():
    a=args(); lambdas=[float(x) for x in a.lambda_grid.split(",")]; selected=ids(a.image_id_file,a.num_shards,a.shard_id)
    a.output.parent.mkdir(parents=True,exist_ok=True); done=set()
    if a.resume and a.output.exists():
        for line in a.output.read_text().splitlines():
            try:
                row=json.loads(line); done.add((int(row["image_id"]),float(row["base_lambda"])))
            except (ValueError,KeyError,json.JSONDecodeError): pass
    disable_torch_init(); loader=ModelLoader(a.model); model=loader.llm_model; template=myutils.prepare_template(a)
    out=a.output.open("a",encoding="utf-8")
    with torch.inference_mode():
      for image_id in selected:
        pending=[lam for lam in lambdas if (image_id,lam) not in done]
        if not pending: continue
        image=loader.image_processor(Image.open(a.data_path/f"COCO_val2014_{image_id:012d}.jpg").convert("RGB"))
        prompt=["Please help me describe the image in detail."]
        # The negative prompt helper must receive the fully formatted
        # question, exactly as in chair_eval.py (not the raw user text).
        questions, pos_kwargs=loader.prepare_inputs_for_model(template,prompt,image)
        neg_kwargs=loader.prepare_neg_prompt(a, questions, template=template)
        vsv, neg, pos, raw=obtain_vsv_with_diagnostics(a,model,[[neg_kwargs,pos_kwargs]],rank=1)
        feature=build_vsv_features(torch.stack([neg,pos]),vsv)
        diagnostic_sink={}; cos_layers=[]
        for layer in range(vsv.shape[0]):
            # x_mlp is captured during the first wrapped prefill below.
            cos_layers.append(float("nan"))
        # Pre-generation local sensitivity, VSV-only and label-free.
        z0=model(**pos_kwargs,use_cache=True,return_dict=True).logits[0,-1].float()
        add_vsv_layers(model,torch.stack([vsv],dim=1).cuda(),[a.local_sensitivity_epsilon],a.layers,sim_gate=a.vsv_sim_gate,diagnostic_sink=diagnostic_sink)
        z_eps=model(**pos_kwargs,use_cache=True,return_dict=True).logits[0,-1].float(); remove_vsv_layers(model)
        delta=z_eps-z0; sensitivity={"delta_logit_l2":float(delta.norm().cpu()/a.local_sensitivity_epsilon),"cosine_logit_change":float(F.cosine_similarity(delta,z0,dim=0).cpu()),"js":js(z0,z_eps),"entropy_0":float(torch.distributions.Categorical(logits=z0).entropy().cpu()),"entropy_eps":float(torch.distributions.Categorical(logits=z_eps).entropy().cpu()),"top1_changed":bool(z0.argmax()!=z_eps.argmax())}
        geo_cos=torch.stack([F.cosine_similarity(pos[i],vsv[i],dim=0) for i in range(vsv.shape[0])])
        base_features=feature.to_dict(); base_features.update(sensitivity); base_features.update(layer_consistency(vsv));
        # Actual intervention-site geometry, not the residual stream, is used.
        x_cos=[]
        for layer_index in sorted(diagnostic_sink):
            x_cos.append(float(F.cosine_similarity(diagnostic_sink[layer_index]["x_mlp_last"].squeeze(0), vsv[layer_index].cpu(), dim=0)))
        if a.controller == "geometry":
            if a.geometry_target is None: raise ValueError("geometry controller requires --geometry-target")
            adaptive_lambda = solve_lambda_for_angle(torch.tensor(x_cos), a.geometry_target, a.vsv_sim_gate, hi=.30)[0]
        elif a.controller == "visual_deficit":
            adaptive_lambda = law_dose(base_features["relative_raw_diff_norm_mean"], "visual_deficit_inverse", a.law_scale)
        elif a.controller == "sensitivity":
            adaptive_lambda = law_dose(base_features["js"], "sensitivity_inverse", a.law_scale)
        elif a.controller == "consistency":
            adaptive_lambda = law_dose(base_features["layer_adjacent_cos_mean"], "consistency", a.law_scale)
        else: adaptive_lambda = None
        if adaptive_lambda is not None: pending=[float(adaptive_lambda)]
        for lam in pending:
            sink=diagnostic_sink if not cos_layers or all(not torch.isfinite(torch.tensor(x)) for x in cos_layers) else {}
            add_vsv_layers(model,torch.stack([vsv],dim=1).cuda(),[lam],a.layers,sim_gate=a.vsv_sim_gate,diagnostic_sink=sink)
            _, kwargs=loader.prepare_inputs_for_model(template,prompt,image)
            generated=model.generate(
                do_sample=False, max_new_tokens=a.max_new_tokens, use_cache=True,
                num_beams=1, return_dict_in_generate=True, output_attentions=False,
                **kwargs,
            )
            text=loader.decode(generated.sequences)[0]; remove_vsv_layers(model)
            if sink:
                captured=[sink[k]["x_mlp_last"].squeeze(0) for k in sorted(sink)]
                for index,value in enumerate(captured):
                    if index < len(cos_layers): cos_layers[index]=float(F.cosine_similarity(value,vsv[index].cpu(),dim=0))
            gate_values=torch.tensor([float(d["lambda_sim"].mean()) for d in diagnostic_sink.values()]) if diagnostic_sink else torch.ones(1)
            effective=gate_values*lam
            record={"image_id":image_id,"base_lambda":lam,"controller":a.controller,"effective_lambda_definition":"base_lambda * lambda_sim; lambda_sim=legacy or 1","vsv_sim_gate":a.vsv_sim_gate,"legacy_lambda_sim_mean":float(gate_values.mean()),"effective_lambda_mean":float(effective.mean()),"effective_lambda_std":float(effective.std(unbiased=False)),"effective_lambda_min":float(effective.min()),"effective_lambda_max":float(effective.max()),"effective_lambda_early":float(effective[:max(1,len(effective)//4)].mean()),"effective_lambda_late":float(effective[-max(1,len(effective)//4):].mean()),"caption":text,"generated_length":int(generated.sequences.shape[1]-kwargs["input_ids"].shape[1]),"vsv_layer_count":int(vsv.shape[0]),"features":base_features,"intervention_site_cosines":x_cos,"geometry":summarize_angles(torch.tensor(x_cos),lam,a.vsv_sim_gate),"geometry_cos_mean":float(geo_cos.mean()),"geometry_cos_median":float(geo_cos.median()),"official_raw_diff_cos_mean":float(F.cosine_similarity(raw,vsv,dim=-1).mean()),"raw_diff_norm_mean":float(raw.norm(dim=-1).mean()),"token_ids":generated.sequences[0,kwargs["input_ids"].shape[1]:].cpu().tolist()}
            out.write(json.dumps(record)+"\n"); out.flush()
    out.close()

if __name__=="__main__": main()
