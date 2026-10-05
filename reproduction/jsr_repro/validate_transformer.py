"""Offline smoke evidence for speech-inspired model; never a quality ranking."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .bracket_data import BracketBurstDataset, save_burst, synthesize_bracket
from .calibration import analytic_profile, fit_profile
from .data import linear_to_srgb
from .evaluate_transformer import evaluate
from .infer_transformer import infer
from .prepare import build_manifest
from .train_transformer import batch_sample, load_checkpoint, run_training
from .transformer import SpeechTransformer
from .utils import load_config, save_json, seed_all


@torch.no_grad()
def numerical_probes(model, sample):
    original = batch_sample(sample,"cpu")
    result = model(original,trace=True)
    zero = {k:v.clone() for k,v in original.items()}
    zero["raw"].zero_()
    zero["variance"].zero_()
    zero["saturation"].zero_()
    zero["black_invalid"].zero_()
    values = []
    for gain in (.125,.5,2.):
        scaled = {k:v.clone() for k,v in original.items()}
        scaled["raw"] *= gain
        scaled["variance"] *= gain**2
        pred = model(scaled)["rgb"]
        ref = result["rgb"]*gain
        values.append({"gain":gain,"relative_rmse":((pred-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item()})
    return {"zero_max_abs":model(zero)["rgb"].abs().max().item(),"conditional_gain":values,"trace":result["trace"],
            "gain_conditions":"same geometry/exposure/masks; RAW scaled, noise covariance scaled squared; no capture/clipping transition"}


def validate(output, config="configs/transformer_smoke.yaml"):
    seed_all(1234,2)
    root = Path(output).resolve()
    if (root/"validation.json").exists() or (root/"training/last.pt").exists():
        raise FileExistsError("use a fresh Transformer validation output")
    root.mkdir(parents=True,exist_ok=True)
    manifest = build_manifest(None,root/"fixtures",procedural=12)
    cfg = load_config(config)
    cfg["output"],cfg["data"]["manifest"] = str(root/"training"),str(manifest)
    trained = run_training(cfg)
    checkpoint = root/"training/last.pt"
    oracle = evaluate(checkpoint,manifest,root/"evaluation_oracle.json",alignment="oracle")
    estimated = evaluate(checkpoint,manifest,root/"evaluation_estimated.json",alignment="estimated")
    dataset = BracketBurstDataset(manifest,"test",cfg["data"]["options"],cfg["seed"]+2,cfg["data"].get("profile"))
    sample = dataset[0]
    save_burst(root/"burst.npz",sample)
    inferred = infer(checkpoint,root/"burst.npz",root/"inference",alignment="provided",radiance_white=4.)
    model,_ = load_checkpoint(checkpoint)
    model.eval()
    probes = numerical_probes(model,sample)
    save_json(root/"probes.json",probes)
    if probes["zero_max_abs"] != 0 or max(v["relative_rmse"] for v in probes["conditional_gain"])>1e-4:
        raise AssertionError("conditional zero/gain property failed")
    # Synthetic input calibration verifies units; not a measured camera PTC.
    csv_path = root/"synthetic-ptc.csv"
    csv_path.write_text("iso,mean_dn,variance_dn2\n"+"".join(f"{iso},{m},{m/(4*100/iso)+(3/(4*100/iso))**2+1/12}\n" for iso in (100,200,400,800) for m in (10,100,500,1000)),encoding="utf-8")
    fitted = fit_profile(csv_path,root/"fitted-synthetic-profile.json")
    calibration_error = max(abs(fitted["iso"][str(iso)]["gain_e_per_dn"]-4*100/iso) for iso in (100,200,400,800))
    save_json(root/"calibration-check.json",{"input":"generated synthetic PTC, not measured","gain_max_abs_error_e_per_dn":calibration_error,"diagnostics":fitted["fit_diagnostics"]})
    # Resume evidence separate from the16-step model: identical 4 vs2+2.
    short = copy.deepcopy(cfg)
    short["train"]["steps"],short["train"]["validate_every"] = 4,2
    short["output"] = str(root/"resume-full")
    run_training(short)
    part = copy.deepcopy(short)
    part["output"] = str(root/"resume-part")
    run_training(part,stop_after=2)
    run_training(part,resume=root/"resume-part/last.pt")
    a,sa = load_checkpoint(root/"resume-full/last.pt")
    b,sb = load_checkpoint(root/"resume-part/last.pt")
    resume_equal = all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()) and sa["scheduler"]==sb["scheduler"]
    if not resume_equal:
        raise AssertionError("CPU exact resume mismatch")
    save_json(root/"resume-check.json",{"continuous_steps":4,"split_steps":[2,2],"state_dict_bitwise_equal":resume_equal,"schedule_equal":True})
    # Larger configured width/native crop; full proposed100k schedule is unrun.
    larger = synthesize_bracket(torch.full((3,144,144),.2),dict(cfg["data"]["options"],native_size=32),torch.Generator().manual_seed(331))
    big = SpeechTransformer(width=32,heads=4)
    optimizer = torch.optim.AdamW(big.parameters(),lr=.0001)
    loss = (big(batch_sample(larger,"cpu"))["rgb"]-larger["target"][None]).abs().mean()
    loss.backward()
    grads = [p.grad for p in big.parameters()]
    finite = all(g is not None and torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    if not finite:
        raise AssertionError("larger-model gradient contract failed")
    optimizer.step()
    save_json(root/"larger-check.json",{"native_size":32,"width":32,"heads":4,"frames":7,"parameters":sum(p.numel() for p in big.parameters()),"all_parameter_gradients_finite_nonzero":finite,"one_optimizer_step":True,"loss":loss.item()})
    with torch.no_grad():
        result = model(batch_sample(sample,"cpu"))
    white = 4.
    panels = []
    for label,index in (("Short: sensor",1),("Reference: sensor",0),("Long: sensor",6)):
        # Simple green-mosaic visualization of sensor levels, no exposure fit.
        value = sample["raw"][index].expand(3,-1,-1)
        panels.append((label,value,1.))
    panels += [("GT: linear /4",sample["target"],white),("Merge: linear /4",result["baseline"][0],white),
               ("Model: linear /4",result["rgb"][0],white),("Saturation: reference",sample["saturation"][0].expand(3,-1,-1),1.),
               ("Amplitude /4",result["amplitude"][0].expand(3,-1,-1),white)]
    sheet = Image.new("RGB",(4*192,2*220),"#fafafa")
    draw = ImageDraw.Draw(sheet)
    for i,(label,value,scale) in enumerate(panels):
        preview = (linear_to_srgb(value/scale).permute(1,2,0).numpy()*255).round().astype(np.uint8)
        x,y = (i%4)*192,(i//4)*220
        sheet.paste(Image.fromarray(preview).resize((192,192),Image.Resampling.NEAREST),(x,y+28))
        draw.text((x+6,y+6),label,fill="#202020")
    sheet.save(root/"comparison.png")
    save_json(root/"visual-index.json",{"files":[{"path":"comparison.png","description":"Synthetic fixture. RAW mosaic shown as gray; GT/merge/model/amplitude use fixed radiance_white4, sensor/masks white1. sRGB diagnostic encoding, no fitted gains."},{"path":"inference/preview.png","description":"RGB output diagnostic at fixed radiance_white4"}]})
    metadata = json.loads(sample["metadata"])
    report = {"status":"passed","implementation":inferred["implementation"],"scope":"CPU engineering smoke; no paper-quality, real-camera or SOTA evidence",
              "training":trained,"oracle":oracle["mean_per_image_metrics"],"estimated":estimated["mean_per_image_metrics"],
              "zero_max_abs":probes["zero_max_abs"],"gain_max_relative_rmse":max(v["relative_rmse"] for v in probes["conditional_gain"]),
              "resume_bitwise_equal":resume_equal,"ptc_gain_max_abs_error":calibration_error,"test_signal_saturation_fraction":metadata["signal_saturation_fraction"],
              "test_gt_above_white_fraction":metadata["target_above_reference_white_fraction"],"storage":inferred["encoding"]}
    save_json(root/"validation.json",report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--config",default="configs/transformer_smoke.yaml")
    args=p.parse_args()
    print(json.dumps(validate(args.output,args.config),indent=2))


if __name__=="__main__":
    main()
