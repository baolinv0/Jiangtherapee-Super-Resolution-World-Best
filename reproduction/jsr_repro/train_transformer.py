"""Dedicated speech-inspired training/checkpoint path; no public-weight loading."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .bracket_data import make_burst_dataset
from .calibration import profile_identity
from .data import verify_manifest
from .metrics import image_metrics, reconstruction_loss
from .train import capture_rng, restore_rng
from .transformer import FIELDS, IMPLEMENTATION, SpeechTransformer
from .utils import device_from, load_config, save_json, seed_all, sha256


def batch_sample(sample, device):
    return {key:sample[key][None].to(device) for key in (*FIELDS, *(("capture_order",) if "capture_order" in sample else ()))}


def load_checkpoint(path, device="cpu"):
    state = torch.load(path,map_location=device,weights_only=True)
    if state.get("format_version") != 1 or state.get("implementation") != IMPLEMENTATION:
        raise ValueError("expected speech-inspired-transformer-v1 checkpoint; v1/public weights are incompatible")
    model = SpeechTransformer(**state["config"]["model"]).to(device)
    model.load_state_dict(state["model"],strict=True)
    return model,state


def transformer_loss(prediction, target, border=4, highlight_weight=.2):
    loss = reconstruction_loss(prediction,target,border,.1,.05)
    pred,truth = (prediction[...,border:-border,border:-border],target[...,border:-border,border:-border]) if border else (prediction,target)
    mask = truth>1
    if mask.any():
        loss = loss + highlight_weight*(pred-truth).abs()[mask].mean()
    return loss


@torch.no_grad()
def validate_model(model,dataset,device,border):
    training = model.training
    model.eval()
    scores = []
    for sample in dataset:
        result = model(batch_sample(sample,device))["rgb"]
        scores.append(image_metrics(result,sample["target"][None].to(device),border)["psnr_linear_db"])
    model.train(training)
    return sum(scores)/len(scores)


def run_training(config,resume=None,stop_after=None):
    cfg = json.loads(json.dumps(config))
    seed_all(int(cfg.get("seed",1234)),int(cfg.get("threads",2)))
    device = device_from(cfg.get("device","cpu"))
    data,settings = cfg["data"],cfg["train"]
    if settings.get("batch_size",1) != 1:
        raise ValueError("reference Transformer trainer supports batch_size1")
    steps,every,border = int(settings["steps"]),int(settings.get("validate_every",8)),int(settings.get("crop_border",4))
    if steps<1 or every<1 or border<0 or 2*border>=int(data["options"].get("native_size",16))*2:
        raise ValueError("invalid steps/validation interval/crop")
    verify_manifest(data["manifest"])
    train = make_burst_dataset(data["manifest"],"train",data["options"],int(cfg.get("seed",1234)),data.get("profile"))
    val = make_burst_dataset(data["manifest"],"val",data["options"],int(cfg.get("seed",1234))+1,data.get("profile"))
    identity = {"manifest_sha256":sha256(data["manifest"]),"profile_sha256":profile_identity(train.profile)}
    if hasattr(train, 'identity'):
        identity['construction'] = train.identity
    model = SpeechTransformer(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(settings.get("lr",.0002)),weight_decay=float(settings.get("weight_decay",0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
    output = Path(cfg["output"])
    output.mkdir(parents=True,exist_ok=True)
    step,best = 0,-float("inf")
    if resume:
        _,state = load_checkpoint(resume,device)
        for key in ("model","data","train","seed","threads"):
            if state["config"].get(key) != cfg.get(key):
                raise ValueError(f"resume {key} configuration changed")
        if state["data_identity"] != identity:
            raise ValueError("resume data/calibration identity changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step,best = int(state["step"]),state["best_val_psnr"]
        restore_rng(state["rng"])
    elif (output/"last.pt").exists():
        raise FileExistsError("output contains last.pt; use resume or new output")
    end = min(steps,int(stop_after)) if stop_after is not None else steps
    if end<=step:
        raise ValueError("stop-after must exceed checkpoint step")
    save_json(output/"config.json",cfg)
    save_json(output/"camera_profile.json",train.profile)
    save_json(output/"environment.json",{"torch":str(torch.__version__),"device":str(device),"data_identity":identity,"deterministic":True})
    start = time.perf_counter()
    model.train()
    while step<end:
        epoch,cursor = divmod(step,len(train))
        train.set_epoch(epoch)
        permutation = torch.randperm(len(train),generator=torch.Generator().manual_seed(int(cfg.get("seed",1234))+epoch))
        sample = train[int(permutation[cursor])]
        inputs,target = batch_sample(sample,device),sample["target"][None].to(device)
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs)
        loss = transformer_loss(result["rgb"],target,border,float(settings.get("highlight_weight",.2)))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Transformer loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),float(settings.get("clip_grad",1)),error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step+=1
        record = {"step":step,"epoch":epoch,"cursor":cursor,"loss":loss.item(),"grad_norm":grad.item(),"lr":optimizer.param_groups[0]["lr"],
                  "signal_saturation_fraction":json.loads(sample["metadata"])["signal_saturation_fraction"]}
        if step%every==0 or step==end:
            score = validate_model(model,val,device,border)
            record["val_psnr_linear_db"] = score
            improved = score>best
            best = max(best,score)
            state = {"format_version":1,"implementation":IMPLEMENTATION,"config":cfg,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                     "scheduler":scheduler.state_dict(),"rng":capture_rng(),"step":step,"epoch":step//len(train),"cursor":step%len(train),
                     "data_identity":identity,"camera_profile":train.profile,"best_val_psnr":best}
            torch.save(state,output/"last.tmp.pt")
            (output/"last.tmp.pt").replace(output/"last.pt")
            if improved:
                torch.save(state,output/"best.pt")
            print(json.dumps(record),flush=True)
        with (output/"train.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
    report = {"implementation":IMPLEMENTATION,"steps_completed":step,"wall_seconds":time.perf_counter()-start,"best_val_psnr":best,
              "checkpoint":"last.pt","scope":"synthetic engineering smoke; not author quality or benchmark"}
    save_json(output/"training_summary.json",report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after",type=int)
    args = p.parse_args()
    print(json.dumps(run_training(load_config(args.config),args.resume,args.stop_after),indent=2))


if __name__ == "__main__":
    main()
