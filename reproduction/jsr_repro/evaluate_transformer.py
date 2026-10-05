"""Held-out bracketed proxy metrics, fixed radiance white1, no gain fitting."""
from __future__ import annotations

import argparse
import json

import torch

from .alignment import estimate_translations
from .bracket_data import BracketBurstDataset
from .calibration import load_profile, profile_identity
from .data import verify_manifest
from .metrics import crop_pair, image_metrics
from .train_transformer import batch_sample, load_checkpoint
from .utils import device_from, save_json, seed_all, sha256


def estimated_shifts(sample):
    # Constant per-frame exposure correction BEFORE green-cell FFT baseline.
    return estimate_translations(sample["raw"]/sample["exposure"][:,:,None,None,None])


def detailed_metrics(prediction,target,border):
    result = image_metrics(prediction,target,border)
    p,t = crop_pair(prediction,target,border)
    mask = t>1
    result["highlight_rmse"] = (p-t)[mask].square().mean().sqrt().item() if mask.any() else None
    result["highlight_fraction"] = mask.double().mean().item()
    return result


@torch.no_grad()
def evaluate(checkpoint,manifest,output,alignment="oracle",device="cpu",limit=None,split="test"):
    seed_all(1234,2)
    device = device_from(device)
    model,state = load_checkpoint(checkpoint,device)
    model.eval()
    verify_manifest(manifest)
    cfg = state["config"]
    profile = load_profile(cfg["data"].get("profile"))
    if profile_identity(profile) != state["data_identity"]["profile_sha256"]:
        raise ValueError("evaluation calibration differs from checkpoint")
    if alignment not in ("oracle","estimated"):
        raise ValueError("alignment must be oracle or estimated")
    dataset = BracketBurstDataset(manifest,split,cfg["data"]["options"],cfg.get("seed",1234)+2,cfg["data"].get("profile"))
    border = cfg["train"].get("crop_border",4)
    records = []
    for i in range(min(len(dataset),limit) if limit is not None else len(dataset)):
        source = dataset[i]
        sample = batch_sample(source,device)
        truth = sample["shifts"].clone()
        if alignment=="estimated":
            sample["shifts"] = estimated_shifts(sample)
        result = model(sample)
        target = source["target"][None].to(device)
        records.append({"index":i,"metadata":json.loads(source["metadata"]),"alignment_rmse_native":(truth-sample["shifts"]).square().mean().sqrt().item(),
                        "all_saturated_or_invalid_packed_fraction":result["all_saturated_or_invalid_packed"].mean().item(),
                        "methods":{name:detailed_metrics(result[key],target,border) for name,key in (("corrected_merge","baseline"),("speech_transformer","rgb"))}})
    if not records:
        raise ValueError("empty evaluation set")
    summary = {}
    for name in records[0]["methods"]:
        summary[name] = {}
        for key in records[0]["methods"][name]:
            vals = [row["methods"][name][key] for row in records if row["methods"][name][key] is not None]
            summary[name][key] = sum(vals)/len(vals) if vals else None
    report = {"implementation":state["implementation"],"scope":"held-out synthetic RGB camera proxy, NOT official JSR/BurstSR score",
              "count":len(records),"split":split,"alignment":alignment,"data_range":1.,"gain_fit":False,"crop_border_hr":border,
              "checkpoint_sha256":sha256(checkpoint),"manifest_sha256":sha256(manifest),"mean_per_image_metrics":summary,"samples":records}
    save_json(output,report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","manifest","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--alignment",choices=["oracle","estimated"],default="oracle")
    p.add_argument("--device",default="cpu")
    p.add_argument("--limit",type=int)
    p.add_argument("--split",choices=["train","val","test"],default="test")
    print(json.dumps(evaluate(**vars(p.parse_args()))["mean_per_image_metrics"],indent=2))


if __name__ == "__main__":
    main()
