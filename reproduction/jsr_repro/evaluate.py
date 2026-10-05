"""Held-out synthetic evaluation: matched color, size, crop and alignment track."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .alignment import estimate_translations, single_frame_baseline
from .data import SyntheticBurstDataset, verify_manifest
from .metrics import image_metrics
from .train import load_checkpoint
from .utils import device_from, save_json, seed_all, sha256


@torch.no_grad()
def evaluate(checkpoint, manifest, output, split="test", alignment="oracle", device="cpu", limit=None, frames=None):
    seed_all(1234)
    device = device_from(device)
    model, state = load_checkpoint(checkpoint, device)
    model.eval()
    verify_manifest(manifest)
    options = dict(state["config"]["data"]["options"])
    if frames is not None:
        options["frames"] = frames
    dataset = SyntheticBurstDataset(manifest, split, options, state["config"].get("seed", 1234) + 2)
    if alignment not in ("oracle", "estimated"):
        raise ValueError("alignment must be oracle or estimated")
    border = int(state["config"]["train"].get("crop_border", 4))
    records = []
    for i in range(min(len(dataset), limit) if limit else len(dataset)):
        sample = dataset[i]
        raw, truth, target = (sample[k][None].to(device) for k in ("raw", "shifts", "target"))
        shifts = truth if alignment == "oracle" else estimate_translations(raw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model(raw, shifts)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        methods = {"single_bilinear": single_frame_baseline(raw, options.get("scale", 2)), "legacy": result["legacy"], "learned": result["learned"], "jsr_repro": result["rgb"]}
        entry = {"index": i, "metadata": json.loads(sample["metadata"]), "methods": {name: image_metrics(value, target, border) for name, value in methods.items()}, "seconds_model": seconds, "alignment_rmse_native": (shifts - truth).square().mean().sqrt().item()}
        records.append(entry)
    if not records:
        raise ValueError("empty evaluation set")
    summary = {}
    for method in records[0]["methods"]:
        summary[method] = {}
        for key in records[0]["methods"][method]:
            vals = [r["methods"][method][key] for r in records if r["methods"][method][key] is not None]
            summary[method][key] = float(np.mean(vals)) if vals else None
    report = {"scope": "held-out synthetic proxy, not official JSR/BurstSR benchmark", "alignment": alignment, "split": split, "count": len(records), "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": sha256(manifest), "crop_border_hr": border, "data_range": 1., "gain_fit": False, "options": options, "mean_per_image_metrics": summary, "samples": records}
    save_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--alignment", choices=["oracle", "estimated"], default="oracle")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int)
    p.add_argument("--frames", type=int)
    args = p.parse_args()
    result = evaluate(**vars(args))
    print(json.dumps(result["mean_per_image_metrics"], indent=2))


if __name__ == "__main__":
    main()
