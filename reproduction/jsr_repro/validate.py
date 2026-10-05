"""One-command offline engineering validation; writes evidence, not SOTA claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .data import SyntheticBurstDataset, linear_to_srgb
from .evaluate import evaluate
from .infer import infer
from .prepare import build_manifest
from .probes import run_probes
from .train import run_training, load_checkpoint
from .utils import load_config, save_json


def validate(output, config="configs/smoke.yaml"):
    root = Path(output).resolve()
    if (root / "validation.json").exists() or (root / "training/last.pt").exists():
        raise FileExistsError("use a fresh validation output directory")
    manifest = build_manifest(None, root / "fixtures", procedural=12)
    cfg = load_config(config)
    cfg["output"] = str(root / "training")
    cfg["data"]["manifest"] = str(manifest)
    trained = run_training(cfg)
    checkpoint = root / "training/last.pt"
    oracle = evaluate(checkpoint, manifest, root / "evaluation_oracle.json", alignment="oracle")
    estimated = evaluate(checkpoint, manifest, root / "evaluation_estimated.json", alignment="estimated")
    model, _ = load_checkpoint(checkpoint)
    probes = run_probes(model)
    save_json(root / "probes.json", probes)
    if probes["zero_max_abs"] != 0 or max(r["relative_rmse"] for r in probes["gain_equivariance"]) > 1e-4:
        raise AssertionError("zero/gain numerical contract failed")
    sample = SyntheticBurstDataset(manifest, "test", cfg["data"]["options"], cfg["seed"] + 2)[0]
    np.savez_compressed(root / "burst.npz", raw=sample["raw"].numpy(), shifts=sample["shifts"].numpy(), target=sample["target"].numpy(), metadata=sample["metadata"])
    infer(checkpoint, root / "burst.npz", root / "inference", "provided")
    with torch.no_grad():
        result = model(sample["raw"][None], sample["shifts"][None])
    # Visual evidence of this procedural fixture, identical encoding for all.
    panels = [("Target", sample["target"]), ("Legacy", result["legacy"][0]), ("Reproduction", result["rgb"][0])]
    sheet = Image.new("RGB", (3 * 256, 288), "#fafafa")
    draw = ImageDraw.Draw(sheet)
    for i, (name, value) in enumerate(panels):
        array = (linear_to_srgb(value).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        panel = Image.fromarray(array).resize((256, 256), Image.Resampling.NEAREST)
        sheet.paste(panel, (i * 256, 32))
        draw.text((i * 256 + 10, 10), name, fill="#202020")
    sheet.save(root / "comparison.png")
    report = {"status": "passed", "scope": "CPU engineering smoke; no full dataset training or original-quality claim", "training": trained, "oracle": oracle["mean_per_image_metrics"], "estimated": estimated["mean_per_image_metrics"], "zero_max_abs": probes["zero_max_abs"], "gain_max_relative_rmse": max(r["relative_rmse"] for r in probes["gain_equivariance"])}
    save_json(root / "validation.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/smoke.yaml")
    args = p.parse_args()
    print(json.dumps(validate(args.output, args.config), indent=2))


if __name__ == "__main__":
    main()
