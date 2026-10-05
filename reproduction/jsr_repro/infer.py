"""Run a reproduction checkpoint on a normalized Bayer burst archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .alignment import estimate_translations
from .data import linear_to_srgb
from .train import load_checkpoint
from .utils import seed_all, device_from, save_json, sha256


@torch.no_grad()
def infer(checkpoint, burst, output, alignment="estimated", device="cpu", max_native_pixels=65536):
    seed_all(1234)
    device = device_from(device)
    model, state = load_checkpoint(checkpoint, device)
    model.eval()
    with np.load(burst, allow_pickle=False) as archive:
        raw_np = np.asarray(archive["raw"], np.float32)
        shifts_np = np.asarray(archive["shifts"], np.float32) if "shifts" in archive else None
    if raw_np.ndim != 4 or raw_np.shape[1] != 1 or not 1 <= raw_np.shape[0] <= 14 or any(d % 2 for d in raw_np.shape[-2:]):
        raise ValueError("archive.raw must be K,1,H,W, even dimensions, K=1..14")
    if not np.isfinite(raw_np).all():
        raise ValueError("RAW archive contains NaN/Inf")
    if max_native_pixels < 1 or raw_np.shape[-2] * raw_np.shape[-1] > max_native_pixels:
        raise ValueError("burst exceeds the full-image reference memory limit; import a small even CFA crop with import_raw --crop, or explicitly raise --max-native-pixels on a machine with sufficient memory")
    raw = torch.from_numpy(raw_np)[None].to(device)
    if alignment == "provided":
        if shifts_np is None or shifts_np.shape != (raw_np.shape[0], 2) or not np.isfinite(shifts_np).all() or np.any(shifts_np[0] != 0):
            raise ValueError("provided alignment requires finite K,2 shifts with frame 0 at zero")
        shifts = torch.from_numpy(shifts_np)[None].to(device)
    elif alignment == "estimated":
        shifts = estimate_translations(raw)
    else:
        raise ValueError("choose provided or estimated alignment")
    # Full image reference path. No misleading crop-and-stitch equivalence claim.
    result = model(raw, shifts)["rgb"][0].cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite output")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "linear_rgb.npy", result.permute(1, 2, 0).numpy())
    preview = (linear_to_srgb(result).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(preview).save(output / "preview.png")
    save_json(output / "inference.json", {"scope": "independent inferred JSR pipeline", "checkpoint_sha256": sha256(checkpoint), "burst_sha256": sha256(burst), "alignment": alignment, "shifts_xy_native": shifts[0].cpu().tolist(), "scale_native": state["config"]["model"].get("scale", 2), "input_shape": list(raw_np.shape), "output_shape_hwc": list(result.permute(1, 2, 0).shape), "preview": "sRGB encoding only, no camera color/WB correction; not a finished photograph"})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--burst", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alignment", choices=["estimated", "provided"], default="estimated")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-native-pixels", type=int, default=65536, help="memory guard for full-image reference path; default 256x256")
    print(infer(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
