"""Run the recovered v9.8 core on explicitly supplied release-stage tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .recovered_frontend import FRONTEND_VERSION
from .recovered_model import load_recovered_jsr


def run(input_path, weights_dir, output, *, refinement=True, max_output_pixels=65536):
    input_path, weights_dir, output = Path(input_path), Path(weights_dir), Path(output)
    if output.exists():
        raise ValueError("output directory already exists; select a new directory")
    with np.load(input_path, allow_pickle=False) as archive:
        required = {"frames", "flow", "covariance"}
        optional = {"fshift", "color_index", "origin_yx", "lca_coefficients_rb"}
        if not required <= set(archive.files) or set(archive.files) - required - optional:
            raise ValueError("expected frames, flow, covariance and optional fshift/color_index/origin_yx/lca_coefficients_rb")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    frames, flow, covariance = [torch.from_numpy(np.asarray(arrays[name], np.float32))[None]
                                for name in ("frames", "flow", "covariance")]
    if covariance.ndim != 4 or covariance.shape[-2] * covariance.shape[-1] > max_output_pixels:
        raise ValueError("covariance must be [3,Hout,Wout] within max_output_pixels")
    options = {}
    if "origin_yx" in arrays:
        options["origin_yx"] = arrays["origin_yx"]
    if "color_index" in arrays:
        options["color_index"] = arrays["color_index"]
    if "fshift" in arrays:
        options["fshift"] = torch.from_numpy(np.asarray(arrays["fshift"], np.float32))[None]
    if "lca_coefficients_rb" in arrays:
        options["lca_coefficients_rb"] = arrays["lca_coefficients_rb"]
    model = load_recovered_jsr(weights_dir, frames.shape[1], refinement=refinement)
    with torch.no_grad():
        stages = model(frames, flow, covariance, diagnostics=True, **options)
    saved = {name: value[0].cpu().numpy() for name, value in stages.items()}
    for name in ("rgb", "learned", "legacy"):
        saved[name] = saved[name].transpose(1, 2, 0)
    if not all(np.isfinite(value).all() for value in saved.values()):
        raise RuntimeError("reconstruction produced nonfinite values")
    report = {"frontend_version": FRONTEND_VERSION, "release": "v9.8", "k": frames.shape[1],
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "weight_manifest_sha256": hashlib.sha256((weights_dir / "manifest.json").read_bytes()).hexdigest(),
              "lca": "provided fixed radial coefficients" if "lca_coefficients_rb" in arrays else "disabled", "refinement": bool(refinement),
              "registration": "provided at release module boundary", "covariance": "provided at release module boundary",
              "scope": "reconstruction core only; no end-to-end RAW registration or paper-quality claim",
              "shapes": {name: list(value.shape) for name, value in saved.items()},
              "torch": str(torch.__version__), "numpy": np.__version__}
    output.mkdir(parents=True)
    np.savez_compressed(output / "stages.npz", **saved)
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="boundary NPZ: frames[K,H,W], flow[K,N,M,2], covariance[3,Ho,Wo]")
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Core-Only-Source-Code/weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--max-output-pixels", type=int, default=65536)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run(args.input, args.weights_dir, args.output, refinement=not args.no_refine,
                 max_output_pixels=args.max_output_pixels)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
