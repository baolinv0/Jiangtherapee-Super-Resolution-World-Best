"""Optional WGPU vs PyTorch module parity, using released K4 NPZ weights.

This compares identical input tensors at individual public module boundaries.
It cannot establish end-to-end compatibility of the independently inferred RAW
statistics / Tap mapping. Requires optional wgpu and an available GPU adapter.
Driver/import failures are saved as diagnostics rather than hidden as a pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduction"))
sys.path.insert(0, str(ROOT / "Core-Only-Source-Code"))

from jsr_repro.model import REFERENCE_AMPLITUDE
from jsr_repro.weights import load_controller_npz, load_refinenet_npz, verify_manifest_file


def compare(reference, actual, atol=2e-5, rtol=2e-4):
    difference = np.abs(np.asarray(reference, np.float64) - np.asarray(actual, np.float64))
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    return {"shape": list(reference.shape), "finite": finite,
            "max_absolute_error": float(difference.max()) if finite else None,
            "mean_absolute_error": float(difference.mean()) if finite else None,
            "rmse": float(np.sqrt(np.mean(difference ** 2))) if finite else None,
            "atol": atol, "rtol": rtol,
            "passed": finite and bool(np.allclose(reference, actual, atol=atol, rtol=rtol))}


def run(weights_dir, output):
    report = {"scope": "public standalone module parity only; not inferred RAW pipeline parity",
              "status": "blocked", "python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": np.__version__, "seed": 1329,
              "note": "PyTorch exact-erf GELU vs public WGPU polynomial-erf GELU; tolerate accumulation/activation rounding."}
    try:
        import wgpu
        import wgpu_buf as gb
        from wgpu_unet_tiled import TiledUNet
        from wgpu_greenfilm_rarm import GreenFiLMTiledRArm
        report["wgpu"] = wgpu.__version__
        torch.set_num_threads(2)
        weights_dir = Path(weights_dir)
        controller_path = weights_dir / "v97_k04_controller_wgpu.npz"
        refine_path = weights_dir / "v97_k04_rarm_wgpu.npz"
        report["weight_sha256"] = {p.name: verify_manifest_file(p, weights_dir / "manifest.json") for p in (controller_path, refine_path)}
        torch_controller = load_controller_npz(controller_path)
        torch_refine = load_refinenet_npz(refine_path)
        gpu_controller = TiledUNet({name: value.numpy() for name, value in torch_controller.state_dict().items()})
        gpu_refine = GreenFiLMTiledRArm({name: value.numpy() for name, value in torch_refine.state_dict().items()})
        report["adapter"] = gb.info()
        rng = np.random.default_rng(report["seed"])
        feature = rng.standard_normal((151, 32, 32), dtype=np.float32) * np.float32(.2)
        rgb_pair = rng.random((1, 6, 16, 16), dtype=np.float32)
        with torch.no_grad():
            expected_controller = torch_controller(torch.from_numpy(feature)[None])[0].numpy()
            expected_refine = torch_refine(torch.from_numpy(rgb_pair)).numpy()
            amplitude = np.full((1, 16, 16), REFERENCE_AMPLITUDE, dtype=np.float32)
            expected_factorized = torch_refine.factorized(torch.from_numpy(rgb_pair), torch.from_numpy(amplitude)[None])[0].numpy()
        report["controller"] = compare(expected_controller, gpu_controller.forward(feature))
        report["refinenet_residual"] = compare(expected_refine, gpu_refine.forward_residual_batch(rgb_pair))
        report["refinenet_factorized"] = compare(expected_factorized, gpu_refine.factorized(rgb_pair[0], amplitude, (0,)))
        gpu_refine.close()
        report["status"] = "passed" if all(report[key]["passed"] for key in ("controller", "refinenet_residual", "refinenet_factorized")) else "failed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "Core-Only-Source-Code/weights")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.weights_dir, args.output)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
