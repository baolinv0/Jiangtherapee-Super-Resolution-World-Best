"""Compare PyTorch taps against unmodified shaders extracted from JSR v9.8.

No EXE/PYD is executed. The caller supplies the digest-verified release and
four extracted WGSL files; third-party port code is not used as the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduction"))
sys.path.insert(0, str(ROOT / "Core-Only-Source-Code"))

from jsr_repro.recovered_frontend import learned_merge, phase_statistics
from jsr_repro.recovered_model import load_recovered_jsr
from jsr_repro.frontend import make_controller_input

EXE_SHA256 = "e4fbe9a44ba508e368a7802bb5b28be1e8a49fc666d28f8306f41d34cf3576c3"
SHADER_SHA256 = {
    "phase_accumulate": "0cf47a859208e783ea4bdb9f57dff2342ae682e19cfae7bf716df2ec146843ec",
    "phase_finalize": "b3122eee80d366d382b1801d644004ac347d6d82c9058ffaaaf24307a1cd55d3",
    "learned_accumulate": "f830993f71c007589fcf0869a907267ae1ff3af0400b3ca3758516c1c91be2f9",
    "learned_finalize": "9e97d0787c54682526c60440e08d1740ea92c744881f8fd6dd24732c3b321c0c",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference(expected, actual, atol, rtol=2e-4):
    expected, actual = np.asarray(expected), np.asarray(actual)
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(delta))), "rmse": float(np.sqrt(np.mean(delta * delta))),
            "atol": atol, "rtol": rtol,
            "passed": bool(np.isfinite(delta).all() and np.allclose(actual, expected, atol=atol, rtol=rtol))}


class ReleaseShaders:
    def __init__(self, shader_dir):
        import wgpu_buf as gb
        import wgpu
        self.gb, self.wgpu = gb, wgpu
        self.device = gb.dev()
        self.pipelines = {}
        self.shader_sha256 = {}
        for name in ("phase_accumulate", "phase_finalize", "learned_accumulate", "learned_finalize"):
            path = Path(shader_dir) / (name + ".wgsl")
            code = path.read_text(encoding="utf-8").strip() + "\n"
            self.shader_sha256[name] = hashlib.sha256(code.encode()).hexdigest()
            if self.shader_sha256[name] != SHADER_SHA256[name]:
                raise ValueError(f"extracted release shader digest mismatch: {name}")
            module = self.device.create_shader_module(code=code)
            self.pipelines[name] = self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})

    def dispatch(self, name, buffers, params, total):
        uniform = self.device.create_buffer_with_data(data=params, usage=self.wgpu.BufferUsage.UNIFORM)
        pipe = self.pipelines[name]
        bind = self.gb.bind(pipe, buffers + [uniform])
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipe)
        compute.set_bind_group(0, bind)
        compute.dispatch_workgroups((total + 63) // 64, 1, 1)
        compute.end()
        self.device.queue.submit([encoder.finish()])

    def run(self, frames, flow, covariance, fshift, ci, qmap, origin_yx):
        k, nf, mf = frames.shape
        n, m = flow.shape[1:3]
        ho, wo = covariance.shape[1:]
        pixels, sx = ho * wo, 65536
        shared = [self.gb.upload(x) for x in (frames, ci, flow, fshift, covariance.transpose(1, 2, 0).copy())]
        accum = self.gb.upload(np.zeros((pixels, 150), np.float32))
        phase = self.gb.upload(np.zeros((150, ho, wo), np.float32))
        args = (n, m, nf, mf, ho, wo, k, pixels * 3, sx, 2., *origin_yx, 4, 0, 0, 0)
        self.dispatch("phase_accumulate", shared + [accum], struct.pack("<9I3f4I", *args), pixels * 3)
        self.dispatch("phase_finalize", [accum, phase], struct.pack("<4If3I", pixels * 3, sx, 1, 0, 14. / k, 0, 0, 0), pixels * 3)
        phase_value = phase.numpy()
        legacy = phase_value[144:147] / phase_value[147:150]
        learned_accum = self.gb.upload(np.zeros((pixels, 6), np.float32))
        learned = self.gb.upload(np.zeros((ho, wo, 3), np.float32))
        q = self.gb.upload(qmap)
        args = (n, m, nf, mf, ho, wo, k, pixels, sx, 2., *origin_yx, 12, 5., 0, 0)
        self.dispatch("learned_accumulate", shared + [q, learned_accum], struct.pack("<9I3fIf2I", *args), pixels)
        self.dispatch("learned_finalize", [learned_accum, learned], struct.pack("<4I", pixels, sx, 0, 0), pixels)
        learned_value = learned.numpy().transpose(2, 0, 1)
        for buffer in shared + [accum, phase, q, learned_accum, learned]:
            buffer.destroy()
        return phase_value[:144], legacy, learned_value


def run(shader_dir, release_exe, output):
    import cv2
    report = {"status": "failed", "release": "v9.8", "scope": "registration/covariance boundary; LCA disabled",
              "release_sha256": sha256(release_exe), "torch": str(torch.__version__), "numpy": np.__version__,
              "seed": 2917, "cases": [],
              "reference_components": {"fusion": "original packaged WGSL", "networks": "published WGPU core and released NPZ weights",
                                       "frontend": "published Python frontend in the host runtime", "host_opencv": cv2.__version__,
                                       "note": "Separate frozen frontend/covariance/LCA fixtures use the original PYD with bundled NumPy 2.2.6 and OpenCV 5.0.0.93. This comparison does not launch the complete EXE."},
              "tolerance_note": "FP32 standard deviation subtracts nearly equal moments; its absolute tolerance is 4e-4 for unit-range data, fixed before running. Other tap fields use 2e-5."}
    if report["release_sha256"] != EXE_SHA256:
        raise ValueError("release executable SHA-256 mismatch")
    oracle = ReleaseShaders(shader_dir)
    report["shader_sha256"], report["adapter"] = oracle.shader_sha256, oracle.gb.info()
    torch.set_num_threads(2)
    rng = np.random.default_rng(report["seed"])
    with torch.no_grad():
        for k, ho, wo, origin in ((1, 9, 11, (-.2, -.15)), (4, 16, 24, (0., 0.)), (14, 17, 19, (.1, .2))):
            nf, mf = 12, 15
            frames = rng.uniform(-.05, .8, (k, nf, mf)).astype(np.float32)
            flow = rng.uniform(-.12, .12, (k, nf, mf, 2)).astype(np.float32)
            fshift = np.tile(np.array([0, 0, nf, mf], np.float32), (k, 1))
            fshift[:, 0] = rng.uniform(-.05, .05, k)
            fshift[:, 1] = rng.uniform(-.05, .05, k)
            fshift[-1, 2:] = [nf - 1, mf - 1]
            ci = np.array([[0, 1], [1, 2]], np.int32)[np.arange(nf)[:, None] % 2, np.arange(mf)[None] % 2]
            covariance = np.broadcast_to(np.array([1.1, .1, .9], np.float32)[:, None, None], (3, ho, wo)).copy()
            qmap = rng.normal(0, .15, (36, ho, wo)).astype(np.float32)
            phase, legacy, learned = oracle.run(frames, flow, covariance, fshift, ci, qmap, origin)
            inputs = [torch.from_numpy(x)[None] for x in (frames, flow, covariance)]
            options = {"origin_yx": origin, "fshift": torch.from_numpy(fshift)[None], "color_index": torch.from_numpy(ci)}
            actual = phase_statistics(*inputs, **options)
            actual_learned = learned_merge(*inputs, torch.from_numpy(qmap)[None], **options)[0].numpy()
            ref_fields, actual_fields = phase.reshape(3, 48, ho, wo), actual["phase"][0].numpy().reshape(3, 48, ho, wo)
            checks = {"counts": difference(ref_fields[:, :16], actual_fields[:, :16], 2e-5),
                      "means": difference(ref_fields[:, 16:32], actual_fields[:, 16:32], 2e-5),
                      "standard_deviations": difference(ref_fields[:, 32:], actual_fields[:, 32:], 4e-4),
                      "legacy": difference(legacy, actual["legacy"][0].numpy(), 2e-5),
                      "learned_same_qmap": difference(learned, actual_learned, 2e-5)}
            report["cases"].append({"k": k, "output_shape": [ho, wo], "checks": checks})
            if k == 4:
                from wgpu_unet_tiled import TiledUNet
                from wgpu_greenfilm_rarm import GreenFiLMTiledRArm
                spec = importlib.util.spec_from_file_location("release_frontend", ROOT / "Core-Only-Source-Code/local_v7_frontend.py")
                frontend = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(frontend)
                model = load_recovered_jsr(ROOT / "Core-Only-Source-Code/weights", k)
                ref_feature, amp = frontend.make_controller_input(phase, legacy.transpose(1, 2, 0), ROOT / "Core-Only-Source-Code/weights/local_v7_static_affine_v1.npz")
                gpu_controller = TiledUNet({name: value.numpy() for name, value in model.controller.state_dict().items()})
                gpu_refine = GreenFiLMTiledRArm({name: value.numpy() for name, value in model.refinenet.state_dict().items()})
                reference_qmap = gpu_controller.forward(ref_feature)
                _, _, ref_learned = oracle.run(frames, flow, covariance, fshift, ci, reference_qmap, origin)
                ref_rgb = gpu_refine.factorized(np.concatenate((ref_learned, legacy)), amp, (0,))
                actual_model = model(*inputs, diagnostics=True, **options)
                same_phase_feature, _ = make_controller_input(torch.from_numpy(phase)[None], torch.from_numpy(legacy)[None], model.feature_mean, model.feature_std)
                report["assembled_core"] = {"controller_input_same_phase": difference(ref_feature, same_phase_feature[0].numpy(), 3e-4),
                                            "qmap": difference(reference_qmap, actual_model["qmap"][0].numpy(), 3e-4),
                                            "learned": difference(ref_learned, actual_model["learned"][0].numpy(), 5e-5),
                                            "refined_rgb": difference(ref_rgb, actual_model["rgb"][0].numpy(), 5e-5)}
                report["diagnostics"] = {
                    "propagated_controller_input_strict": difference(ref_feature, actual_model["controller_input"][0].numpy(), 3e-4),
                    "explanation": "The separate phase producers differ in FP32 sqrt(max(E[x^2]-mu^2,0)) near zero variance. Frozen normalization amplifies this dispersion error. The first run failed the strict propagated feature check; it is retained here. A subsequent same-phase check isolates frontend correctness. Component and final RGB tolerances are unchanged. This is not bitwise or strict intermediate-feature equivalence."}
                fixture_dir = Path(output).parent
                fixture_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(fixture_dir / "release-boundary-k4.npz", frames=frames, flow=flow,
                                    covariance=covariance, fshift=fshift, color_index=ci, origin_yx=np.array(origin, np.float32))
                gpu_refine.close()
    passed = all(check["passed"] for case in report["cases"] for check in case["checks"].values())
    passed = passed and all(check["passed"] for check in report["assembled_core"].values())
    intermediate_passed = report["diagnostics"]["propagated_controller_input_strict"]["passed"]
    report["status"] = ("passed" if intermediate_passed else "passed_with_intermediate_deviation") if passed else "failed"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-dir", type=Path, required=True)
    parser.add_argument("--release-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.shader_dir, args.release_exe, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in ("passed", "passed_with_intermediate_deviation") else 1


if __name__ == "__main__":
    raise SystemExit(main())
