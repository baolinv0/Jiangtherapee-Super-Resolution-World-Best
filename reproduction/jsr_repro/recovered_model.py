"""Released-weight reconstruction with explicit registration/covariance inputs.

This is the v9.8 reconstruction boundary, with optional fixed-coefficient LCA. It
requires flow and kernel covariance instead of silently borrowing the older
independent registration. Original training data/recipe are not recovered.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path

import torch
from torch import Tensor, nn

from .frontend import local_amplitude, make_controller_input
from .model import Controller, RefineNet
from .recovered_frontend import FRONTEND_VERSION, learned_merge, phase_statistics
from .recovered_lca import warp_rgb_tile
from .weights import load_controller_npz, load_refinenet_npz, load_static_affine_npz, verify_manifest_file


class RecoveredJSR(nn.Module):
    """Trainable continuous-tap core; consumes release-coordinate tensors.

    frames [B,K,Hframe,Wframe], flow [B,K,Hreference,Wreference,2] in
    (dx,dy), covariance [B,3,Hout,Wout] in (xx,xy,yy). RAW and linear RGB
    share units. Output grid is fixed at two samples per native pixel.
    Learned and legacy RGB enter RefineNet in that order. This entry point
    is separate from JSRModel and its independently trained checkpoints.
    """
    frontend_version = FRONTEND_VERSION

    def __init__(self, controller_width=32, refine_width=116, refine_blocks=8, *, refinement=True):
        super().__init__()
        self.controller = Controller(controller_width)
        self.refinenet = RefineNet(refine_width, refine_blocks)
        self.refinement = refinement
        self.expected_frames = None
        self.register_buffer("feature_mean", torch.zeros(1, 151, 1, 1))
        self.register_buffer("feature_std", torch.ones(1, 151, 1, 1))

    def forward(self, frames: Tensor, flow: Tensor, covariance: Tensor, *, origin_yx=(0., 0.),
                fshift=None, color_index=None, cap=5., lca_coefficients_rb=None, diagnostics=False) -> dict[str, Tensor]:
        if frames.ndim != 4:
            raise ValueError("frames must be [B,K,Hframe,Wframe]")
        if self.expected_frames is not None and frames.shape[1] != self.expected_frames:
            raise ValueError(f"loaded weight route requires K={self.expected_frames}")
        if covariance.ndim == 4 and min(covariance.shape[-2:]) < 8:
            raise ValueError('Released Controller pooling requires output height and width at least 8')
        options = dict(origin_yx=origin_yx, fshift=fshift, color_index=color_index)
        evidence = phase_statistics(frames, flow, covariance, **options)
        legacy = evidence["legacy"]
        feature, amplitude = make_controller_input(evidence["phase"], legacy, self.feature_mean, self.feature_std)
        qmap = self.controller(feature)
        learned = learned_merge(frames, flow, covariance, qmap, cap=cap, **options)
        controller_amplitude = amplitude
        if lca_coefficients_rb is not None:
            if (tuple(float(x) for x in origin_yx) != (0., 0.)
                    or covariance.shape[-2:] != (flow.shape[2] * 2, flow.shape[3] * 2)):
                raise ValueError('Integrated LCA requires the full reference output at origin zero; warp tiles on complete RGB sources')
            learned = warp_rgb_tile(learned, lca_coefficients_rb)
            legacy = warp_rgb_tile(legacy, lca_coefficients_rb)
            amplitude = local_amplitude(legacy)
        rgb = self.refinenet.factorized(torch.cat((learned, legacy), 1), amplitude) if self.refinement else learned
        result = {"rgb": rgb, "learned": learned, "legacy": legacy}
        if diagnostics:
            result.update(phase=evidence["phase"], controller_input=feature, qmap=qmap, amplitude=amplitude,
                          controller_amplitude=controller_amplitude)
        return result


def load_recovered_jsr(weights_dir: str | Path, k: int, *, refinement=True) -> RecoveredJSR:
    """Verify all three artifacts and apply the published K-specific routing."""
    if isinstance(k, bool):
        raise ValueError("k must be an integer from 1 to 14")
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError("k must be an integer from 1 to 14") from exc
    if not 1 <= k <= 14:
        raise ValueError("k must be an integer from 1 to 14")
    root = Path(weights_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    route = dict(manifest["routes"][str(k)], static_affine=manifest["static_affine"])
    paths = {}
    for role, name in route.items():
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("invalid weight filename in route")
        path = root / name
        verify_manifest_file(path, manifest_path)
        paths[role] = path
    controller = load_controller_npz(paths["controller"])
    refine = load_refinenet_npz(paths["refinenet"] if "refinenet" in paths else paths["rarm"])
    mean, std = load_static_affine_npz(paths["static_affine"])
    model = RecoveredJSR(refine_width=refine.inp.out_channels, refine_blocks=len(refine.body), refinement=refinement)
    model.controller, model.refinenet = controller, refine
    model.expected_frames = k
    model.feature_mean.copy_(mean)
    model.feature_std.copy_(std)
    return model.eval()
