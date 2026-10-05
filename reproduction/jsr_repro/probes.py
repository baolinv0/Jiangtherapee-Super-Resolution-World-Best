"""Diagnostic response/axial controls inspired by author descriptions.

These newly specified fixtures are not the author's undisclosed experiment
scripts. All numbers use normalized linear values; no learned gain fitting
is applied to image reconstruction metrics.
"""
from __future__ import annotations

import argparse
import math

import torch

from .data import mosaic_rggb
from .train import load_checkpoint
from .utils import seed_all, save_json


@torch.no_grad()
def run_probes(model, size=32, frames=4):
    model.eval()
    device = next(model.parameters()).device
    shifts = torch.zeros(1, frames, 2, device=device)
    # Static, repeat-frame diagnostic deliberately has no SR sampling diversity.
    raw = torch.full((1, frames, 1, size, size), .2, device=device)
    reference = model(raw, shifts)["rgb"]
    gains = [2 ** -9, 2 ** -6, 2 ** -3, .5, 1., 2.]
    equivariance = []
    for gain in gains:
        out = model(raw * gain, shifts)["rgb"]
        equivariance.append({"gain": gain, "max_abs_error": (out - reference * gain).abs().max().item(), "relative_rmse": ((out - reference * gain).square().mean().sqrt() / (reference * gain).square().mean().sqrt().clamp_min(1e-12)).item()})
    levels = torch.logspace(-5, math.log10(.5), 29, device=device)
    responses = torch.stack([model(raw * (level / .2), shifts)["rgb"][..., 12:-12, 12:-12].mean() for level in levels])
    design = torch.stack([levels, torch.ones_like(levels)], 1)
    coeff = torch.linalg.lstsq(design.double().cpu(), responses.double().cpu()).solution
    residual = responses.cpu().double() - design.cpu().double() @ coeff
    zero = model(torch.zeros_like(raw), shifts)["rgb"]
    axes = []
    for axis in ("x", "y"):
        for freq in (.2, .35):
            y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
            coordinate = x if axis == "x" else y
            stripe = .1 * torch.sin(2 * math.pi * freq * coordinate)
            rgb = torch.stack([.2 + stripe, torch.full_like(stripe, .2), .2 - stripe])[None]
            mosaic = mosaic_rggb(rgb)[:, None].expand(-1, frames, -1, -1, -1)
            out = model(mosaic, shifts)["rgb"][..., 12:-12, 12:-12]
            chroma = out[:, 0] - out[:, 1]
            orthogonal_dim = -2 if axis == "x" else -1
            cross_energy = (chroma - chroma.mean(orthogonal_dim, keepdim=True)).square().mean()
            # Amplitude least-squares fit at the *physical* frequency on HR centers.
            hr_y, hr_x = torch.meshgrid(torch.arange(out.shape[-2], device=device) + 12, torch.arange(out.shape[-1], device=device) + 12, indexing="ij")
            native_coordinate = ((hr_x if axis == "x" else hr_y) + .5) / model.scale - .5
            phase = 2 * math.pi * freq * native_coordinate
            matrix = torch.stack([torch.sin(phase).flatten(), torch.cos(phase).flatten(), torch.ones_like(phase).flatten()], 1).double().cpu()
            fit = torch.linalg.lstsq(matrix, chroma.flatten().double().cpu()).solution
            amplitude = float(torch.linalg.vector_norm(fit[:2]))
            axes.append({"axis": axis, "cycles_per_native_pixel": freq, "cross_axis_chroma_rms": cross_energy.sqrt().item(), "recovered_rg_amplitude": amplitude, "input_rg_amplitude": .1})
    return {"scope": "new constant-intensity and static axial diagnostics, not original JSR ring/optics protocol", "zero_max_abs": zero.abs().max().item(), "gain_equivariance": equivariance, "response": {"levels": levels.cpu().tolist(), "means": responses.cpu().tolist(), "affine_slope": coeff[0].item(), "affine_intercept": coeff[1].item(), "affine_residual_rmse": residual.square().mean().sqrt().item(), "identity_rmse": (responses - levels).square().mean().sqrt().item()}, "axial_controls": axes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    seed_all(1234)
    model, _ = load_checkpoint(args.checkpoint)
    save_json(args.output, run_probes(model))


if __name__ == "__main__":
    main()
