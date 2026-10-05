"""Simple estimated global-translation baseline, not the author's aligner."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def estimate_translations(raw):
    """Phase correlation on Bayer-cell green averages, native-pixel dx/dy.

    Fails on moving objects/occlusions and cannot estimate affine or dense motion.
    A three-point peak fit gives subcell shifts; oracle/estimated tracks stay separate.
    """
    if raw.ndim != 5 or raw.shape[2] != 1 or min(raw.shape[-2:]) < 8:
        raise ValueError("expected B,K,1,H,W raw with H,W>=8")
    g = .5 * (raw[:, :, 0, 0::2, 1::2] + raw[:, :, 0, 1::2, 0::2])
    h, w = g.shape[-2:]
    window = torch.hann_window(h, periodic=False, device=raw.device)[:, None] * torch.hann_window(w, periodic=False, device=raw.device)[None, :]
    g = (g - g.mean((-2, -1), keepdim=True)) * window
    spectra = torch.fft.rfft2(g)
    cross = spectra[:, :1] * spectra.conj()
    corr = torch.fft.irfft2(cross / cross.abs().clamp_min(1e-12), s=(h, w))
    result = torch.zeros(raw.shape[0], raw.shape[1], 2, device=raw.device)
    for b in range(raw.shape[0]):
        for k in range(1, raw.shape[1]):
            c = corr[b, k]
            idx = int(c.argmax())
            py, px = idx // w, idx % w
            def offset(a, m, z):
                denom = a - 2 * m + z
                return (0.5 * (a - z) / denom).clamp(-.5, .5) if abs(float(denom)) > 1e-12 else torch.zeros_like(m)
            dy = offset(c[(py - 1) % h, px], c[py, px], c[(py + 1) % h, px])
            dx = offset(c[py, (px - 1) % w], c[py, px], c[py, (px + 1) % w])
            result[b, k, 0] = 2 * ((px if px <= w // 2 else px - w) + dx)
            result[b, k, 1] = 2 * ((py if py <= h // 2 else py - h) + dy)
    return result


def single_frame_baseline(raw, scale=2):
    """Bilinear demosaic in sensor coordinates, then bilinear upscale."""
    raw = raw[:, 0]  # B,1,H,W
    mask = torch.zeros(1, 3, *raw.shape[-2:], device=raw.device, dtype=raw.dtype)
    mask[:, 0, 0::2, 0::2] = 1
    mask[:, 1, 0::2, 1::2] = 1
    mask[:, 1, 1::2, 0::2] = 1
    mask[:, 2, 1::2, 1::2] = 1
    kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=raw.device)[None, None].expand(3, 1, -1, -1)
    rgb = F.conv2d(raw * mask, kernel, padding=1, groups=3) / F.conv2d(mask, kernel, padding=1, groups=3).clamp_min(1e-8)
    return F.interpolate(rgb, scale_factor=scale, mode="bilinear", align_corners=False)
