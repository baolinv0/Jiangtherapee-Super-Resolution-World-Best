"""Differentiable reconstruction stages in the released v9.8 coordinate system.

The legacy independent-phase-splat-v1 checkpoint path is separate. This API
consumes registration and covariance at their release module boundaries;
it does not estimate either of them. See docs/REA_RECOVERY.md for provenance.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .frontend import safe_sqrt

FRONTEND_VERSION = "release-v98-continuous-taps-v1"


def _prepare(frames, flow, covariance, origin_yx, fshift, color_index):
    if frames.ndim != 4 or not frames.is_floating_point():
        raise ValueError("frames must be floating [B,K,Hframe,Wframe]")
    b, k, nf, mf = frames.shape
    if min(b, k, nf, mf) < 1 or k > 14:
        raise ValueError("nonempty frames and burst length 1..14 required")
    if flow.ndim != 5 or flow.shape[:2] != (b, k) or flow.shape[-1] != 2 or min(flow.shape[2:4]) < 1:
        raise ValueError("flow must be [B,K,Hreference,Wreference,2] (dx,dy)")
    if covariance.ndim != 4 or covariance.shape[:2] != (b, 3) or min(covariance.shape[-2:]) < 1:
        raise ValueError("covariance must be [B,3,Hout,Wout] (xx,xy,yy)")
    for name, value in (("frames", frames), ("flow", flow), ("covariance", covariance)):
        if value.device != frames.device or value.dtype != frames.dtype or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite with the same device/dtype as frames")
    origin = torch.as_tensor(origin_yx, dtype=frames.dtype, device=frames.device)
    if origin.shape != (2,) or not torch.isfinite(origin).all():
        raise ValueError("origin_yx must contain two finite native coordinates")
    if fshift is None:
        fshift = frames.new_tensor([0., 0., nf, mf]).expand(b, k, 4)
    else:
        fshift = torch.as_tensor(fshift, dtype=frames.dtype, device=frames.device)
    if (fshift.shape != (b, k, 4) or not torch.isfinite(fshift).all()
            or (fshift[..., 2:] < 1).any() or (fshift[..., 2:] != fshift[..., 2:].trunc()).any()
            or (fshift[..., 2] > nf).any() or (fshift[..., 3] > mf).any()):
        raise ValueError("fshift must be [B,K,4]: y offset, x offset, valid height, valid width")
    if color_index is None:
        y, x = torch.meshgrid(torch.arange(nf, device=frames.device), torch.arange(mf, device=frames.device), indexing="ij")
        color_index = torch.where((y % 2 == 0) & (x % 2 == 0), 0,
                                  torch.where((y % 2 == 1) & (x % 2 == 1), 2, 1))
    else:
        color_index = torch.as_tensor(color_index, device=frames.device)
    if (color_index.shape != (nf, mf) or color_index.is_floating_point()
            or (color_index < 0).any() or (color_index > 2).any()):
        raise ValueError("color_index must be integer [Hframe,Wframe] with R=0,G=1,B=2")
    ho, wo = covariance.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(ho, device=frames.device, dtype=frames.dtype),
                            torch.arange(wo, device=frames.device, dtype=frames.dtype), indexing="ij")
    oy, ox = yy / 2 + origin[0], xx / 2 + origin[1]
    iy0, ix0 = oy.trunc().long().clamp(0, flow.shape[2] - 1), ox.trunc().long().clamp(0, flow.shape[3] - 1)
    return fshift, color_index.long(), oy, ox, iy0, ix0


def _samples(frames, flow, covariance, prepared):
    fshift, ci, oy, ox, iy0, ix0 = prepared
    batch_index = torch.arange(frames.shape[0], device=frames.device)[:, None, None]
    for k in range(frames.shape[1]):
        displacement = flow[:, k, iy0, ix0]
        my = oy + displacement[..., 1] + fshift[:, k, 0, None, None]
        mx = ox + displacement[..., 0] + fshift[:, k, 1, None, None]
        cy, cx = my.floor(), mx.floor()
        for ay in range(-1, 3):
            for ax in range(-1, 3):
                yy, xx = cy + ay, cx + ax
                # Clamp the array access, but keep unclamped offsets in the
                # kernel and phase calculation. Duplicated edge samples count.
                iy = torch.minimum(yy.trunc().long().clamp_min(0), fshift[:, k, 2, None, None].long() - 1)
                ix = torch.minimum(xx.trunc().long().clamp_min(0), fshift[:, k, 3, None, None].long() - 1)
                dy, dx = yy - my, xx - mx
                z = covariance[:, 0] * dx * dx + 2 * covariance[:, 1] * dx * dy + covariance[:, 2] * dy * dy
                yield ci[iy, ix], frames[batch_index, k, iy, ix], dy, dx, z


def phase_statistics(frames: Tensor, flow: Tensor, covariance: Tensor, *,
                     origin_yx=(0., 0.), fshift=None, color_index=None) -> dict[str, Tensor]:
    """Gather 4x4 RAW neighborhoods and return count/mean/std and legacy RGB.

    Each HR output (y,x) maps to reference (y/2+y0,x/2+x0). Flow is
    reference-to-frame displacement in native pixels, sampled with truncation.
    fshift adds per-frame crop offsets before gathering. The phase of the
    *sample offset* has period two native pixels, quantized into 4x4 bins.
    Counts are multiplied by 14/K. No evidence smoothing or frame averaging
    occurs before gathering. RAW units are preserved, including negatives.
    """
    prepared = _prepare(frames, flow, covariance, origin_yx, fshift, color_index)
    b, k = frames.shape[:2]
    h, w = covariance.shape[-2:]
    counts = frames.new_zeros(b, 48, h, w)
    sums, squares = torch.zeros_like(counts), torch.zeros_like(counts)
    total_count, total_sum = frames.new_zeros(b, 3, h, w), frames.new_zeros(b, 3, h, w)
    for color, value, dy, dx, z in _samples(frames, flow, covariance, prepared):
        weight = torch.exp(-.5 * z)
        # Explicit floor/remainder rather than projected-position phase.
        py = ((dy - torch.floor(dy / 2) * 2) * 2).long().clamp_max(3)
        px = ((dx - torch.floor(dx / 2) * 2) * 2).long().clamp_max(3)
        index = (color * 16 + py * 4 + px)[:, None]
        weighted_value = weight * value
        counts = counts.scatter_add(1, index, weight[:, None])
        sums = sums.scatter_add(1, index, weighted_value[:, None])
        squares = squares.scatter_add(1, index, (weighted_value * value)[:, None])
        total_count = total_count.scatter_add(1, color[:, None], weight[:, None])
        total_sum = total_sum.scatter_add(1, color[:, None], weighted_value[:, None])
    counts, sums, squares = [x.reshape(b, 3, 16, h, w) for x in (counts, sums, squares)]
    mean = sums / counts.clamp_min(1e-9)
    std = safe_sqrt((squares / counts.clamp_min(1e-9) - mean.square()).clamp_min(0))
    if (total_count <= 0).any():
        raise ValueError("legacy reconstruction has no positive support for a color")
    phase = torch.cat((counts * (14. / k), mean, std), 2).reshape(b, 144, h, w)
    return {"phase": phase, "legacy": total_sum / total_count, "count": total_count, "sum": total_sum}


def learned_merge(frames: Tensor, flow: Tensor, covariance: Tensor, qmap: Tensor, *,
                  origin_yx=(0., 0.), fshift=None, color_index=None, cap: float = 5.) -> Tensor:
    """Continuous, bounded log-kernel correction, 12 basis coefficients/color.

    basis = [1,dy,dx,dy²,dx²,dy*dx,r²,r⁴,cos(2πdy),sin(2πdy),
             cos(2πdx),sin(2πdx)]. Weights = exp(-z/2+cap*tanh(q·basis/cap)).
    This resamples RAW observations, rather than filtering an RGB evidence map.
    Gradients flow through RAW, covariance, qmap and continuous coordinates;
    sample/color/bin selection has the release's discrete semantics.
    """
    prepared = _prepare(frames, flow, covariance, origin_yx, fshift, color_index)
    b = frames.shape[0]
    h, w = covariance.shape[-2:]
    if (qmap.shape != (b, 36, h, w) or qmap.device != frames.device or qmap.dtype != frames.dtype
            or not torch.isfinite(qmap).all()):
        raise ValueError("qmap must be finite [B,36,Hout,Wout] with frames device/dtype")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")
    q = qmap.reshape(b, 3, 12, h, w)
    numerator, denominator = frames.new_zeros(b, 3, h, w), frames.new_zeros(b, 3, h, w)
    for color, value, dy, dx, z in _samples(frames, flow, covariance, prepared):
        coefficients = q.gather(1, color[:, None, None].expand(b, 1, 12, h, w))[:, 0]
        r2 = dy * dy + dx * dx
        # Preserve the released polynomial multiplication/accumulation order.
        raw = coefficients[:, 0] + coefficients[:, 1] * dy + coefficients[:, 2] * dx
        raw = raw + coefficients[:, 3] * dy * dy + coefficients[:, 4] * dx * dx
        raw = raw + coefficients[:, 5] * dy * dx + coefficients[:, 6] * r2 + coefficients[:, 7] * r2 * r2
        raw = raw + coefficients[:, 8] * torch.cos(2 * math.pi * dy) + coefficients[:, 9] * torch.sin(2 * math.pi * dy)
        raw = raw + coefficients[:, 10] * torch.cos(2 * math.pi * dx) + coefficients[:, 11] * torch.sin(2 * math.pi * dx)
        weight = torch.exp(-.5 * z + cap * torch.tanh(raw / cap))
        numerator = numerator.scatter_add(1, color[:, None], (weight * value)[:, None])
        denominator = denominator.scatter_add(1, color[:, None], weight[:, None])
    supported = denominator > 0
    return torch.where(supported, numerator / torch.where(supported, denominator, torch.ones_like(denominator)), torch.zeros_like(numerator))
