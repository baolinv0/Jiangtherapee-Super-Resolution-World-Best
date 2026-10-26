"""Fixed-coefficient radial LCA tile warp recovered from v9.8 numeric probes.

The estimator is outside this module. The verified EXE bundles OpenCV 5.0.0.93,
whose FP32 remap uses continuous fractional cubic weights. Pixel and supplied
coefficient gradients are supported away from sample-selection boundaries.
"""
from __future__ import annotations

import operator

import torch
from torch import Tensor

LCA_VERSION = 'release-v98-fixed-radial-lca-v1'


def _reflect(index: Tensor, length: int) -> Tensor:
    index = index.remainder(2 * length)
    return torch.where(index < length, index, 2 * length - index - 1)


def _cubic(fraction: Tensor) -> tuple[Tensor, ...]:
    # OpenCV interpolateCubic, A=-0.75, including its residual fourth weight.
    a = -.75
    p = fraction + 1
    w0 = ((a * p - 5 * a) * p + 8 * a) * p - 4 * a
    w1 = ((a + 2) * fraction - (a + 3)) * fraction * fraction + 1
    p = 1 - fraction
    w2 = ((a + 2) * p - (a + 3)) * p * p + 1
    return w0, w1, w2, 1 - w0 - w1 - w2


def warp_rgb_tile(source: Tensor, coefficients_rb, *, y0=0, x0=0, height=None,
                  width=None, applied=True) -> Tensor:
    """Warp a global-coordinate tile using externally supplied R/B scales.

    source is finite FP32 [B,3,H,W]; coefficients_rb is [2] or [B,2].
    sample = center + (destination-center)*(1+coefficient), with center
    ((W-1)/2,(H-1)/2), cubic interpolation, BORDER_REFLECT (edge repeated).
    Applied correction clamps all three output channels to nonnegative values,
    even when both coefficients are zero. Disabled correction copies the ROI
    unchanged, including negative values, exactly as the released entry does.
    """
    if (source.ndim != 4 or source.shape[1] != 3 or min(source.shape) < 1
            or source.dtype != torch.float32 or not torch.isfinite(source).all()):
        raise ValueError('source must be finite FP32 [B,3,H,W]')
    b, _, h, w = source.shape
    try:
        y0, x0 = operator.index(y0), operator.index(x0)
        height = h - y0 if height is None else operator.index(height)
        width = w - x0 if width is None else operator.index(width)
    except TypeError as exc:
        raise ValueError('tile origin and dimensions must be integers') from exc
    if y0 < 0 or x0 < 0 or height <= 0 or width <= 0 or y0 + height > h or x0 + width > w:
        raise ValueError('invalid LCA tile bounds')
    roi = source[:, :, y0:y0 + height, x0:x0 + width]
    if not applied:
        return roi.clone()
    coefficients = torch.as_tensor(coefficients_rb, device=source.device, dtype=source.dtype)
    if coefficients.shape == (2,):
        coefficients = coefficients[None].expand(b, 2)
    if coefficients.shape != (b, 2) or not torch.isfinite(coefficients).all():
        raise ValueError('coefficients_rb must be finite [2] or [B,2]')
    yy, xx = torch.meshgrid(torch.arange(y0, y0 + height, dtype=source.dtype, device=source.device),
                            torch.arange(x0, x0 + width, dtype=source.dtype, device=source.device), indexing='ij')
    batch = torch.arange(b, device=source.device)[:, None, None]
    output = []
    for channel in range(3):
        if channel == 1:
            output.append(roi[:, 1])
            continue
        scale = 1 + coefficients[:, 0 if channel == 0 else 1, None, None]
        map_x = (xx - (w - 1) * .5) * scale + (w - 1) * .5
        map_y = (yy - (h - 1) * .5) * scale + (h - 1) * .5
        ix, iy = map_x.floor().long(), map_y.floor().long()
        wx, wy = _cubic(map_x - ix.to(source.dtype)), _cubic(map_y - iy.to(source.dtype))
        value = torch.zeros_like(map_x)
        for ry in range(4):
            row = torch.zeros_like(map_x)
            for rx in range(4):
                sample = source[batch, channel, _reflect(iy + ry - 1, h), _reflect(ix + rx - 1, w)]
                row = row + sample * wx[rx]
            value = value + row * wy[ry]
        output.append(value)
    return torch.stack(output, 1).clamp_min(0.)
