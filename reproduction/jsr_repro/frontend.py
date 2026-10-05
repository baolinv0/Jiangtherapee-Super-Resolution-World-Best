"""RAW front end for the independent reproduction.

CONFIRMED: local-V7 amplitude, 151-channel layout, and homogeneous degrees
come from upstream local_v7_frontend.py. INFERRED: sample splatting, phase
statistics, and 36-control interpretation below are an independently defined
replacement for the unpublished Tap producer/consumer. Official Controller
weights must not be assumed compatible with these statistics.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

FEATURE_DIM = 151
FRONTEND_VERSION = "independent-phase-splat-v1"


def safe_sqrt(value: Tensor) -> Tensor:
    """Exact nonnegative square root with a finite zero subgradient."""
    positive = value > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, value, torch.ones_like(value))), torch.zeros_like(value))


def reflect_pad(value: Tensor, radius: int) -> Tensor:
    """OpenCV REFLECT_101, including dimensions smaller than the padding."""
    def indices(length: int) -> Tensor:
        if length == 1:
            return torch.zeros(length + 2 * radius, device=value.device, dtype=torch.long)
        coordinate = torch.arange(-radius, length + radius, device=value.device)
        coordinate = coordinate.remainder(2 * (length - 1))
        return torch.minimum(coordinate, 2 * (length - 1) - coordinate).long()
    return value.index_select(-2, indices(value.shape[-2])).index_select(-1, indices(value.shape[-1]))


def gaussian_blur(value: Tensor, radius: int, sigma: float) -> Tensor:
    coordinate = torch.arange(-radius, radius + 1, dtype=value.dtype, device=value.device)
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    padded = reflect_pad(value, radius)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    return F.conv2d(F.conv2d(padded, horizontal, groups=channels), vertical, groups=channels)


def local_amplitude(legacy: Tensor) -> Tensor:
    """Published local-V7 RMS: RGB energy, Gaussian radius16/sigma4."""
    if legacy.ndim != 4 or legacy.shape[1] != 3:
        raise ValueError("legacy must be [B,3,H,W]")
    return safe_sqrt(gaussian_blur(legacy.square().mean(1, keepdim=True), 16, 4.0))


def finish_feature(phase: Tensor, legacy: Tensor) -> Tensor:
    """Match the published feature completion (Sobel, tensor smoothing)."""
    if phase.ndim != 4 or phase.shape[1] != 144 or phase.shape[0] != legacy.shape[0] or phase.shape[-2:] != legacy.shape[-2:]:
        raise ValueError("phase must be [B,144,H,W] matching legacy")
    grey = (legacy * legacy.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
    sobel = legacy.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = F.conv2d(reflect_pad(grey, 1), sobel[None, None])
    gy = F.conv2d(reflect_pad(grey, 1), sobel.t()[None, None])
    structure = gaussian_blur(torch.cat([gx.square(), gx * gy, gy.square()], 1), 8, 2.0)
    return torch.cat([phase, legacy, structure, torch.zeros_like(grey)], 1)


def canonicalize(feature: Tensor, legacy: Tensor) -> tuple[Tensor, Tensor]:
    amplitude = local_amplitude(legacy)
    denominator = torch.where(amplitude > 0, amplitude, torch.ones_like(amplitude))
    degree = torch.zeros(FEATURE_DIM, dtype=feature.dtype, device=feature.device)
    for color in range(3):
        degree[color * 48 + 16:color * 48 + 48] = 1
    degree[144:147] = 1
    degree[147:150] = 2
    degree[150] = 1
    return feature / denominator.pow(degree[None, :, None, None]), amplitude


def make_controller_input(phase: Tensor, legacy: Tensor, mean: Tensor | None = None, std: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Published feature completion/canonicalization, optional frozen affine.

    Omitted affine is identity. Official affine is suitable for module parity
    on original phase features, not for the independently inferred splat data.
    """
    canonical, amplitude = canonicalize(finish_feature(phase, legacy), legacy)
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together")
    if mean is not None:
        if mean.shape != (1, 151, 1, 1) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid affine; expected finite [1,151,1,1] and positive std")
        canonical = (canonical - mean.to(canonical)) / std.to(canonical)
    return canonical, amplitude


def _ratio(numerator: Tensor, denominator: Tensor, fallback: Tensor | float = 0.0) -> Tensor:
    positive = denominator > 0
    return torch.where(positive, numerator / torch.where(positive, denominator, torch.ones_like(denominator)), torch.as_tensor(fallback, dtype=numerator.dtype, device=numerator.device))


def phase_splat(raw: Tensor, shifts: Tensor, scale: int = 2, geometry=None, validity: Tensor | None = None) -> dict[str, Tensor]:
    """Independent 16-phase count/mean/RMS reconstruction statistics.

    Input raw is black-subtracted RGGB [B,K,1,H,W]. (dx,dy) means a
    sample at (x,y) in that frame measures reference (x+dx,y+dy).
    Native sensel centers map to HR coordinates
    ((x+dx+0.5)*scale-0.5, (y+dy+0.5)*scale-0.5). This matches
    synthesis by averaging scale*scale HR cells into each native sensel.
    Each sample is bilinearly splatted, then Gaussian-smoothed on the HR
    grid (radius=2*scale, sigma=scale). Phase is the fractional *native*
    projected coordinate quantized into a 4x4 grid. Per-color phase channels
    are [16 counts / 14, 16 means, 16 RMS]. These semantics are inferred.
    """
    if raw.ndim != 5 or raw.shape[2] != 1 or not raw.is_floating_point():
        raise ValueError("raw must be floating [B,K,1,H,W]")
    if validity is not None and (validity.shape != raw.shape or not torch.isfinite(validity).all() or ((validity != 0) & (validity != 1)).any()):
        raise ValueError("validity must be a finite binary mask matching RAW")
    batch, frames, _, height, width = raw.shape
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("RGGB height and width must be even and at least 2")
    if not 1 <= frames <= 14:
        raise ValueError("supported burst length is 1..14")
    if scale not in (1, 2, 3, 4):
        raise ValueError("scale must be an integer from 1 to 4")
    if shifts.shape != (batch, frames, 2) or shifts.device != raw.device:
        raise ValueError("shifts must be [B,K,2] on the RAW device")
    if not torch.isfinite(raw).all() or not torch.isfinite(shifts).all():
        raise ValueError("RAW and shifts must be finite")
    if not torch.allclose(shifts[:, 0], torch.zeros_like(shifts[:, 0]), atol=1e-7, rtol=0):
        raise ValueError("frame zero must define the reference with shift (0,0)")
    if geometry is not None:
        from .geometry import validate_geometry
        validate_geometry(geometry, raw)
    out_h, out_w = height * scale, width * scale
    yy, xx = torch.meshgrid(torch.arange(height, device=raw.device), torch.arange(width, device=raw.device), indexing="ij")
    color = torch.where((yy % 2 == 0) & (xx % 2 == 0), 0, torch.where((yy % 2 == 1) & (xx % 2 == 1), 2, 1)).reshape(-1)
    counts, sums, squares = [], [], []
    for b in range(batch):
        count = raw.new_zeros(3 * 16 * out_h * out_w)
        total = torch.zeros_like(count)
        second = torch.zeros_like(count)
        for k in range(frames):
            if geometry is None:
                native_x = xx.to(raw.dtype).reshape(-1) + shifts[b, k, 0]
                native_y = yy.to(raw.dtype).reshape(-1) + shifts[b, k, 1]
            else:
                native_x, native_y = geometry.forward[b, k].reshape(-1, 2).unbind(-1)
            phase = (native_y.remainder(1) * 4).floor().long().clamp(0, 3) * 4 + (native_x.remainder(1) * 4).floor().long().clamp(0, 3)
            px, py = (native_x + 0.5) * scale - 0.5, (native_y + 0.5) * scale - 0.5
            x0, y0 = px.floor(), py.floor()
            value = raw[b, k, 0].reshape(-1)
            for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1)):
                x, y = x0.long() + ox, y0.long() + oy
                weight = (1 - (px - x.to(px.dtype)).abs()) * (1 - (py - y.to(py.dtype)).abs())
                valid = (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
                weight = weight * valid.to(weight.dtype)
                if validity is not None:
                    weight = weight * validity[b, k, 0].reshape(-1).to(weight)
                index = ((color * 16 + phase) * out_h + y.clamp(0, out_h - 1)) * out_w + x.clamp(0, out_w - 1)
                count.scatter_add_(0, index, weight)
                total.scatter_add_(0, index, weight * value)
                second.scatter_add_(0, index, weight * value.square())
        counts.append(count.view(1, 48, out_h, out_w))
        sums.append(total.view(1, 48, out_h, out_w))
        squares.append(second.view(1, 48, out_h, out_w))
    counts = gaussian_blur(torch.cat(counts), 2 * scale, float(scale)).view(batch, 3, 16, out_h, out_w)
    sums = gaussian_blur(torch.cat(sums), 2 * scale, float(scale)).view_as(counts)
    squares = gaussian_blur(torch.cat(squares), 2 * scale, float(scale)).view_as(counts)
    means, rms = _ratio(sums, counts), safe_sqrt(_ratio(squares, counts))
    total_count, total_sum = counts.sum(2), sums.sum(2)
    # No cross-color borrowing: fallback is that color's burst mean.
    per_color = torch.stack([raw[:, :, 0].reshape(batch, frames, -1)[:, :, color == c].mean((1, 2)) for c in range(3)], 1)[..., None, None]
    legacy = _ratio(total_sum, total_count, per_color)
    phase = torch.cat([counts / 14.0, means, rms], 2).reshape(batch, 144, out_h, out_w)
    return {"phase": phase, "legacy": legacy, "count": total_count, "sum": total_sum}


def controlled_fusion(logits: Tensor, total: Tensor, count: Tensor, legacy: Tensor) -> Tensor:
    """INFERRED: 36 logits = 12 taps per color, 3x4 HR neighborhoods.

    Rows are [-1,0,1], columns [-1,0,1,2], with reflect101 boundaries.
    Softmax weights multiply smoothed sample sums/counts, so the operation
    is intensity-linear for fixed logits. No-evidence pixels use legacy.
    """
    batch, _, height, width = total.shape
    weights = logits.reshape(batch, 3, 12, height, width).softmax(2)
    total_pad, count_pad = reflect_pad(total, 2), reflect_pad(count, 2)
    numerator, denominator = torch.zeros_like(total), torch.zeros_like(count)
    tap = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1, 2):
            numerator = numerator + weights[:, :, tap] * total_pad[:, :, 2 + dy:2 + dy + height, 2 + dx:2 + dx + width]
            denominator = denominator + weights[:, :, tap] * count_pad[:, :, 2 + dy:2 + dy + height, 2 + dx:2 + dx + width]
            tap += 1
    return _ratio(numerator, denominator, legacy)
