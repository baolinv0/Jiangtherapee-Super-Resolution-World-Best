"""Trainable JSR modules and explicitly independent RAW-to-RGB assembly.

Controller/RefineNet layer contracts are transcribed from the public WGPU
implementation. The RAW statistics and interpretation of Controller outputs
are independent assumptions in frontend.py. Module weight compatibility is
not complete-pipeline numerical compatibility with the author's executable.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .frontend import canonicalize, controlled_fusion, finish_feature, local_amplitude, phase_splat

REFERENCE_AMPLITUDE = 0.34523068818006963


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())


class Controller(nn.Module):
    """Published depth-three 151→36 U-Net (zero-padded convolutions)."""
    def __init__(self, width: int = 32):
        super().__init__()
        if width < 1:
            raise ValueError("Controller width must be positive")
        self.enc = nn.ModuleList([_double_conv(151, width), _double_conv(width, width * 2), _double_conv(width * 2, width * 4)])
        self.mid = _double_conv(width * 4, width * 8)
        self.dec = nn.ModuleList([_double_conv(width * 12, width * 4), _double_conv(width * 6, width * 2), _double_conv(width * 3, width)])
        self.out = nn.Conv2d(width, 36, 1)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1] != 151:
            raise ValueError("Controller input must be [B,151,H,W]")
        height, width = value.shape[-2:]
        # Only sub-eight dimensions need extension; ordinary odd dimensions
        # use exactly upstream floor pooling and resize-to-skip behavior.
        value = F.pad(value, (0, max(0, 8 - width), 0, max(0, 8 - height)))
        skips = []
        for block in self.enc:
            value = block(value)
            skips.append(value)
            value = F.avg_pool2d(value, 2)
        value = self.mid(value)
        for block, skip in zip(self.dec, reversed(skips)):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat([value, skip], 1))
        return self.out(value)[..., :height, :width]


class PreActResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.c1 = nn.Conv2d(width, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.c2(F.gelu(self.c1(F.gelu(value))))


class RefineNet(nn.Module):
    """Published GreenFiLMColorDiffRArm; released K4 is width116/8 blocks."""
    def __init__(self, width: int = 116, blocks: int = 8):
        super().__init__()
        if min(width, blocks) < 1:
            raise ValueError("RefineNet width/blocks must be positive")
        self.inp = nn.Conv2d(6, width, 3, padding=1)
        self.guide = nn.Conv2d(2, width, 3, padding=1)
        self.body = nn.ModuleList([PreActResidual(width) for _ in range(blocks)])
        self.gammas = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(blocks)])
        self.betas = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(blocks)])
        self.out = nn.Conv2d(width, 3, 1)

    def forward(self, value: Tensor) -> Tensor:
        """Return dG, d(R-G), d(B-G), without factorization/render clipping."""
        if value.ndim != 4 or value.shape[1] != 6:
            raise ValueError("RefineNet input must be [B,6,H,W]")
        hidden, guide = self.inp(value), self.guide(value[:, [1, 4]])
        for block, gamma, beta in zip(self.body, self.gammas, self.betas):
            hidden = block(hidden)
            hidden = hidden * (1 + 0.1 * torch.tanh(gamma(guide))) + beta(guide)
        return self.out(hidden)

    def factorized(self, value: Tensor, amplitude: Tensor, reference_amplitude: float = REFERENCE_AMPLITUDE) -> Tensor:
        if reference_amplitude <= 0:
            raise ValueError("reference_amplitude must be positive")
        scale = amplitude / reference_amplitude
        denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
        normalized = value / denominator
        delta = self(normalized)
        green = normalized[:, 1:2] + delta[:, 0:1]
        red = green + normalized[:, 0:1] - normalized[:, 1:2] + delta[:, 1:2]
        blue = green + normalized[:, 2:3] - normalized[:, 1:2] + delta[:, 2:3]
        return torch.cat([red, green, blue], 1).clamp(0, 1) * scale


class JSRModel(nn.Module):
    """Independent trainable assembly, not an official-weight inference API.

    raw: [B,K,1,H,W], RGGB, black-subtracted linear units.
    shifts: [B,K,2] in native pixels; frame(x,y) samples ref(x+dx,y+dy).
    Returns rgb/legacy/learned [B,3,scale*H,scale*W]. No hidden RAW clipping.
    Defaults match the released module widths; the front end remains inferred.
    """
    def __init__(self, scale: int = 2, controller_width: int = 32, refine_width: int = 116, refine_blocks: int = 8, normalization: bool = True, refinement: bool = True):
        super().__init__()
        self.scale, self.normalization, self.refinement = scale, normalization, refinement
        self.controller = Controller(controller_width)
        self.refinenet = RefineNet(refine_width, refine_blocks)
        # An explicit identity affine is an engineering choice; original
        # calibration statistics are only meaningful with original features.
        self.register_buffer("feature_mean", torch.zeros(1, 151, 1, 1))
        self.register_buffer("feature_std", torch.ones(1, 151, 1, 1))
        # Start near linear fusion while permitting gradients through all
        # layers from the first step (a zero output layer would block them).
        with torch.no_grad():
            self.refinenet.out.weight.mul_(0.01)
            self.refinenet.out.bias.zero_()

    def set_feature_affine(self, mean: Tensor, std: Tensor) -> None:
        """Set training-derived calibration; buffers persist in state_dict."""
        if mean.shape != self.feature_mean.shape or std.shape != self.feature_std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("affine requires finite [1,151,1,1] and positive std")
        with torch.no_grad():
            self.feature_mean.copy_(mean)
            self.feature_std.copy_(std)

    def forward(self, raw: Tensor, shifts: Tensor) -> dict[str, Tensor]:
        evidence = phase_splat(raw, shifts, self.scale)
        legacy = evidence["legacy"]
        feature = finish_feature(evidence["phase"], legacy)
        if self.normalization:
            feature, amplitude = canonicalize(feature, legacy)
        else:
            amplitude = local_amplitude(legacy)
        logits = self.controller((feature - self.feature_mean) / self.feature_std)
        learned = controlled_fusion(logits, evidence["sum"], evidence["count"], legacy)
        if self.refinement:
            pair = torch.cat([learned, legacy], 1)
            # Published ordering is first RGB as the residual reconstruction
            # anchor; this assembly chooses learned first, legacy second.
            factor = amplitude if self.normalization else torch.full_like(amplitude, REFERENCE_AMPLITUDE)
            rgb = self.refinenet.factorized(pair, factor)
            # Maintain a physically zero result even in normalization ablation.
            rgb = torch.where(amplitude > 0, rgb, torch.zeros_like(rgb))
        else:
            rgb = learned
        return {"rgb": rgb, "legacy": legacy, "learned": learned}
