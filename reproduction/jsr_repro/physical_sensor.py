"""Linear electron/DN sensor simulation inspired by the EMVA 1288 model.

This implements a declared approximation, not EMVA certification or the author's
camera calibration. Source: EMVA1288Linear_4.0Release.pdf, sections 2.1--2.4.
Signal units map one normalized unit to (white_dn-black_dn)*gain_e_per_dn
electrons before transmission/exposure. The profile therefore absorbs absolute
QE and sensitivity. Physical full-well and ADC clipping are separate operations.
"""
from __future__ import annotations

import math

import torch

from .data import mosaic_rggb


def capture_sensor(rgb_native, exposure, transmission, camera, generator,
                   noise=True, quantize=True, read_noise_bank=None):
    """Capture RGB [K,3,H,W] into RGGB [K,1,H,W] with electron-domain noise.

    Shot counts are drawn BEFORE full-well clipping, followed by read noise,
    gain/black offset, ADC quantization and clipping. ``saturation`` and
    ``black_invalid`` use observed DN only; ``signal_saturation`` is diagnostic
    ground truth and must never be supplied as a deployment model input.
    The variance input uses a clipped observed-electron plug-in estimate. It is
    an unsaturated linear-model approximation; saturated values need masking.

    Optional read_noise_bank [T>=2,H,W] is in electrons. Its per-pixel temporal
    mean is removed, preserving spatial correlation of each sampled residual
    frame. It replaces Gaussian read noise (does not add to it), and its
    empirical population variance supplies the variance prior. It deliberately
    does not simulate the removed DSNU or claim a physical sensor profile.
    """
    if rgb_native.ndim != 4 or rgb_native.shape[1] != 3 or min(rgb_native.shape) < 1 or any(s % 2 for s in rgb_native.shape[-2:]):
        raise ValueError("rgb_native requires [K,3,even_H,even_W]")
    if not rgb_native.is_floating_point() or not torch.isfinite(rgb_native).all() or (rgb_native < 0).any():
        raise ValueError("rgb_native must be finite nonnegative floating radiance")
    exposure = torch.as_tensor(exposure, dtype=rgb_native.dtype, device=rgb_native.device)
    transmission = torch.as_tensor(transmission, dtype=rgb_native.dtype, device=rgb_native.device)
    if exposure.shape != (len(rgb_native),) or not torch.isfinite(exposure).all() or (exposure <= 0).any():
        raise ValueError("exposure must contain K finite positive relative exposures")
    if transmission.shape != (3,) or not torch.isfinite(transmission).all() or (transmission <= 0).any() or (transmission > 1).any():
        raise ValueError("transmission must be three finite numbers in (0,1]")
    names = ("gain_e_per_dn", "read_noise_e", "black_dn", "white_dn", "full_well_e")
    try:
        gain, read_e, black, white, full_well = [float(camera[name]) for name in names]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera requires explicit gain_e_per_dn, read_noise_e, black_dn, white_dn, full_well_e") from exc
    if not all(math.isfinite(v) for v in (gain, read_e, black, white, full_well)) or gain <= 0 or read_e < 0 or black < 0 or white <= black or full_well <= 0:
        raise ValueError("invalid electron/DN sensor profile")
    if quantize and white != int(white):
        raise ValueError("quantized ADC white_dn must be an integer code")
    span = white - black
    electrons_per_unit = span * gain
    clean_rgb = rgb_native * exposure[:, None, None, None] * transmission[None, :, None, None]
    clean = mosaic_rggb(clean_rgb)
    signal_e = clean * electrons_per_unit
    # One PTC/gain law for all CFA colors; response/transmission controls arrival.
    counts = torch.poisson(signal_e, generator=generator) if noise else signal_e.clone()
    captured_e = counts.clamp_max(full_well)
    read_variance_e = torch.zeros_like(signal_e)
    if read_noise_bank is not None:
        bank = torch.as_tensor(read_noise_bank, dtype=rgb_native.dtype, device=rgb_native.device)
        if bank.ndim != 3 or bank.shape[0] < 2 or bank.shape[1:] != rgb_native.shape[-2:] or not torch.isfinite(bank).all():
            raise ValueError("read_noise_bank must be finite electron-domain [T>=2,H,W] with matching spatial shape")
        bank = bank - bank.mean(0, keepdim=True)
        if noise:
            indices = torch.randint(len(bank), (len(rgb_native),), generator=generator, device=rgb_native.device)
            captured_e = captured_e + bank[indices, None]
            read_variance_e = bank.var(0, unbiased=False)[None, None].expand_as(signal_e)
    elif noise:
        captured_e = captured_e + torch.randn(captured_e.shape, generator=generator,
                                              device=rgb_native.device, dtype=rgb_native.dtype) * read_e
        read_variance_e = torch.full_like(signal_e, read_e ** 2)
    observed_dn = captured_e / gain + black
    if quantize:
        observed_dn = observed_dn.round()
    observed_dn = observed_dn.clamp(0, white)
    raw = (observed_dn - black) / span
    # Detect both full-well and ADC saturation using calibrated observed levels.
    # One-code margin accounts for digitization, not an unobserved clean signal.
    sat_dn = min(white, black + full_well / gain)
    saturation = observed_dn >= (sat_dn - (0.5 if quantize else 0.))
    black_invalid = observed_dn < black
    observed_e = raw.clamp_min(0) * electrons_per_unit
    variance = ((observed_e if noise else torch.zeros_like(observed_e)) + read_variance_e) / electrons_per_unit ** 2
    if quantize:
        variance = variance + 1 / (12 * span ** 2)
    signal_saturation = signal_e >= min(full_well, electrons_per_unit)
    return {"raw": raw.float(), "variance": variance.float(),
            "saturation": saturation.float(), "valid": torch.ones_like(raw).float(),
            "black_invalid": black_invalid.float(),
            "signal_saturation": signal_saturation.float(),
            "clean_raw": clean.float(), "signal_e": signal_e.float(),
            "observed_dn": observed_dn.float(),
            "exposure": exposure.float(), "transmission": transmission.float()}
