"""Wavelength-resolved scalar diffraction with explicitly assumed aberrations.

The pupil is circular and OPD coefficients are nanometers, not camera-specific
measurements. PSFs describe *optical* blur: integrating the finite reconstruction
grid cell below is numerical discretization and does not include native-sensel
integration. Finite FFT support and output support are truncated and normalized.
The field interpolation is an independent, energy-normalized approximation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _waves(wavelengths_nm):
    waves = torch.as_tensor(wavelengths_nm, dtype=torch.float32, device="cpu")
    if waves.ndim != 1 or len(waves) == 0 or not torch.isfinite(waves).all() or (waves <= 0).any():
        raise ValueError("wavelengths_nm must be a finite positive one-dimensional sequence")
    return waves


def spectral_psfs(wavelengths_nm, f_number, pitch_um, scale=2, radius=12,
                  field_xy=(0., 0.), aberrations_nm=None, lca_native=0.,
                  pupil_samples=64, fft_size=256):
    """Return normalized nonnegative [L,2*radius+1,2*radius+1] optical PSFs.

    Field coordinates are normalized image coordinates. ``defocus``,
    ``astigmatism`` and ``coma`` are coefficients of the unnormalized Zernike
    forms 2*rho²-1, rho²*cos(2*theta), and (3*rho³-2*rho)*cos(theta).
    Defocus is multiplied by (1+field_radius²), astigmatism by field_radius²,
    and coma by field_radius; the last two are oriented along the field angle.
    This analytic field law is an assumption, not a measured lens prescription.
    ``lca_native`` shifts a 700-nm PSF radially by that many native pixels at
    unit field radius; the 550-nm PSF has zero shift, 400-nm the opposite shift.
    Positive kernel displacement shifts an impulse in the same direction.
    """
    waves = _waves(wavelengths_nm)
    values = [f_number, pitch_um, scale, *field_xy, lca_native]
    if len(field_xy) != 2 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("optics parameters and the two field coordinates must be finite")
    if min(f_number, pitch_um, scale) <= 0 or radius < 1 or int(radius) != radius:
        raise ValueError("f_number, pitch_um, scale must be positive and radius a positive integer")
    if pupil_samples < 16 or pupil_samples % 2 or fft_size < 2 * pupil_samples or fft_size % 2:
        raise ValueError("use an even pupil_samples>=16 and even fft_size>=2*pupil_samples")
    coefficients = {"defocus": 0., "astigmatism": 0., "coma": 0.}
    for key, value in (aberrations_nm or {}).items():
        if key not in coefficients or not math.isfinite(float(value)):
            raise ValueError("aberrations_nm supports finite defocus, astigmatism, coma coefficients in nm")
        coefficients[key] = float(value)
    axis = (torch.arange(pupil_samples, dtype=torch.float32) + .5) * 2 / pupil_samples - 1
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    rho2 = xx.square() + yy.square()
    field_x, field_y = map(float, field_xy)
    field_r = math.hypot(field_x, field_y)
    angle = math.atan2(field_y, field_x)
    radial_x = xx * math.cos(angle) + yy * math.sin(angle)
    radial_y = -xx * math.sin(angle) + yy * math.cos(angle)
    opd = coefficients["defocus"] * (1 + field_r ** 2) * (2 * rho2 - 1)
    opd += coefficients["astigmatism"] * field_r ** 2 * (radial_x.square() - radial_y.square())
    opd += coefficients["coma"] * field_r * (3 * rho2 - 2) * radial_x
    pupil = (rho2 <= 1)[None] * torch.exp(2j * math.pi * opd[None] / waves[:, None, None])
    padding = (fft_size - pupil_samples) // 2
    pupil = F.pad(pupil, (padding,) * 4)
    intensity = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil, dim=(-2, -1))), dim=(-2, -1)).abs().square()
    intensity /= intensity.sum((-2, -1), keepdim=True)
    step_um = float(pitch_um) / float(scale)
    centers = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernels = []
    # Loop wavelengths bounds memory even for a 61-band, finely sampled grid.
    for band, wave in enumerate(waves.tolist()):
        fft_step_um = wave / 1000 * f_number * pupil_samples / fft_size
        # At least four samples per output cell; sample no coarser than the FFT.
        quadrature = max(4, math.ceil(step_um / fft_step_um))
        if quadrature > 64:
            raise ValueError("PSF discretization requires >64 quadrature samples; increase scale or reduce FFT oversampling")
        offsets = (torch.arange(quadrature) + .5) / quadrature - .5
        coords = (centers[:, None] + offsets[None]).flatten() * step_um
        qy, qx = torch.meshgrid(coords, coords, indexing="ij")
        shift = lca_native * pitch_um * (wave - 550.) / 150.
        gx = (qx - shift * field_x) / fft_step_um + fft_size // 2
        gy = (qy - shift * field_y) / fft_step_um + fft_size // 2
        grid = 2 * (torch.stack((gx, gy), -1) + .5) / fft_size - 1
        sampled = F.grid_sample(intensity[band:band + 1, None], grid[None],
                                mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
        side = 2 * radius + 1
        kernel = sampled.reshape(side, quadrature, side, quadrature).sum((1, 3)).clamp_min(0)
        energy = kernel.sum()
        if not torch.isfinite(energy) or energy <= 0:
            raise ValueError("PSF support excludes all energy; increase radius or reduce field/chromatic displacement")
        kernels.append(kernel / energy)
    return torch.stack(kernels)


def _convolve(spectrum, kernels):
    """Physical convolution (torch conv2d itself computes correlation)."""
    radius = kernels.shape[-1] // 2
    # Replication is defined for any patch size and preserves flat radiance.
    padded = F.pad(spectrum[None], (radius,) * 4, mode="replicate")
    return F.conv2d(padded, kernels.to(spectrum).flip(-1, -2)[:, None], groups=len(spectrum))[0]


def prepare_spectral_kernels(wavelengths_nm, f_number, pitch_um, scale=2,
                             radius=12, field_center=(0., 0.), field_extent=0.,
                             aberrations_nm=None, lca_native=0., pupil_samples=64, fft_size=256):
    """Prepare [1 or 4,L,k,k] kernels once and reuse them across burst frames.

    Four nodes are ordered top-left, top-right, bottom-left, bottom-right.
    Field positions refer to sensor coordinates: warp each frame before applying
    these kernels so scene motion does not incorrectly move lens aberrations.
    """
    if not math.isfinite(float(field_extent)) or field_extent < 0 or len(field_center) != 2:
        raise ValueError("field_extent must be finite nonnegative; field_center must have two coordinates")
    common = dict(f_number=f_number, pitch_um=pitch_um, scale=scale, radius=radius,
                  aberrations_nm=aberrations_nm, lca_native=lca_native,
                  pupil_samples=pupil_samples, fft_size=fft_size)
    if field_extent == 0:
        return spectral_psfs(wavelengths_nm, field_xy=field_center, **common)[None]
    corners = [(field_center[0] + ix * field_extent, field_center[1] + iy * field_extent)
               for iy in (-1, 1) for ix in (-1, 1)]
    return torch.stack([spectral_psfs(wavelengths_nm, field_xy=field, **common) for field in corners])


def apply_spectral_kernels(spectrum, kernels):
    """Apply precomputed optical kernels with bilinear output-field mixing."""
    if spectrum.ndim != 3 or min(spectrum.shape) < 1 or not torch.isfinite(spectrum).all():
        raise ValueError("spectrum must be finite [number_of_wavelengths,H,W]")
    if kernels.ndim != 4 or kernels.shape[0] not in (1, 4) or kernels.shape[1] != len(spectrum) or kernels.shape[-1] != kernels.shape[-2] or kernels.shape[-1] % 2 != 1:
        raise ValueError("prepared kernels must have shape [1 or 4,L,odd_k,odd_k]")
    if not torch.isfinite(kernels).all() or (kernels < 0).any() or not torch.allclose(kernels.sum((-2, -1)), torch.ones_like(kernels[:, :, 0, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("prepared kernels must be finite, nonnegative and normalized")
    if len(kernels) == 1:
        return _convolve(spectrum, kernels[0])
    height, width = spectrum.shape[-2:]
    tx = torch.linspace(0, 1, width, dtype=spectrum.dtype, device=spectrum.device)[None, None, :]
    ty = torch.linspace(0, 1, height, dtype=spectrum.dtype, device=spectrum.device)[None, :, None]
    result = torch.zeros_like(spectrum)
    weights = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)
    for kernel, weight in zip(kernels, weights):
        result += _convolve(spectrum, kernel) * weight
    return result


def blur_spectral(spectrum, wavelengths_nm, f_number, pitch_um, scale=2,
                  radius=12, field_center=(0., 0.), field_extent=0.,
                  aberrations_nm=None, lca_native=0., pupil_samples=64, fft_size=256,
                  prepared_kernels=None):
    """Blur [L,H,W] with explicit optical PSFs or reusable prepared kernels.

    field_extent is the patch half-width in normalized field units. Bilinear
    mixing is a local space-variant approximation and exactly preserves constants.
    When prepared_kernels is provided the caller owns their parameter identity.
    """
    if spectrum.ndim != 3 or spectrum.shape[0] != len(_waves(wavelengths_nm)):
        raise ValueError("spectrum must match wavelengths_nm")
    if prepared_kernels is None:
        prepared_kernels = prepare_spectral_kernels(
            wavelengths_nm, f_number, pitch_um, scale, radius, field_center,
            field_extent, aberrations_nm, lca_native, pupil_samples, fft_size)
    return apply_spectral_kernels(spectrum, prepared_kernels)


def load_psf_library(path, wavelengths_nm, pitch_um, scale=2):
    """Load an optional optical-only NPZ and fail on ambiguous units/integration.

    Arrays: kernels [nodes,L,k,k], field_xy [nodes,2], wavelengths_nm [L].
    Scalar JSON metadata must specify schema='jsr-spectral-psf-v1',
    spatial_unit='um', sampling_um, contains_pixel_integration=false and
    provenance. Wavelength interpolation/resampling is deliberately not hidden.
    Returns normalized kernels, nodes, and the provenance metadata. The caller
    explicitly chooses field interpolation; this function does not silently
    substitute measured PSFs for a different lens, f-number or pitch.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        kernels = torch.as_tensor(data["kernels"].copy(), dtype=torch.float32)
        fields = torch.as_tensor(data["field_xy"].copy(), dtype=torch.float32)
        waves = torch.as_tensor(data["wavelengths_nm"].copy(), dtype=torch.float32)
    expected = _waves(wavelengths_nm)
    if metadata.get("schema") != "jsr-spectral-psf-v1" or metadata.get("spatial_unit") != "um" or not metadata.get("provenance"):
        raise ValueError("PSF library requires explicit schema, micrometer units, and provenance")
    if metadata.get("contains_pixel_integration") is not False:
        raise ValueError("PSF library must explicitly exclude pixel integration to prevent double integration")
    if not math.isclose(float(metadata.get("sampling_um", float("nan"))), pitch_um / scale, rel_tol=1e-6):
        raise ValueError("PSF sampling_um does not match pitch_um/scale")
    if waves.shape != expected.shape or not torch.allclose(waves, expected, atol=1e-4, rtol=0):
        raise ValueError("PSF library wavelengths must match exactly; explicit resampling is required")
    if kernels.ndim != 4 or kernels.shape[1] != len(expected) or kernels.shape[-1] != kernels.shape[-2] or kernels.shape[-1] % 2 != 1:
        raise ValueError("kernels must have shape [nodes,L,odd_k,odd_k]")
    if fields.shape != (len(kernels), 2) or not torch.isfinite(fields).all() or not torch.isfinite(kernels).all() or (kernels < 0).any():
        raise ValueError("field coordinates and nonnegative PSFs must be finite")
    energy = kernels.sum((-2, -1), keepdim=True)
    if (energy <= 0).any():
        raise ValueError("all PSF kernels must have positive energy")
    return {"kernels": kernels / energy, "field_xy": fields,
            "wavelengths_nm": waves, "metadata": metadata}
