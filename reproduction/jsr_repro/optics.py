"""Three-band circular-aperture diffraction, not measured prime-lens PSFs."""
import math

import torch
import torch.nn.functional as F


def diffraction_psf(f_number, pitch_um, wavelength_nm, scale=2, radius=12):
    if not 2 <= f_number <= 8 or not 3 <= pitch_um <= 5.76 or not 380 <= wavelength_nm <= 780 or scale != 2 or radius < 2:
        raise ValueError("optics envelope: f/2..8, pitch3..5.76um, wavelength380..780nm, scale2, radius>=2")
    axis = torch.arange(-radius, radius + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    # Image-plane radius in micrometers; Airy intensity [2 J1(u)/u]^2.
    u = math.pi * torch.sqrt(xx.square() + yy.square()) * pitch_um / scale / (wavelength_nm / 1000 * f_number)
    safe = torch.where(u > 0, u, torch.ones_like(u))
    psf = torch.where(u > 0, (2 * torch.special.bessel_j1(u) / safe).square(), torch.ones_like(u))
    psf = (psf / psf.sum()).float()
    return psf


def optical_blur(scene, f_number, pitch_um, radius=12):
    wavelengths = (650., 550., 450.)  # representative bands, NOT RGB->spectrum
    kernels = torch.stack([diffraction_psf(f_number, pitch_um, wave, radius=radius) for wave in wavelengths])[:, None].to(scene)
    return F.conv2d(F.pad(scene[None], (radius,) * 4, mode="reflect"), kernels, groups=3)[0]


def sensor_integrate(scene, shifts, native_size, margin, fill_factor=.95, quadrature=4):
    """4x4 quadrature of square active area (sqrt(fill_factor)*pitch).

    Native center (x+.5)*2-.5; frame(x,y) sees ref(x+dx,y+dy).
    Flat radiance remains flat: fill factor controls spatial footprint, while
    effective sensitivity is already represented by the camera profile.
    """
    if not .8 <= fill_factor <= 1 or quadrature < 2:
        raise ValueError("fill factor must be0.8..1; quadrature>=2")
    n = int(native_size)
    yy, xx = torch.meshgrid(torch.arange(n, dtype=scene.dtype), torch.arange(n, dtype=scene.dtype), indexing="ij")
    offsets = ((torch.arange(quadrature, dtype=scene.dtype) + .5) / quadrature - .5) * math.sqrt(fill_factor) * 2
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    centers = torch.stack(((xx + .5) * 2 - .5 + margin, (yy + .5) * 2 - .5 + margin), -1)
    grid = centers[None, :, :, None, :] + shifts[:, None, None, None, :] * 2 + torch.stack((ox.flatten(), oy.flatten()), -1)[None, None, None]
    extent_y, extent_x = scene.shape[-2:]
    grid = 2 * (grid + .5) / torch.tensor([extent_x, extent_y], dtype=scene.dtype) - 1
    samples = F.grid_sample(scene[None].expand(len(shifts), -1, -1, -1), grid.reshape(len(shifts), n, n * quadrature ** 2, 2), mode="bilinear", align_corners=False)
    return samples.reshape(len(shifts), 3, n, n, quadrature ** 2).mean(-1)
