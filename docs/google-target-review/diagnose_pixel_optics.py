"""Mathematical probes for commit 0bc29f1; no production-code modification.

Run after installing the repository reproduction/test dependencies. Outputs are numerical diagnostics,
not claims that the refined FFT setting is a physical ground truth.
"""
import json
from pathlib import Path

import torch

from jsr_repro.optics import sensor_integrate
from jsr_repro.physical_optics import apply_spectral_kernels, spectral_psfs
from jsr_repro.spectral_data import _warp


def motion_probe():
    size, margin = 32, 24
    extent = 2 * size + 2 * margin
    axis = torch.arange(extent).float()
    kernel = spectral_psfs([550] * 3, 2., 5.76, radius=8,
                          pupil_samples=64, fft_size=256)[None]
    rows = []
    for frequency in [.0625, .125, .20, .25, .375, .45]:
        field = (.5 + .3 * torch.cos(2 * torch.pi * frequency * axis + .23))
        field = field[None, None].expand(3, extent, extent).clone()
        optical = apply_spectral_kernels(field, kernel)
        for displacement in [0., .125, .25, .375, .5]:
            shift = torch.tensor([displacement, 0.])
            # For stationary PSFs, continuous translation commutes with optics.
            # Integrate the same bilinear optical field at shifted centers once.
            # At displacement=.25 and fill=1, Q=4 is exact for this piecewise
            # bilinear field (pixel boundaries and quadrature align with knots).
            direct = sensor_integrate(optical, shift[None], size, margin,
                                      fill_factor=1.)[0, 0, 16]
            # Current spectral_data.py lines 153-156: resample scene first,
            # convolve, then interpolate again for native-area integration.
            moved = _warp(field, shift, 2)
            current = sensor_integrate(apply_spectral_kernels(moved, kernel),
                                       torch.zeros(1, 2), size, margin,
                                       fill_factor=1.)[0, 0, 16]
            a, b = direct - direct.mean(), current - current.mean()
            rows.append({"frequency_hr": frequency, "shift_native": displacement,
                         "amplitude_ratio": float((b @ a) / (a @ a)),
                         "max_difference": float((current - direct).abs().max()),
                         "rms_difference": float((current - direct).square().mean().sqrt())})
    return rows


def psf_probe():
    rows = []
    for f_number in [2., 8.]:
        for pitch in [3., 5.76]:
            for field in [(0., 0.), (.8, .4)]:
                opts = dict(f_number=f_number, pitch_um=pitch, radius=16,
                            field_xy=field, aberrations_nm=dict(defocus=20.,
                            astigmatism=60., coma=80.), lca_native=.5)
                kernels = {}
                settings = [('smoke', 32, 128), ('main', 64, 256),
                            ('pupil_refined', 128, 512), ('image_refined', 64, 1024),
                            ('both_refined', 128, 2048)]
                for name, pupil, fft in settings:
                    kernels[name] = spectral_psfs([400., 550., 700.],
                                                  pupil_samples=pupil,
                                                  fft_size=fft, **opts)
                ref = kernels['both_refined']
                for name, kernel in kernels.items():
                    if name == 'both_refined':
                        continue
                    rows.append({"f_number": f_number, "pitch_um": pitch,
                                 "field": list(field), "config": name,
                                 "l1": (kernel-ref).abs().sum((-2,-1)).tolist(),
                                 "max_abs": (kernel-ref).abs().flatten(1).max(-1).values.tolist(),
                                 "center": kernel[:,16,16].tolist(),
                                 "ref_center": ref[:,16,16].tolist()})
    return rows


def support_probe():
    opts = dict(f_number=2., pitch_um=5.76, field_xy=(1., 1.),
                lca_native=2., pupil_samples=64, fft_size=256)
    waves = [400., 550., 700.]
    small = spectral_psfs(waves, radius=2, **opts)
    large = spectral_psfs(waves, radius=16, **opts)
    a = torch.arange(-2, 3).float()
    b = torch.arange(-16, 17).float()
    rows = []
    for index, wave in enumerate(waves):
        crop = large[index, 14:19, 14:19]
        rows.append({"wavelength_nm": wave,
                     "energy_in_radius2_rel_radius16": float(crop.sum()),
                     "small_centroid_x": float((small[index]*a[None,:]).sum()),
                     "small_centroid_y": float((small[index]*a[:,None]).sum()),
                     "large_centroid_x": float((large[index]*b[None,:]).sum()),
                     "large_centroid_y": float((large[index]*b[:,None]).sum()),
                     "normalized_crop_parity_max": float((small[index]-crop/crop.sum()).abs().max())})
    return rows


if __name__ == '__main__':
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    for name, probe in [('motion_interpolation', motion_probe), ('psf_convergence', psf_probe),
                        ('psf_support_exclusion', support_probe)]:
        rows = probe()
        (root / (name + '.json')).write_text(json.dumps(rows, indent=2))
        print(name, len(rows), 'rows')
