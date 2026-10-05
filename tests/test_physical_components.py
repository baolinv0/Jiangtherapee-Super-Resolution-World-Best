"""Physical invariants/statistical checks, independent of training quality."""
import json

import numpy as np
import pytest
import torch

from jsr_repro.physical_optics import (apply_spectral_kernels, blur_spectral,
                                      load_psf_library, prepare_spectral_kernels,
                                      spectral_psfs)
from jsr_repro.physical_sensor import capture_sensor


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def camera(**kwargs):
    return {"gain_e_per_dn": 2., "read_noise_e": 3., "black_dn": 100.,
            "white_dn": 4095., "full_well_e": 12000., **kwargs}


def capture(value=.1, frames=1, size=16, seed=41, **kwargs):
    return capture_sensor(torch.full((frames, 3, size, size), value),
                          torch.ones(frames), torch.ones(3), camera(),
                          torch.Generator().manual_seed(seed), **kwargs)


def second_moment(k):
    axis = torch.arange(-(k.shape[-1] // 2), k.shape[-1] // 2 + 1)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return (k * (xx.square() + yy.square())).sum((-2, -1))


def test_diffraction_normalization_and_physical_scaling():
    psfs = spectral_psfs([450, 550, 650], 4., 4.43, radius=8)
    assert psfs.shape == (3, 17, 17)
    assert torch.all(psfs >= 0)
    torch.testing.assert_close(psfs.sum((-2, -1)), torch.ones(3), atol=1e-6, rtol=0)
    assert torch.all(second_moment(psfs)[1:] > second_moment(psfs)[:-1])
    narrow = spectral_psfs([550], 2., 4.43, radius=8)
    wide = spectral_psfs([550], 8., 4.43, radius=8)
    assert second_moment(wide).item() > 2 * second_moment(narrow).item()
    # Same lambda*f_number produces identical physical diffraction dimensions.
    torch.testing.assert_close(spectral_psfs([400], 4., 4.43, radius=8),
                               spectral_psfs([800], 2., 4.43, radius=8))


def test_aberration_and_chromatic_field_displacement():
    options = dict(f_number=4., pitch_um=4.43, radius=8,
                   aberrations_nm={"defocus": 45., "astigmatism": 80., "coma": 60.})
    center = spectral_psfs([450, 650], field_xy=(0., 0.), **options)
    edge = spectral_psfs([450, 650], field_xy=(.8, .4), **options)
    assert (edge - center).abs().max() > .01
    shifted = spectral_psfs([400, 700], 4., 4.43, radius=8, field_xy=(1., 0.), lca_native=.8)
    axis = torch.arange(-8, 9)
    centroid_x = (shifted * axis[None, None]).sum((-2, -1))
    assert centroid_x[0] < -1 and centroid_x[1] > 1
    # An impulse moves in the direction of the PSF centroid (convolution).
    impulse = torch.zeros(2, 33, 33)
    impulse[:, 16, 16] = 1
    image = apply_spectral_kernels(impulse, shifted[None])
    output_axis = torch.arange(33) - 16
    centroid_out = (image * output_axis[None, None]).sum((-2, -1))
    torch.testing.assert_close(centroid_x, centroid_out, atol=1e-5, rtol=1e-5)


def test_spatially_varying_flat_radiance_and_reuse():
    spectrum = torch.tensor([.2, .4, .8])[:, None, None].expand(3, 19, 23)
    settings = dict(f_number=5.6, pitch_um=3., radius=5, field_center=(.4, .3),
                    field_extent=.15, aberrations_nm={"coma": 70., "astigmatism": 90.})
    kernels = prepare_spectral_kernels([450, 550, 650], **settings)
    assert kernels.shape == (4, 3, 11, 11)
    out = apply_spectral_kernels(spectrum, kernels)
    torch.testing.assert_close(out, spectrum, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out, blur_spectral(spectrum, [450, 550, 650], **settings, prepared_kernels=kernels))


def test_sensor_electron_units_black_level_and_determinism():
    result = capture(.125, noise=False, quantize=False)
    torch.testing.assert_close(result["raw"], torch.full_like(result["raw"], .125))
    torch.testing.assert_close(result["signal_e"], torch.full_like(result["signal_e"], .125 * 3995 * 2))
    torch.testing.assert_close(result["observed_dn"], torch.full_like(result["observed_dn"], 100 + .125 * 3995))
    assert not result["variance"].any() and not result["saturation"].any()
    zero = capture(0., noise=False, quantize=True)
    assert not zero["raw"].any()
    for key, value in capture(.125, noise=False, seed=1).items():
        torch.testing.assert_close(value, capture(.125, noise=False, seed=9)[key])


def test_unsaturated_shot_and_read_statistics():
    result = capture(.05, frames=16, size=96, noise=True, quantize=False)
    samples = result["raw"].double()
    e_per_unit = 3995 * 2
    expected = (.05 * e_per_unit + 3 ** 2) / e_per_unit ** 2
    assert abs(samples.mean().item() - .05) < 6 * (expected / samples.numel()) ** .5
    assert abs(samples.var(unbiased=True).item() / expected - 1) < .025
    assert abs(result["variance"].mean().item() / expected - 1) < .01


def test_same_ptc_channel_transmission_and_partial_saturation():
    result = capture_sensor(torch.full((1, 3, 8, 8), 1.4), torch.ones(1),
                            torch.tensor([.5, 1., .25]), camera(),
                            torch.Generator().manual_seed(4), noise=False, quantize=False)
    assert result["signal_saturation"].float().mean() == .5  # green sites
    assert result["saturation"].float().mean() == .5
    torch.testing.assert_close(result["raw"][0, 0, 0::2, 0::2], torch.full((4, 4), .7))
    torch.testing.assert_close(result["raw"][0, 0, 1::2, 1::2], torch.full((4, 4), .35))


def test_shot_before_fullwell_and_observable_saturation_only():
    profile = camera(gain_e_per_dn=1., read_noise_e=0., full_well_e=100.)
    # At exactly full well the Poisson lower tail survives while the upper clips.
    result = capture_sensor(torch.full((8, 3, 64, 64), 100 / 3995), torch.ones(8),
                            torch.ones(3), profile, torch.Generator().manual_seed(8),
                            noise=True, quantize=False)
    captured = result["raw"] * 3995
    assert captured.max() <= 100.001
    assert 94 < captured.mean() < 98
    assert .35 < result["saturation"].mean() < .65
    assert result["signal_saturation"].min() == 1
    # Observed saturation is fully calculable without signal_saturation labels.
    observable = (result["observed_dn"] >= 200).float()
    torch.testing.assert_close(result["saturation"], observable)


def test_dark_bank_replaces_gaussian_and_retains_correlation():
    # Coherent row noise is preserved by selecting whole dark residual frames.
    pattern = torch.arange(16).float().sub(7.5)[:, None].expand(16, 16)
    bank = torch.stack([pattern, -pattern])
    result = capture(0., frames=5, noise=True, quantize=False, read_noise_bank=bank)
    electrons = result["raw"][:, 0] * (3995 * 2)
    torch.testing.assert_close(electrons, electrons[:, :, :1].expand_as(electrons), atol=1e-5, rtol=1e-5)
    expected = pattern.square() / (3995 * 2) ** 2
    # Plug-in signal term is nonzero on positive read excursions by design.
    read_component = result["variance"][:, 0] - electrons.clamp_min(0) / (3995 * 2) ** 2
    torch.testing.assert_close(read_component, expected[None].expand_as(read_component), atol=1e-10, rtol=1e-5)
    with pytest.raises(ValueError, match="matching spatial shape"):
        capture(noise=True, read_noise_bank=torch.zeros(2, 8, 8))


def test_psf_library_rejects_double_pixel_integration(tmp_path):
    path = tmp_path / "psf.npz"
    metadata = {"schema": "jsr-spectral-psf-v1", "spatial_unit": "um",
                "sampling_um": 2., "contains_pixel_integration": False,
                "provenance": {"kind": "analytic_test"}}
    def save():
        np.savez(path, metadata=np.asarray(json.dumps(metadata)), kernels=np.ones((1, 2, 3, 3)),
                 field_xy=np.zeros((1, 2)), wavelengths_nm=np.asarray([500, 600]))
    save()
    loaded = load_psf_library(path, [500, 600], 4.)
    torch.testing.assert_close(loaded["kernels"].sum((-2, -1)), torch.ones(1, 2))
    metadata["contains_pixel_integration"] = True
    save()
    with pytest.raises(ValueError, match="double integration"):
        load_psf_library(path, [500, 600], 4.)
