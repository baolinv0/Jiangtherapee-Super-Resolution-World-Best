"""Physical contracts for the dense, pixel-integrated camera target."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from jsr_repro.bracket_data import BracketBurstDataset, make_burst_dataset
from jsr_repro.optics import sensor_integrate
from jsr_repro import spectral_data
from jsr_repro.spectral import SpectralAssets
from jsr_repro.utils import sha256


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def assets():
    return SpectralAssets()


def options(**changes):
    return {"protocol": "spectral-camera-v3", "native_size": 16,
            "frames": 7, "scale": 2, "psf_radius": 4,
            "scene_margin_hr": 16, "max_shift": .5,
            "pupil_samples": 32, "fft_size": 128, "field_extent": 0.,
            "field_center": [.3, .2], "scene_gain": [.1, .1],
            "defocus_nm": [15., 15.], "astigmatism_nm": [50., 50.],
            "coma_nm": [40., 40.], "lca_native": [.2, .2],
            "iso": [800], "camera_ids": ["Canon 5DMarkII"],
            "noise": False, "quantize": False, "augment": False,
            **changes}


def textured_source():
    return torch.rand((3, 64, 64), generator=torch.Generator().manual_seed(5))


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("fill", [.8, 1.])
def test_sensor_dense_grid_preserves_flat_radiance(scale, fill):
    scene = torch.full((3, 20, 20), 2.5, dtype=torch.float64)
    result = sensor_integrate(scene, torch.tensor([[.2, -.1]]), 4, 5,
                              fill_factor=fill, output_scale=scale)
    assert result.shape == (1, 3, 4 * scale, 4 * scale)
    torch.testing.assert_close(result, torch.full_like(result, 2.5))


@pytest.mark.parametrize("scale", [1, 2])
def test_sensor_pixel_centers_follow_half_pixel_coordinate_convention(scale):
    yy, xx = torch.meshgrid(torch.arange(20, dtype=torch.float64),
                           torch.arange(20, dtype=torch.float64), indexing="ij")
    scene = torch.stack((xx, yy, xx + 2 * yy))
    result = sensor_integrate(scene, torch.tensor([[.25, -.125]], dtype=torch.float64),
                              3, 5, output_scale=scale)[0]
    # Native centers are 5.5,7.5,9.5; dense centers are 5,6,...,10.
    axis = torch.tensor([5.5, 7.5, 9.5] if scale == 1 else [5., 6., 7., 8., 9., 10.])
    expected_y, expected_x = torch.meshgrid(axis - .25, axis + .5, indexing="ij")
    expected = torch.stack((expected_x, expected_y, expected_x + 2 * expected_y)).double()
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)


def test_dense_sampling_keeps_the_native_pixel_footprint_and_distinct_phase():
    yy, xx = torch.meshgrid(torch.arange(24, dtype=torch.float64),
                           torch.arange(24, dtype=torch.float64), indexing="ij")
    scene = torch.stack((xx.square(), yy.square(), xx.square() + yy.square()))
    native = sensor_integrate(scene, torch.zeros(1, 2), 4, 5, fill_factor=1.)
    dense = sensor_integrate(scene, torch.zeros(1, 2), 4, 5, fill_factor=1., output_scale=2)
    # For integer center 5, 4-point bilinear area quadrature gives E[x²]=25.5.
    torch.testing.assert_close(dense[0, :, 0, 0], torch.tensor([25.5, 25.5, 51.], dtype=torch.float64))
    assert not torch.allclose(dense[..., ::2, ::2], native)
    # A quarter-native-pixel shift aligns dense even samples to native centers.
    aligned_dense = sensor_integrate(scene, torch.full((1, 2), .25), 4, 5,
                                     fill_factor=1., output_scale=2)
    torch.testing.assert_close(aligned_dense[..., ::2, ::2], native, atol=1e-12, rtol=0)
    explicit_native = sensor_integrate(scene, torch.zeros(1, 2), 4, 5,
                                       fill_factor=1., output_scale=1)
    assert torch.equal(explicit_native, native)


def test_full_spectral_field_target_matches_independent_native_area_oracle(assets, monkeypatch):
    """Catch omitted/doubled/shrunken aperture on the actual frame-zero field.

    At fill=1, four midpoint samples per axis at ±.25 and ±.75 HR
    units with bilinear interpolation equal the discrete [.25,.5,.25]
    kernel at dense integer centers. This oracle does not call the production
    integration helper or replace the real spectral/field-dependent optics.
    """
    source = torch.rand((61, 72, 72), generator=torch.Generator().manual_seed(811))
    projected_fields = []
    project = assets.camera_rgb

    def record_projection(spectrum, camera_id):
        rgb = project(spectrum, camera_id)
        projected_fields.append(rgb.clone())
        return rgb

    monkeypatch.setattr(assets, 'camera_rgb', record_projection)
    sample = spectral_data.synthesize_spectral(source, options(
        field_extent=.2, fill_factor=[1., 1.], target_stage='post_pixel'), 341, assets=assets)
    reference = projected_fields[0]
    one_axis = torch.tensor([.25, .5, .25], dtype=reference.dtype)
    kernel = (one_axis[:, None] * one_axis[None, :]).expand(3, 1, 3, 3)
    integrated = torch.nn.functional.conv2d(reference[None], kernel, padding=1, groups=3)[0]
    expected = integrated[:, 16:48, 16:48]
    torch.testing.assert_close(sample['target'], expected, atol=3e-7, rtol=1e-5)
    assert (sample['target'] - reference[:, 16:48, 16:48]).abs().max() > 1e-5
    twice = torch.nn.functional.conv2d(integrated[None], kernel, padding=1, groups=3)[0, :, 16:48, 16:48]
    assert (sample['target'] - twice).abs().max() > 1e-5
    narrow_axis = torch.tensor([.125, .75, .125], dtype=reference.dtype)
    narrow_kernel = (narrow_axis[:, None] * narrow_axis[None, :]).expand(3, 1, 3, 3)
    shrunk = torch.nn.functional.conv2d(reference[None], narrow_kernel, padding=1, groups=3)[0, :, 16:48, 16:48]
    assert (sample['target'] - shrunk).abs().max() > 1e-5


@pytest.mark.parametrize("scale", [0, 3, 1.5])
def test_sensor_rejects_unsupported_output_density(scale):
    with pytest.raises(ValueError, match="output_scale"):
        sensor_integrate(torch.ones(3, 12, 12), torch.zeros(1, 2), 2, 3, output_scale=scale)


def test_three_target_stages_share_identical_input_bursts_and_legacy_rng(assets):
    source = textured_source()
    samples = [spectral_data.synthesize_spectral(source, options(target_stage=stage,
               noise=True, quantize=True), 341, assets=assets)
               for stage in ("pre_optics", "post_optics", "post_pixel")]
    legacy = spectral_data.synthesize_spectral(source, options(protocol="spectral-camera-v2",
                noise=True, quantize=True), 341, assets=assets)
    for sample in samples:
        for key in ("raw", "shifts", "exposure", "variance", "saturation", "valid",
                    "black_invalid", "transmission", "signal_saturation"):
            assert torch.equal(sample[key], legacy[key]), key
    assert torch.equal(samples[0]["target"], legacy["target"])
    assert not torch.equal(samples[0]["target"], samples[1]["target"])
    assert not torch.equal(samples[1]["target"], samples[2]["target"])
    assert legacy["metadata"]["crop_xy"] == [3, 29]
    expected_shift = torch.tensor([-.16500979661941528, -.17775940895080566])
    assert torch.equal(legacy["shifts"][1], expected_shift)


@pytest.mark.parametrize("stage", ["pre_optics", "post_optics", "post_pixel"])
def test_target_stage_retains_only_its_declared_physical_operators(stage, assets):
    source = textured_source()
    base = spectral_data.synthesize_spectral(source, options(target_stage=stage,
               f_number=[2., 2.], fill_factor=[1., 1.]), 341, assets=assets)
    psf = spectral_data.synthesize_spectral(source, options(target_stage=stage,
               f_number=[8., 8.], fill_factor=[1., 1.]), 341, assets=assets)
    fill = spectral_data.synthesize_spectral(source, options(target_stage=stage,
               f_number=[2., 2.], fill_factor=[.8, .8]), 341, assets=assets)
    assert torch.equal(base["target"], psf["target"]) == (stage == "pre_optics")
    assert torch.equal(base["target"], fill["target"]) == (stage != "post_pixel")
    assert not torch.equal(base["raw"], psf["raw"])
    assert not torch.equal(base["raw"], fill["raw"])


@pytest.mark.parametrize("stage", ["pre_optics", "post_optics", "post_pixel"])
def test_target_is_unclipped_and_independent_of_sensor_noise_and_exposure(stage, assets):
    source = torch.ones(3, 48, 48)
    base = spectral_data.synthesize_spectral(source, options(target_stage=stage,
              scene_gain=[1.4, 1.4], bracket_interval_ev=[0., 0.]), 341, assets=assets)
    noisy = spectral_data.synthesize_spectral(source, options(target_stage=stage,
               scene_gain=[1.4, 1.4], bracket_interval_ev=[1., 1.],
               noise=True, quantize=True), 341, assets=assets)
    torch.testing.assert_close(base["target"], torch.full((3, 32, 32), 1.4), atol=3e-6, rtol=0)
    assert torch.equal(base["target"], noisy["target"])
    assert not torch.equal(base["raw"], noisy["raw"])


def test_default_target_and_metadata_describe_dense_pixel_integrated_camera_rgb(assets):
    opts = options()
    del opts["protocol"]
    sample = spectral_data.synthesize_spectral(textured_source(), opts, 341, assets=assets)
    explicit = spectral_data.synthesize_spectral(textured_source(),
                  options(target_stage="post_pixel"), 341, assets=assets)
    assert torch.equal(sample["target"], explicit["target"])
    meta = sample["metadata"]
    assert meta["protocol"] == "spectral-camera-v3"
    assert meta["target_stage"] == "post_pixel"
    assert meta["pixel_aperture"] == "retained"
    assert meta["area_quadrature"] == 4
    assert meta["output_sample_pitch_native"] == .5
    assert "reference exposure" in meta["target_space"]
    assert "camera RGB" in meta["target_space"]
    assert "0.5" in meta["sampling_phase"]


@pytest.mark.parametrize("opts,expected", [
    ({}, "post_pixel"),
    ({"protocol": "spectral-camera-v2"}, "pre_optics"),
    ({"protocol": "spectral-camera-v2", "target_stage": "pre_optics"}, "pre_optics"),
    ({"protocol": "spectral-camera-v3", "target_stage": "post_optics"}, "post_optics"),
])
def test_target_stage_resolver_normalizes_protocol_defaults(opts, expected):
    assert spectral_data.resolve_target_stage(opts) == expected


@pytest.mark.parametrize("opts", [
    {"target_stage": "unknown"}, {"target_stage": None},
    {"protocol": "spectral-camera-v2", "target_stage": "post_pixel"},
    {"protocol": "spectral-camera-v2", "target_stage": "post_optics"},
    {"protocol": "invalid"},
])
def test_unknown_or_conflicting_target_contracts_are_rejected(opts, assets):
    with pytest.raises(ValueError, match="protocol|target_stage"):
        spectral_data.synthesize_spectral(torch.ones(3, 48, 48), options(**opts), 1, assets=assets)


def test_dataset_identity_pins_v3_target_and_keeps_legacy_identity(tmp_path):
    source = tmp_path / "source.npy"
    np.save(source, np.ones((48, 48, 3), np.float32))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"scene_id": "s", "split": "train", "path": str(source),
                        "sha256": sha256(source), "encoding": "linear"}) + "\n")
    legacy = make_burst_dataset(manifest, "train", options(protocol="spectral-camera-v2"))
    assert legacy.identity == {"spectral_assets": legacy.assets.identity, "protocol": "spectral-camera-v2"}
    identities = []
    for stage in ("pre_optics", "post_optics", "post_pixel"):
        dataset = make_burst_dataset(manifest, "train", options(target_stage=stage))
        assert isinstance(dataset, spectral_data.SpectralBurstDataset)
        assert dataset.identity["target_stage"] == stage
        identities.append(dataset.identity)
    assert identities[0] != identities[1] != identities[2]
    assert isinstance(make_burst_dataset(manifest, "train", {}), BracketBurstDataset)
