"""Semantic data-contract tests for the independently specified spectral chain."""
from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pytest
import torch

from jsr_repro.bracket_data import make_burst_dataset
from jsr_repro.calibration import analytic_profile, profile_identity
from jsr_repro.evaluate_transformer import evaluate
from jsr_repro.infer_transformer import infer, read_burst
from jsr_repro.spectral import ASSET_ROOT, SpectralAssets, load_spectral_scene
from jsr_repro.spectral_data import PROTOCOL, SpectralBurstDataset, main, synthesize_spectral
from jsr_repro.train_transformer import load_checkpoint, run_training
from jsr_repro.transformer import FIELDS
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
    return {"protocol": PROTOCOL, "native_size": 16, "frames": 7, "scale": 2,
            "psf_radius": 4, "scene_margin_hr": 16, "max_shift": .5,
            "pupil_samples": 32, "fft_size": 128, "field_extent": 0.,
            "field_center": [.3, .2], "scene_gain": [.1, .1],
            "defocus_nm": [15., 15.], "astigmatism_nm": [50., 50.],
            "coma_nm": [40., 40.], "lca_native": [.2, .2],
            "iso": [800], "camera_ids": ["Canon 5DMarkII"],
            "noise": False, "quantize": False, "augment": False,
            **changes}


def test_clear_gt_is_invariant_to_optical_interventions(assets):
    source = torch.rand((3, 64, 64), generator=torch.Generator().manual_seed(5))
    base = synthesize_spectral(source, options(), 341, assets=assets)
    interventions = {"f_number": [8., 8.], "pitch_um": [3., 3.],
                     "fill_factor": [.8, .8], "psf_radius": 6,
                     "field_center": [.8, .6]}
    for name, setting in interventions.items():
        changed = synthesize_spectral(source, options(**{name: setting}), 341, assets=assets)
        assert torch.equal(base["target"], changed["target"]), name
        assert (base["raw"] - changed["raw"]).abs().max() > 1e-6, name
    margin = synthesize_spectral(source, options(scene_margin_hr=18), 341, assets=assets)
    torch.testing.assert_close(base["target"], margin["target"], atol=1e-6, rtol=0)


def test_flat_scene_exposure_transmission_and_cfa_units(assets):
    profile = analytic_profile()
    source = torch.ones(3, 48, 48)
    sample = synthesize_spectral(source, options(scene_gain=[.05, .05],
                                 bracket_interval_ev=[.3, .3]), 42,
                                 profile=profile, assets=assets)
    torch.testing.assert_close(sample["target"], torch.full((3, 32, 32), .05), atol=1e-6, rtol=1e-5)
    for color, y, x in ((0, 0, 0), (1, 0, 1), (1, 1, 0), (2, 1, 1)):
        corrected = sample["raw"][:, 0, y::2, x::2] / sample["exposure"][:, None, None] / profile["channel_transmission"][color]
        torch.testing.assert_close(corrected, torch.full_like(corrected, .05), atol=2e-6, rtol=1e-5)
    assert not sample["signal_saturation"].any()
    assert not sample["saturation"].any()


def test_true_partial_channel_overexposure_has_unclipped_gt(assets):
    profile = analytic_profile()
    profile["channel_transmission"] = [.5, 1., .25]
    sample = synthesize_spectral(torch.ones(3, 48, 48),
                                 options(scene_gain=[1.4, 1.4], bracket_interval_ev=[0., 0.]),
                                 43, profile=profile, assets=assets)
    torch.testing.assert_close(sample["target"], torch.full((3, 32, 32), 1.4), atol=2e-6, rtol=1e-5)
    assert sample["signal_saturation"].mean() == .5
    assert sample["saturation"].mean() == .5
    torch.testing.assert_close(sample["raw"][:, 0, 0::2, 0::2], torch.full((7, 8, 8), .7), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(sample["raw"][:, 0, 1::2, 1::2], torch.full((7, 8, 8), .35), atol=2e-6, rtol=1e-5)
    assert sample["raw"][:, 0, 0::2, 1::2].min() == 1
    assert not {"signal_saturation", "signal_e", "clean_raw", "target"}.intersection(FIELDS)


def test_spectral_import_matches_lifted_rgb_without_claiming_measurement(tmp_path, assets):
    source = torch.rand((3, 48, 48), generator=torch.Generator().manual_seed(71))
    path = tmp_path / "provided_spectrum.npz"
    np.savez(path, wavelengths_nm=assets.wavelengths.numpy(), radiance=assets.lift(source).numpy())
    imported = load_spectral_scene(path, assets.wavelengths)
    rgb = synthesize_spectral(source, options(), 99, assets=assets)
    spectral = synthesize_spectral(imported, options(), 99, assets=assets)
    torch.testing.assert_close(rgb["target"], spectral["target"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rgb["raw"], spectral["raw"], atol=1e-6, rtol=1e-5)
    assert not rgb["metadata"]["spectral_input"]
    assert spectral["metadata"]["spectral_input"]
    assert spectral["metadata"]["spectral_prior"] != "measured-input"
    # A narrow-band file must not silently extrapolate unknown visible bands.
    np.savez(path, wavelengths_nm=np.asarray([450., 650.]), radiance=np.ones((2, 4, 4), np.float32))
    with pytest.raises(ValueError, match="extrapolation"):
        load_spectral_scene(path, assets.wavelengths)


def write_manifest(root):
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        path = root / f"scene_{i}.npy"
        image = np.random.default_rng(700 + i).uniform(.1, .9, (48, 48, 3)).astype(np.float32)
        np.save(path, image)
        rows.append({"scene_id": f"scene_{i}", "split": split, "path": str(path),
                     "sha256": sha256(path), "encoding": "linear"})
    manifest = root / "sources.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_training_resume_generation_inference_and_asset_identity_guards(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path)
    # Copy assets so the mutation check cannot affect another worker or test.
    copied_assets = tmp_path / "spectral_assets"
    copied_assets.mkdir()
    for name in ("basis.json", "cameras.json"):
        shutil.copyfile(ASSET_ROOT / name, copied_assets / name)
    opts = options(asset_root=str(copied_assets), noise=True, quantize=True,
                   bracket_interval_ev=[.2, .2])
    dataset = make_burst_dataset(manifest, "train", opts, 21)
    assert isinstance(dataset, SpectralBurstDataset)
    assert dataset.identity["spectral_assets"]["basis.json"] == sha256(copied_assets / "basis.json")
    cfg = {"seed": 21, "threads": 1, "device": "cpu", "output": str(tmp_path / "train"),
           "model": {"width": 4, "heads": 1, "window": 2, "scale": 2, "frames": 7},
           "data": {"manifest": str(manifest), "options": opts},
           "train": {"steps": 2, "batch_size": 1, "validate_every": 1,
                     "crop_border": 4, "lr": .0001, "clip_grad": 1.}}
    first = run_training(cfg, stop_after=1)
    checkpoint = tmp_path / "train" / "last.pt"
    assert first["steps_completed"] == 1
    second = run_training(cfg, resume=checkpoint, stop_after=2)
    assert second["steps_completed"] == 2
    _, state = load_checkpoint(checkpoint)
    assert state["data_identity"]["construction"] == dataset.identity
    records = [json.loads(line) for line in (tmp_path / "train" / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert all(np.isfinite(row["loss"]) and row["grad_norm"] > 0 for row in records)
    evaluation = evaluate(checkpoint, manifest, tmp_path / "evaluation.json", limit=1)
    assert evaluation["data_protocol"] == PROTOCOL and evaluation["count"] == 1

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    generated = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["spectral_data", "--config", str(cfg_path),
                                      "--output", str(generated), "--split", "test", "--count", "1"])
    main()
    burst = generated / "test_000000.npz"
    sample, metadata = read_burst(burst, "cpu")
    assert sample["raw"].shape == (1, 7, 1, 16, 16)
    assert metadata["protocol"] == PROTOCOL
    report = infer(checkpoint, burst, tmp_path / "inference", alignment="provided")
    assert report["calibration_domain_match"]
    assert report["output_shape_hwc"] == [32, 32, 3]
    assert (tmp_path / "inference" / "linear_rgb16.tiff").is_file()

    # Even a provenance-only or whitespace change invalidates the pinned asset
    # identity. Config paths remain unchanged, so this tests content guarding.
    with (copied_assets / "basis.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    changed = make_burst_dataset(manifest, "train", opts, 21)
    assert changed.identity != dataset.identity
    with pytest.raises(ValueError, match="identity changed"):
        run_training(cfg, resume=checkpoint)
    with pytest.raises(ValueError, match="assets differ"):
        evaluate(checkpoint, manifest, tmp_path / "invalid_evaluation.json", limit=1)


def test_generation_checks_source_hash_and_bank_units(tmp_path, monkeypatch, assets):
    manifest = write_manifest(tmp_path)
    cfg = {'data':{'manifest':str(manifest),'options':options()}}
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    np.save(tmp_path/'scene_0.npy',np.zeros((48,48,3),np.float32))
    monkeypatch.setattr(sys,'argv',['spectral_data','--config',str(config),'--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='hash'):
        main()
    profile = analytic_profile()
    bank = tmp_path/'wrong-units.npz'
    np.savez(bank,read_noise_e=np.zeros((3,16,16),np.float32),metadata=np.asarray(json.dumps({
        'schema':'jsr-read-bank-v1','iso':800,'profile_sha256':profile_identity(profile),'unit':'DN'})))
    with pytest.raises(ValueError,match='electron units'):
        synthesize_spectral(torch.ones(3,48,48),options(read_noise_bank=str(bank)),7,profile,assets)


def test_validation_uses_custom_calibration_and_assets(tmp_path):
    from jsr_repro.validate_spectral import validate
    profile = analytic_profile()
    profile['name'] = 'custom-validation-camera'
    profile['channel_transmission'] = [.6,1.,.4]
    profile_path = tmp_path/'profile.json'
    profile_path.write_text(json.dumps(profile))
    copied = tmp_path/'assets'
    shutil.copytree(ASSET_ROOT,copied)
    with (copied/'basis.json').open('a') as stream:
        stream.write('\n')
    cfg = {'seed':21,'threads':1,'device':'cpu','model':{'width':4,'heads':1,'window':2,'scale':2,'frames':7},
           'data':{'profile':str(profile_path),'options':options(asset_root=str(copied))},
           'train':{'steps':1,'batch_size':1,'validate_every':1,'crop_border':4,'lr':.0001,'clip_grad':1.}}
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    report = validate(tmp_path/'validation',config)
    assert report['asset_identity']['basis.json'] == sha256(copied/'basis.json')
    assert report['sample_metadata']['profile_sha256'] == profile_identity(profile)
    assert report['inference']['calibration_domain_match']
    assert {x['iso'] for x in report['explicit_endpoint_coverage']} == {800}
