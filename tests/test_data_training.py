import copy
import json

import numpy as np
import pytest
import torch

from jsr_repro.alignment import estimate_translations, single_frame_baseline
from jsr_repro.data import SyntheticBurstDataset, mosaic_rggb, synthesize, verify_manifest
from jsr_repro.metrics import image_metrics
from jsr_repro.prepare import build_manifest
from jsr_repro.train import load_checkpoint, run_training
from jsr_repro.utils import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def test_scene_split_repeatability_and_hash_guard(tmp_path):
    first = build_manifest(None, tmp_path / "a", procedural=8)
    second = build_manifest(None, tmp_path / "b", procedural=8)
    rows = verify_manifest(first)
    other = verify_manifest(second)
    assert [(r["sha256"], r["split"]) for r in rows] == [(r["sha256"], r["split"]) for r in other]
    opts = dict(native_size=16, scale=2, frames=4)
    d = SyntheticBurstDataset(first, "train", opts)
    assert torch.equal(d[0]["raw"], d[0]["raw"])
    d.set_epoch(1)
    assert not torch.equal(d[0]["raw"], SyntheticBurstDataset(first, "train", opts)[0]["raw"])
    with open(rows[0]["resolved_path"], "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(first)


def test_reject_cross_split_leak(tmp_path):
    manifest = build_manifest(None, tmp_path, procedural=6)
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    duplicate = dict(rows[0], split="test" if rows[0]["split"] != "test" else "train")
    with manifest.open("a") as f:
        f.write(json.dumps(duplicate) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        verify_manifest(manifest)


def test_noiseless_cfa_exposure_and_sampling():
    rgb = torch.tensor([.4, .2, .1])[:, None, None].expand(3, 96, 96)
    opts = dict(native_size=16, scale=2, frames=4, max_shift=1.5, blur_sigma=[0, 0], exposure_ev=[-1, -1], color_gain=[1, 1], shot_noise=[0, 0], read_noise=[0, 0], quantization_bits=0, augment=False)
    sample = synthesize(rgb, opts, torch.Generator().manual_seed(1))
    torch.testing.assert_close(sample["target"], (rgb * .5)[:, :32, :32])
    torch.testing.assert_close(sample["raw"], mosaic_rggb((rgb * .5)[None, :, :16, :16].expand(4, -1, -1, -1)))
    assert torch.equal(sample["shifts"][0], torch.zeros(2))
    assert sample["raw"].shape == (4, 1, 16, 16)


def test_poisson_gaussian_noise_variance():
    torch.set_num_threads(2)
    rgb = torch.full((3, 300, 300), .2)
    opts = dict(native_size=128, scale=1, frames=4, max_shift=0, blur_sigma=[0, 0], exposure_ev=[0, 0], color_gain=[1, 1], shot_noise=[.01, .01], read_noise=[.02, .02], quantization_bits=0, clip_sensor=False)
    sample = synthesize(rgb, opts, torch.Generator().manual_seed(3))
    assert abs(sample["raw"].mean().item() - .2) < .001
    assert abs(sample["raw"].var().item() - (.01 * .2 + .02 ** 2)) < .0001


def test_phase_correlation_motion_sign():
    gen = torch.Generator().manual_seed(41)
    # Broad features prevent CFA phase differences from dominating correlation.
    field = torch.rand(1, 1, 64, 64, generator=gen)
    field = torch.nn.functional.avg_pool2d(field, 5, stride=1, padding=2)
    moved = torch.roll(field, shifts=(-2, 4), dims=(-2, -1))
    burst = torch.stack([field, moved], dim=1)
    shifts = estimate_translations(burst)
    torch.testing.assert_close(shifts[0, 1], torch.tensor([-4., 2.]), atol=.5, rtol=0)


def test_baseline_constant_color_and_fixed_range_metric():
    raw = mosaic_rggb(torch.tensor([.3, .2, .1])[None, :, None, None].expand(1, 3, 16, 16))[:, None]
    rgb = single_frame_baseline(raw)
    torch.testing.assert_close(rgb, torch.tensor([.3, .2, .1])[None, :, None, None].expand(1, 3, 32, 32))
    m = image_metrics(rgb + .1, rgb)
    assert m["psnr_linear_db"] == pytest.approx(20, abs=1e-5)
    assert m["mean_bias"] == pytest.approx(.1, abs=1e-6)


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    manifest = build_manifest(None, tmp_path / "data", procedural=6)
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    cfg["data"]["manifest"] = str(manifest)
    cfg["data"]["samples_per_scene"] = 1
    cfg["train"]["steps"] = 4
    cfg["train"]["validate_every"] = 2
    cfg["output"] = str(tmp_path / "full")
    run_training(cfg)
    cfg_part = copy.deepcopy(cfg)
    cfg_part["output"] = str(tmp_path / "resumed")
    run_training(cfg_part, stop_after=2)
    run_training(cfg_part, resume=tmp_path / "resumed/last.pt")
    a, sa = load_checkpoint(tmp_path / "full/last.pt")
    b, sb = load_checkpoint(tmp_path / "resumed/last.pt")
    assert sa["step"] == sb["step"] == 4
    assert sa["scheduler"] == sb["scheduler"]
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    assert all(torch.isfinite(v).all() for v in a.state_dict().values())
