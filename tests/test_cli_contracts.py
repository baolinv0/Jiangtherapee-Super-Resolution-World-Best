import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jsr_repro.data import read_manifest
from jsr_repro.infer import infer
from jsr_repro.prepare import build_manifest
from jsr_repro.train import run_training, load_checkpoint, validate_model
from jsr_repro.utils import load_config


def test_div2k_derived_split_preserves_official_train(tmp_path):
    from PIL import Image
    for i in (1, 2, 801, 802, 851, 852):
        directory = tmp_path / "source" / ("DIV2K_train_HR" if i <= 800 else "DIV2K_valid_HR")
        directory.mkdir(parents=True, exist_ok=True)
        image = np.random.default_rng(i).integers(0, 255, (12, 12, 3), dtype=np.uint8)
        Image.fromarray(image).save(directory / f"{i:04d}.png")
    manifest = build_manifest(tmp_path / "source", tmp_path / "manifest", layout="div2k")
    rows = read_manifest(manifest)
    assert {int(Path(r["path"]).stem) for r in rows if r["split"] == "train"} == {1, 2}
    assert {int(Path(r["path"]).stem) for r in rows if r["split"] == "val"} == {801, 802}
    assert {int(Path(r["path"]).stem) for r in rows if r["split"] == "test"} == {851, 852}


def test_validation_uses_controller_prediction():
    class DistinctOutputs(torch.nn.Module):
        def forward(self, raw, shifts):
            return {"rgb": torch.zeros(raw.shape[0], 3, 16, 16), "learned": torch.ones(raw.shape[0], 3, 16, 16)}
    dataset = [{"raw": torch.zeros(1, 1, 8, 8), "shifts": torch.zeros(1, 2), "target": torch.ones(3, 16, 16)}]
    model = DistinctOutputs()
    assert validate_model(model, dataset, "cpu", output_key="learned")["psnr_linear_db"] == 120
    assert validate_model(model, dataset, "cpu", output_key="rgb")["psnr_linear_db"] == 0


def test_stage_initialization_and_inference_archive(tmp_path):
    manifest = build_manifest(None, tmp_path / "data", procedural=6)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["data"]["manifest"] = str(manifest)
    config["train"]["steps"] = 1
    config["train"]["stage"] = "controller"
    config["output"] = str(tmp_path / "controller")
    run_training(config)
    first = tmp_path / "controller/last.pt"
    original, _ = load_checkpoint(first)
    config["output"] = str(tmp_path / "refine")
    config["train"]["stage"] = "refine"
    run_training(config, initialize=first)
    checkpoint = tmp_path / "refine/last.pt"
    model, _ = load_checkpoint(checkpoint)
    for name, value in model.controller.state_dict().items():
        torch.testing.assert_close(value, original.controller.state_dict()[name], rtol=0, atol=0)
    raw = np.full((4, 1, 16, 16), .1, np.float32)
    archive = tmp_path / "burst.npz"
    np.savez(archive, raw=raw, shifts=np.zeros((4, 2), np.float32))
    infer(checkpoint, archive, tmp_path / "result", "provided")
    assert np.load(tmp_path / "result/linear_rgb.npy").shape == (32, 32, 3)
    with pytest.raises(ValueError, match="memory limit"):
        infer(checkpoint, archive, tmp_path / "too_large", "provided", max_native_pixels=10)
    np.savez(archive, raw=raw)
    with pytest.raises(ValueError, match="provided alignment requires"):
        infer(checkpoint, archive, tmp_path / "missing", "provided")
