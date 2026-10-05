"""Strict import of individual public NPZ modules, not a full JSR pipeline."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn

from .model import Controller, RefineNet


def verify_manifest_file(path: str | Path, manifest_path: str | Path) -> str:
    path = Path(path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if path.name not in manifest["files"]:
        raise ValueError(f"file not in manifest: {path.name}")
    record = manifest["files"][path.name]
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != record["bytes"] or digest != record["sha256"]:
        raise ValueError(f"weight checksum/size mismatch: {path.name}")
    return digest


def _read(path: str | Path, prefix: str = "") -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = archive.files
        if prefix and any(k.startswith(prefix) for k in keys):
            if not all(k.startswith(prefix) for k in keys):
                raise ValueError("mixed prefixed and unprefixed weight keys")
            state = {k[len(prefix):]: np.array(archive[k], copy=True) for k in keys}
        else:
            state = {k: np.array(archive[k], copy=True) for k in keys}
    for name, value in state.items():
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"invalid floating weight: {name}")
    return state


def _load_strict(module: nn.Module, state: dict[str, np.ndarray]) -> nn.Module:
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"weight keys differ: missing={sorted(set(expected) - set(state))}, extra={sorted(set(state) - set(expected))}")
    for name, tensor in expected.items():
        if tuple(state[name].shape) != tuple(tensor.shape):
            raise ValueError(f"shape mismatch for {name}: {state[name].shape} != {tuple(tensor.shape)}")
    module.load_state_dict({name: torch.from_numpy(value).to(dtype=torch.float32) for name, value in state.items()}, strict=True)
    return module.eval()


def load_controller_npz(path: str | Path) -> Controller:
    """Import a released Controller only. Original Tap semantics still missing."""
    state = _read(path, "net.")
    if "enc.0.0.weight" not in state or state["enc.0.0.weight"].shape != (32, 151, 3, 3):
        raise ValueError("released Controller must have width32 / input151")
    return _load_strict(Controller(32), state)


def load_refinenet_npz(path: str | Path) -> RefineNet:
    """Import only RefineNet, inferring width/block count then checking all keys."""
    state = _read(path)
    if "inp.weight" not in state or state["inp.weight"].ndim != 4:
        raise ValueError("missing or invalid RefineNet inp.weight")
    width = state["inp.weight"].shape[0]
    blocks = 0
    while f"body.{blocks}.c1.weight" in state:
        blocks += 1
    if blocks < 1:
        raise ValueError("RefineNet requires contiguous residual blocks")
    return _load_strict(RefineNet(width, blocks), state)


def load_static_affine_npz(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read official affine as a standalone artifact; never auto-apply to v1."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "std", "calibration_ids"}:
            raise ValueError("unexpected static affine keys")
        mean, std, ids = archive["mean"], archive["std"], archive["calibration_ids"]
        if mean.shape != (1, 151, 1, 1) or std.shape != mean.shape or tuple(ids.tolist()) != tuple(range(16)):
            raise ValueError("invalid static affine schema")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid static affine values")
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
