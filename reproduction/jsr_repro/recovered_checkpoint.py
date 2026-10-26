"""Tensor-only, self-describing checkpoints for the recovered v9.8 graph."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import torch

from .recovered_frontend import FRONTEND_VERSION
from .recovered_model import RecoveredJSR

CHECKPOINT_FORMAT = 'jsr-recovered-checkpoint-v1'
_CONFIG_KEYS = {'k', 'controller_width', 'refine_width', 'refine_blocks', 'refinement', 'frontend_version'}


def _json_copy(value, name):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain finite JSON-compatible values') from exc


def recovered_model_config(model: RecoveredJSR) -> dict:
    """Architecture and non-tensor routing needed to recreate a model."""
    if not isinstance(model, RecoveredJSR):
        raise ValueError('expected a RecoveredJSR model')
    config = {'k': model.expected_frames,
              'controller_width': model.controller.enc[0][0].out_channels,
              'refine_width': model.refinenet.inp.out_channels,
              'refine_blocks': len(model.refinenet.body),
              'refinement': model.refinement,
              'frontend_version': model.frontend_version}
    _validate_config(config)
    return config


def _validate_config(config):
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError('invalid recovered checkpoint architecture schema')
    if config['frontend_version'] != FRONTEND_VERSION:
        raise ValueError('checkpoint frontend version differs from recovered graph')
    for key in ('controller_width', 'refine_width', 'refine_blocks'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    k = config['k']
    if k is not None and (type(k) is not int or not 1 <= k <= 14):
        raise ValueError('checkpoint K must be null or an integer from 1 to 14')
    if type(config['refinement']) is not bool:
        raise ValueError('checkpoint refinement must be boolean')


def _validate_payload(payload):
    if (not isinstance(payload, dict)
            or set(payload) != {'format', 'config', 'provenance', 'training', 'state_dict'}
            or payload['format'] != CHECKPOINT_FORMAT):
        raise ValueError('unsupported recovered checkpoint format')
    config = payload['config']
    _validate_config(config)
    if not isinstance(payload['provenance'], dict) or not payload['provenance']:
        raise ValueError('checkpoint provenance must be a nonempty JSON object')
    if payload['training'] is not None and not isinstance(payload['training'], dict):
        raise ValueError('checkpoint training metadata must be a JSON object or null')
    metadata = _json_copy({name: payload[name] for name in ('format', 'config', 'provenance', 'training')}, 'metadata')
    state = payload['state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError('checkpoint state_dict must be a nonempty tensor mapping')
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'invalid FP32 checkpoint tensor: {key}')
    first = state.get('controller.enc.0.0.weight')
    refine = state.get('refinenet.inp.weight')
    if first is None or tuple(first.shape) != (config['controller_width'], 151, 3, 3):
        raise ValueError('Controller tensor shape differs from architecture config')
    if refine is None or tuple(refine.shape) != (config['refine_width'], 6, 3, 3):
        raise ValueError('RefineNet tensor shape differs from architecture config')
    mean, std = state.get('feature_mean'), state.get('feature_std')
    if (mean is None or std is None or tuple(mean.shape) != (1, 151, 1, 1)
            or tuple(std.shape) != (1, 151, 1, 1) or (std <= 0).any()):
        raise ValueError('invalid checkpoint feature affine')
    last = config['refine_blocks'] - 1
    if f'refinenet.body.{last}.c1.weight' not in state or f'refinenet.body.{last + 1}.c1.weight' in state:
        raise ValueError('RefineNet block count differs from architecture config')
    return metadata, state


def save_recovered_checkpoint(model: RecoveredJSR, path: str | Path, *, provenance: dict,
                              training: dict | None = None) -> dict:
    """Save FP32 tensors and JSON metadata, without overwriting an existing file.

    This is a model checkpoint for inference or a new optimization run. Optimizer
    and RNG state are intentionally absent, so it is not an exact training resume.
    Callers importing author NPZ weights should identify that source explicitly;
    independently trained weights must retain their own provenance.
    """
    path = Path(path)
    if path.exists():
        raise ValueError('checkpoint file already exists; select a new path')
    payload = {'format': CHECKPOINT_FORMAT, 'config': recovered_model_config(model),
               'provenance': _json_copy(provenance, 'provenance'),
               'training': _json_copy(training, 'training'),
               'state_dict': {name: value.detach().to(device='cpu', dtype=torch.float32).contiguous().clone()
                              for name, value in model.state_dict().items()}}
    metadata, _ = _validate_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return metadata


def _read(path):
    try:
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, TypeError) as exc:
        raise ValueError('checkpoint is not a supported tensor-only payload') from exc
    return _validate_payload(payload)


def read_recovered_checkpoint_metadata(path: str | Path) -> dict:
    """Read JSON metadata using only the restricted weights-only deserializer."""
    metadata, _ = _read(path)
    return metadata


def load_recovered_checkpoint(path: str | Path, *, device='cpu') -> RecoveredJSR:
    """Recreate the complete recovered graph, affine and K routing strictly."""
    metadata, state = _read(path)
    config = metadata['config']
    model = RecoveredJSR(config['controller_width'], config['refine_width'], config['refine_blocks'],
                         refinement=config['refinement'])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError('checkpoint tensor names or shapes differ from the recovered architecture') from exc
    model.expected_frames = config['k']
    return model.to(device).eval()
