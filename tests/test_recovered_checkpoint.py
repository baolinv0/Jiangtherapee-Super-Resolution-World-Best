"""Portable recovered-model checkpoints and explicit-boundary fine tuning."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jsr_repro.recovered_model import RecoveredJSR


def _model():
    torch.set_num_threads(2)
    torch.manual_seed(149)
    model = RecoveredJSR(controller_width=2, refine_width=4, refine_blocks=1)
    model.expected_frames = 2
    with torch.no_grad():
        model.feature_mean.fill_(.01)
        model.feature_std.fill_(1.1)
        model.refinenet.out.weight.mul_(.01)
        model.refinenet.out.bias.zero_()
    return model


def _sample(path, model):
    rng = np.random.default_rng(16)
    frames = rng.uniform(.05, .2, (2, 6, 6)).astype(np.float32)
    flow = np.zeros((2, 6, 6, 2), np.float32)
    covariance = np.broadcast_to(np.array([1., .1, 1.1], np.float32)[:, None, None], (3, 8, 8)).copy()
    with torch.no_grad():
        rgb = model(*(torch.from_numpy(x)[None] for x in (frames, flow, covariance)))['rgb'][0]
    np.savez_compressed(path, frames=frames, flow=flow, covariance=covariance,
                        target=(rgb * .98 + .003).numpy().transpose(1, 2, 0))


def test_checkpoint_roundtrip_reconstructs_architecture_routing_affine_and_output(tmp_path):
    from jsr_repro.recovered_checkpoint import load_recovered_checkpoint, read_recovered_checkpoint_metadata, save_recovered_checkpoint
    model = _model()
    model.refinement = False
    path = tmp_path / 'model.pt'
    metadata = save_recovered_checkpoint(model, path, provenance={'kind': 'test', 'source': ['numeric fixture']})
    saved = torch.load(path, weights_only=True)
    assert set(saved) == {'format', 'config', 'provenance', 'training', 'state_dict'}
    restored = load_recovered_checkpoint(path)
    assert restored.expected_frames == 2 and restored.refinement is False
    assert restored.controller.enc[0][0].out_channels == 2 and len(restored.refinenet.body) == 1
    assert metadata == read_recovered_checkpoint_metadata(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    with pytest.raises(ValueError):
        save_recovered_checkpoint(model, path, provenance={'kind': 'test'})


def test_checkpoint_loader_rejects_schema_shape_and_affine_corruption(tmp_path):
    from jsr_repro.recovered_checkpoint import load_recovered_checkpoint, save_recovered_checkpoint
    path = tmp_path / 'good.pt'
    save_recovered_checkpoint(_model(), path, provenance={'kind': 'test'})
    original = torch.load(path, weights_only=True)
    for name, mutate in (
        ('frontend', lambda x: x['config'].__setitem__('frontend_version', 'inferred-jsr-v1')),
        ('shape', lambda x: x['config'].__setitem__('controller_width', 3)),
        ('affine', lambda x: x['state_dict'].__setitem__('feature_std', torch.zeros(1, 151, 1, 1))),
        ('keys', lambda x: x['state_dict'].pop('controller.out.bias')),
        ('nonfinite', lambda x: x['state_dict'].__setitem__('feature_mean', torch.full((1, 151, 1, 1), float('nan')))),
    ):
        payload = torch.load(path, weights_only=True)
        mutate(payload)
        corrupt = tmp_path / f'{name}.pt'
        torch.save(payload, corrupt)
        with pytest.raises(ValueError):
            load_recovered_checkpoint(corrupt)
    assert original['config']['controller_width'] == 2


def test_checkpoint_provenance_is_json_and_saving_does_not_modify_model(tmp_path):
    from jsr_repro.recovered_checkpoint import save_recovered_checkpoint
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        save_recovered_checkpoint(model, tmp_path / 'bad.pt', provenance={'invalid': Path('somewhere')})
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize('stage', ['controller', 'refine', 'joint'])
def test_fine_tuning_updates_selected_modules_and_saves_reloadable_model(tmp_path, stage):
    from jsr_repro.finetune_recovered import run
    from jsr_repro.recovered_checkpoint import load_recovered_checkpoint, read_recovered_checkpoint_metadata, save_recovered_checkpoint
    model = _model()
    source = tmp_path / 'initial.pt'
    save_recovered_checkpoint(model, source, provenance={'kind': 'test'})
    sample = tmp_path / 'sample.npz'
    _sample(sample, model)
    report = run([sample], tmp_path / stage, initialize=source, stage=stage, steps=2, lr=1e-3, seed=6)
    restored = load_recovered_checkpoint(tmp_path / stage / 'model.pt')
    for prefix in ('controller', 'refinenet'):
        changed = any(not torch.equal(value, restored.state_dict()[name])
                      for name, value in model.state_dict().items() if name.startswith(prefix + '.'))
        assert changed == (stage == 'joint' or prefix == ('controller' if stage == 'controller' else 'refinenet'))
    assert restored.refinement is True
    assert report['reloaded_output_max_abs'] == 0 and report['optimizer_resume'] is False
    metadata = read_recovered_checkpoint_metadata(tmp_path / stage / 'model.pt')
    assert metadata['provenance']['kind'] == 'fine_tuned'
    assert metadata['training']['stage'] == stage and metadata['training']['steps'] == 2
    assert json.loads((tmp_path / stage / 'training.json').read_text())['history']


def test_training_rejects_wrong_target_shape_and_frame_route(tmp_path):
    from jsr_repro.finetune_recovered import run
    from jsr_repro.recovered_checkpoint import save_recovered_checkpoint
    model = _model()
    initial = tmp_path / 'model.pt'
    save_recovered_checkpoint(model, initial, provenance={'kind': 'test'})
    sample = tmp_path / 'bad.npz'
    np.savez(sample, frames=np.ones((1, 6, 6), np.float32), flow=np.zeros((1, 6, 6, 2), np.float32),
             covariance=np.ones((3, 8, 8), np.float32), target=np.ones((3, 8, 8), np.float32))
    with pytest.raises(ValueError):
        run([sample], tmp_path / 'bad-run', initialize=initial, steps=1)
    np.savez(sample, frames=np.ones((2, 6, 6), np.float32), flow=np.zeros((2, 6, 6, 2), np.float32),
             covariance=np.ones((3, 8, 8), np.float32), target=np.ones((3, 9, 8), np.float32))
    with pytest.raises(ValueError):
        run([sample], tmp_path / 'bad-target', initialize=initial, steps=1)
