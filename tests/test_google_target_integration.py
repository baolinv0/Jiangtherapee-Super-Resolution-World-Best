"""Checkpoint and CLI contracts across the target-stage protocol boundary."""
import json

import pytest

from jsr_repro.bracket_data import make_burst_dataset, save_burst
from jsr_repro.evaluate_transformer import evaluate
from jsr_repro.infer_transformer import infer
from jsr_repro.train_transformer import load_checkpoint, run_training
from test_spectral_data import options, write_manifest


@pytest.mark.parametrize('protocol,stage', [
    ('spectral-camera-v2', 'pre_optics'),
    ('spectral-camera-v3', 'post_pixel'),
])
def test_checkpoint_resume_evaluation_and_output_stage(tmp_path, protocol, stage):
    """A protocol migration must not silently change a checkpoint's labels."""
    manifest = write_manifest(tmp_path)
    opts = options(protocol=protocol)
    cfg = {'seed':21, 'threads':1, 'device':'cpu', 'output':str(tmp_path/'training'),
           'model':{'width':4, 'heads':1, 'window':2, 'scale':2, 'frames':7},
           'data':{'manifest':str(manifest), 'options':opts},
           'train':{'steps':2, 'batch_size':1, 'validate_every':1,
                    'crop_border':4, 'lr':.0001, 'clip_grad':1.}}
    run_training(cfg, stop_after=1)
    checkpoint = tmp_path/'training/last.pt'
    _, state = load_checkpoint(checkpoint)
    construction = state['data_identity']['construction']
    if protocol == 'spectral-camera-v2':
        # Legacy state layout is intentionally unchanged, so pre-v3 saves
        # remain evaluable and resumable with their original recipe.
        assert set(construction) == {'protocol', 'spectral_assets'}
    else:
        assert construction['target_stage'] == stage
    assert run_training(cfg, resume=checkpoint)['steps_completed'] == 2
    report = evaluate(checkpoint, manifest, tmp_path/'evaluation.json', limit=1)
    assert report['data_protocol'] == protocol
    assert report['target_stage'] == stage
    dataset = make_burst_dataset(manifest, 'test', opts, 21)
    sample = dataset[0]
    assert json.loads(sample['metadata'])['target_stage'] == stage
    save_burst(tmp_path/'burst.npz', sample)
    prediction = infer(checkpoint, tmp_path/'burst.npz', tmp_path/'inference', alignment='provided')
    assert prediction['training_target_stage'] == stage

    changed = json.loads(json.dumps(cfg))
    changed['data']['options'].update(protocol='spectral-camera-v3', target_stage='post_optics')
    with pytest.raises(ValueError, match='resume data configuration changed'):
        run_training(changed, resume=checkpoint)
