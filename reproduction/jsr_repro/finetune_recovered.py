"""Small supervised fine-tuning entry for explicit recovered-boundary NPZ data.

Targets are linear RGB in the same units as RAW; no alignment, tone mapping,
gain fitting or author training recipe is synthesized. Initialization from a
checkpoint starts a new AdamW run and does not resume optimizer state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .recovered_checkpoint import load_recovered_checkpoint, read_recovered_checkpoint_metadata, save_recovered_checkpoint
from .recovered_model import load_recovered_jsr


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sample(path, device):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {'frames', 'flow', 'covariance', 'target'}
        optional = {'fshift', 'color_index', 'origin_yx', 'lca_coefficients_rb'}
        if not required <= set(archive.files) or set(archive.files) - required - optional:
            raise ValueError('supervised NPZ requires frames, flow, covariance, target and only supported boundary options')
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name in required:
        if arrays[name].dtype.kind != 'f' or not np.isfinite(arrays[name]).all():
            raise ValueError(f'{name} must contain finite floating-point values')
    frames, flow, covariance = [torch.from_numpy(np.asarray(arrays[name], np.float32))[None].to(device)
                                for name in ('frames', 'flow', 'covariance')]
    if frames.ndim != 4 or flow.ndim != 5 or covariance.ndim != 4 or covariance.shape[1] != 3:
        raise ValueError('boundary shapes must be frames[K,H,W], flow[K,N,M,2], covariance[3,Ho,Wo]')
    h, w = covariance.shape[-2:]
    target = arrays['target']
    if target.shape == (h, w, 3):
        target = target.transpose(2, 0, 1).copy()
    if target.shape != (3, h, w):
        raise ValueError('target must be CHW or HWC RGB matching the covariance output grid')
    target = torch.from_numpy(np.asarray(target, np.float32))[None].to(device)
    options = {}
    if 'origin_yx' in arrays:
        options['origin_yx'] = arrays['origin_yx']
    if 'color_index' in arrays:
        options['color_index'] = torch.from_numpy(arrays['color_index']).to(device)
    if 'fshift' in arrays:
        options['fshift'] = torch.from_numpy(np.asarray(arrays['fshift'], np.float32))[None].to(device)
    if 'lca_coefficients_rb' in arrays:
        options['lca_coefficients_rb'] = torch.from_numpy(np.asarray(arrays['lca_coefficients_rb'], np.float32)).to(device)
    record = {'file': path.name, 'sha256': _sha256(path), 'k': frames.shape[1],
              'output_shape': [h, w], 'target': 'linear RGB in RAW units'}
    return (frames, flow, covariance), target, options, record


def run(samples, output, *, initialize=None, weights_dir=None, bundle_dir=None, k=None,
        stage='joint', steps=1, lr=1e-5, seed=20261027, device='cpu'):
    """Start a new supervised run and save a reloadable model plus loss log.

    Samples are visited in deterministic round-robin order, one sample/step.
    Stages train the Controller on learned RGB, or RefineNet/joint on final RGB.
    There is no optimizer-resume feature; --initialize is a model warm start.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('output directory already exists; select a new directory')
    if sum(source is not None for source in (initialize, weights_dir, bundle_dir)) != 1:
        raise ValueError('select exactly one initialization source: checkpoint, weights directory or model bundle')
    if stage not in ('controller', 'refine', 'joint') or type(steps) is not int or steps < 1:
        raise ValueError('stage must be controller/refine/joint and steps must be a positive integer')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning rate must be finite and positive')
    if not samples:
        raise ValueError('at least one supervised NPZ sample is required')
    torch.manual_seed(seed)
    data = [_sample(path, device) for path in samples]
    if initialize is not None:
        model = load_recovered_checkpoint(initialize, device=device)
        source = {'kind': 'checkpoint_warm_start', 'sha256': _sha256(initialize),
                  'metadata': read_recovered_checkpoint_metadata(initialize)}
    else:
        if type(k) is not int or not 1 <= k <= 14:
            raise ValueError('original weights or bundle initialization requires integer --k from 1 to 14')
        if bundle_dir is not None:
            from .network_bundle import load_pretrained_bundle
            model = load_pretrained_bundle(bundle_dir, k).to(device)
            source = {'kind': 'original_release_bundle', 'k': k,
                      'manifest_sha256': _sha256(Path(bundle_dir) / 'manifest.json')}
        else:
            model = load_recovered_jsr(weights_dir, k).to(device)
            source = {'kind': 'original_release_npz', 'k': k,
                      'manifest_sha256': _sha256(Path(weights_dir) / 'manifest.json')}
    if k is not None and model.expected_frames not in (None, k):
        raise ValueError('requested K differs from the initialization checkpoint route')
    expected_k = model.expected_frames if model.expected_frames is not None else (k if k is not None else data[0][0][0].shape[1])
    if any(item[0][0].shape[1] != expected_k for item in data):
        raise ValueError('all supervised samples must match the checkpoint K route')
    model.expected_frames = expected_k
    inference_refinement = model.refinement
    if stage != 'controller' and not inference_refinement:
        raise ValueError('refine/joint stage requires an initialization with refinement enabled')
    model.controller.requires_grad_(stage in ('controller', 'joint'))
    model.refinenet.requires_grad_(stage in ('refine', 'joint'))
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
    output.mkdir(parents=True)
    history = []
    model.train()
    if stage == 'controller':
        model.refinement = False
    for step in range(steps):
        index = step % len(data)
        inputs, target, options, _ = data[index]
        optimizer.zero_grad(set_to_none=True)
        result = model(*inputs, **options)
        prediction = result['learned'] if stage == 'controller' else result['rgb']
        loss = (prediction - target).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('fine-tuning loss became nonfinite')
        loss.backward()
        if any(value.grad is not None and not torch.isfinite(value.grad).all() for value in parameters):
            raise RuntimeError('fine-tuning gradients became nonfinite')
        gradient_norm = math.sqrt(sum(float(value.grad.detach().square().sum())
                                      for value in parameters if value.grad is not None))
        optimizer.step()
        history.append({'step': step + 1, 'sample': index, 'loss': float(loss.detach()), 'gradient_norm': gradient_norm})
    model.refinement = inference_refinement
    model.eval()
    training = {'stage': stage, 'steps': steps, 'lr': lr, 'seed': seed, 'device': str(device),
                'optimizer': 'AdamW', 'weight_decay': 0., 'optimizer_resume': False,
                'sample_order': 'round_robin', 'loss': 'linear RGB L1; learned for controller, final RGB otherwise',
                'samples': [item[3] for item in data]}
    save_recovered_checkpoint(model, output / 'model.pt', provenance={'kind': 'fine_tuned', 'source': source}, training=training)
    restored = load_recovered_checkpoint(output / 'model.pt', device=device)
    inputs, _, options, _ = data[0]
    with torch.no_grad():
        reference = model(*inputs, **options)['rgb']
        reloaded = restored(*inputs, **options)['rgb']
    report = dict(training, history=history, checkpoint='model.pt',
                  reloaded_output_max_abs=float((reference - reloaded).abs().max()),
                  checkpoint_sha256=_sha256(output / 'model.pt'),
                  source=source, note='Model warm start only; optimizer state is reset for every invocation.')
    (output / 'training.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--initialize', type=Path, help='Tensor-only recovered checkpoint; model warm start, not optimizer resume')
    source.add_argument('--weights-dir', type=Path, help='Verified original NPZ weight directory')
    source.add_argument('--bundle-dir', type=Path, help='Exported original model bundle')
    parser.add_argument('--k', type=int, help='Required with --weights-dir or --bundle-dir')
    parser.add_argument('--samples', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('controller', 'refine', 'joint'), default='joint')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run(args.samples, args.output, initialize=args.initialize, weights_dir=args.weights_dir,
                 bundle_dir=args.bundle_dir, k=args.k, stage=args.stage, steps=args.steps,
                 lr=args.lr, seed=args.seed, device=args.device)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
