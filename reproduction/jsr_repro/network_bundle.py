"""Portable, deduplicated, exact original v9.8 PyTorch weight components."""
from __future__ import annotations

import argparse
import hashlib
import json
import operator
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .model import REFERENCE_AMPLITUDE
from .recovered_frontend import FRONTEND_VERSION
from .recovered_model import RecoveredJSR
from .weights import load_controller_npz, load_refinenet_npz, load_static_affine_npz, verify_manifest_file

FORMAT = 'jsr-v98-pytorch-bundle-v1'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _k(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('k must be an integer from 1 to 14')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError('k must be an integer from 1 to 14') from exc
    if not 1 <= result <= 14:
        raise ValueError('k must be an integer from 1 to 14')
    return result


def describe_layers(module) -> list[dict]:
    """Every parameter-bearing layer, in module order, with OIHW shapes."""
    rows = []
    for name, layer in module.named_modules():
        parameters = list(layer.named_parameters(recurse=False))
        if not parameters:
            continue
        row = {'name': name, 'type': type(layer).__name__,
               'parameters': sum(value.numel() for _, value in parameters),
               'tensors': {key: list(value.shape) for key, value in parameters}}
        if isinstance(layer, torch.nn.Conv2d):
            row.update(in_channels=layer.in_channels, out_channels=layer.out_channels,
                       kernel=list(layer.kernel_size), stride=list(layer.stride),
                       padding=list(layer.padding), groups=layer.groups)
        rows.append(row)
    return rows


def export_pretrained_bundle(weights_dir: str | Path, output_dir: str | Path, *, ks=None) -> dict:
    """Convert selected original routes without changing any tensor value.

    Components shared by K routes are saved once. Only CPU state dictionaries
    are serialized; architecture/provenance live in the JSON manifest.
    """
    source, output = Path(weights_dir), Path(output_dir)
    if output.exists():
        raise ValueError('output directory already exists')
    source_manifest_path = source / 'manifest.json'
    original = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    selected = sorted({_k(k) for k in (range(1, 15) if ks is None else ks)})
    if not selected or any(str(k) not in original['routes'] for k in selected):
        raise ValueError('selected route is unavailable in source manifest')
    names = {original['static_affine']}
    for k in selected:
        names.update(original['routes'][str(k)].values())
    for name in names:
        if Path(name).name != name or not name.endswith('.npz'):
            raise ValueError('invalid source component path')
        verify_manifest_file(source / name, source_manifest_path)
    (output / 'modules').mkdir(parents=True)
    manifest = {'format': FORMAT, 'release': 'v9.8', 'weight_family': 'v9.7 weights reused by v9.8',
                'frontend_version': FRONTEND_VERSION, 'reference_amplitude': REFERENCE_AMPLITUDE,
                'source_manifest_sha256': _sha256(source_manifest_path), 'selected_ks': selected,
                'static_affine_calibration_ids': list(range(16)),
                'tensor_layout': 'PyTorch Conv2d OIHW; float32; no transpose, quantization or retraining',
                'routes': {}, 'files': {}, 'static_affine': None}
    mapping = {}
    for name in sorted(names):
        if name == original['static_affine']:
            mean, std = load_static_affine_npz(source / name)
            state, role, architecture, layers = {'mean': mean, 'std': std}, 'static_affine', {}, []
        else:
            role = 'controller' if '_controller_' in name else 'refinenet'
            module = load_controller_npz(source / name) if role == 'controller' else load_refinenet_npz(source / name)
            state = {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}
            architecture = ({'width': module.enc[0][0].out_channels, 'depth': 3} if role == 'controller'
                            else {'width': module.inp.out_channels, 'blocks': len(module.body)})
            layers = describe_layers(module)
        relative = 'modules/' + Path(name).with_suffix('.pt').name
        path = output / relative
        torch.save({'state_dict': state}, path)
        restored = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
        if set(restored) != set(state) or any(not torch.equal(restored[key], value) for key, value in state.items()):
            raise RuntimeError(f'export changed original tensor values: {name}')
        mapping[name] = relative
        manifest['files'][relative] = {
            'role': role, 'bytes': path.stat().st_size, 'sha256': _sha256(path),
            'source_npz': name, 'source_npz_sha256': original['files'][name]['sha256'],
            'source_npz_bytes': original['files'][name]['bytes'], 'architecture': architecture,
            'tensor_count': len(state), 'scalar_count': sum(value.numel() for value in state.values()),
            'tensors': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'numel': value.numel()}
                        for key, value in state.items()}, 'layers': layers,
            'export_tensor_equality': 'exact'}
    manifest['static_affine'] = mapping[original['static_affine']]
    for k in selected:
        manifest['routes'][str(k)] = {role: mapping[name] for role, name in original['routes'][str(k)].items()}
    manifest['unique_networks'] = sum(record['role'] != 'static_affine' for record in manifest['files'].values())
    manifest['verified_tensor_count'] = sum(record['tensor_count'] for record in manifest['files'].values())
    manifest['verified_scalar_count'] = sum(record['scalar_count'] for record in manifest['files'].values())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def _component_path(root: Path, manifest: dict, name: str) -> Path:
    if (not isinstance(name, str) or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
            or PurePosixPath(name).parts[:1] != ('modules',) or not name.endswith('.pt')):
        raise ValueError('invalid component path')
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or name not in manifest['files']:
        raise ValueError('component path is outside bundle')
    return path


def _component(root: Path, manifest: dict, name: str) -> dict[str, torch.Tensor]:
    path = _component_path(root, manifest, name)
    record = manifest['files'][name]
    if path.stat().st_size != record['bytes'] or _sha256(path) != record['sha256']:
        raise ValueError(f'component checksum/size mismatch: {name}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'state_dict'} or not isinstance(payload['state_dict'], dict):
        raise ValueError('invalid component state dictionary')
    state = payload['state_dict']
    if not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                        or value.dtype != torch.float32 or not torch.isfinite(value).all()
                        for key, value in state.items()):
        raise ValueError('component must contain finite float32 tensors')
    return state


def load_pretrained_bundle(bundle_dir: str | Path, k: int) -> RecoveredJSR:
    """Load an original route from standalone PT files, restoring requested K."""
    root, k = Path(bundle_dir), _k(k)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('format') != FORMAT or manifest.get('frontend_version') != FRONTEND_VERSION
            or manifest.get('reference_amplitude') != REFERENCE_AMPLITUDE):
        raise ValueError('unsupported network bundle format or computation version')
    if str(k) not in manifest['routes']:
        raise ValueError(f'K={k} route is not exported')
    route = manifest['routes'][str(k)]
    if set(route) != {'controller', 'refinenet'}:
        raise ValueError('invalid network route roles')
    for name in (*route.values(), manifest['static_affine']):
        _component_path(root, manifest, name)
    # Resolve/verify paths before reading architecture from their records.
    controller_state = _component(root, manifest, route['controller'])
    refine_state = _component(root, manifest, route['refinenet'])
    affine = _component(root, manifest, manifest['static_affine'])
    controller = manifest['files'][route['controller']]
    refine = manifest['files'][route['refinenet']]
    if controller['role'] != 'controller' or refine['role'] != 'refinenet':
        raise ValueError('component role differs from route')
    c, r = controller['architecture'], refine['architecture']
    if c.get('depth') != 3:
        raise ValueError('unsupported Controller depth')
    model = RecoveredJSR(controller_width=c['width'], refine_width=r['width'], refine_blocks=r['blocks'])
    model.controller.load_state_dict(controller_state, strict=True)
    model.refinenet.load_state_dict(refine_state, strict=True)
    if (set(affine) != {'mean', 'std'} or affine['mean'].shape != model.feature_mean.shape
            or affine['std'].shape != model.feature_std.shape or (affine['std'] <= 0).any()):
        raise ValueError('invalid static affine component')
    model.feature_mean.copy_(affine['mean'])
    model.feature_std.copy_(affine['std'])
    model.expected_frames = k
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ks', type=int, nargs='+')
    args = parser.parse_args()
    manifest = export_pretrained_bundle(args.weights_dir, args.output, ks=args.ks)
    print(json.dumps({key: manifest[key] for key in ('selected_ks', 'unique_networks', 'verified_tensor_count', 'verified_scalar_count')}, indent=2))


if __name__ == '__main__':
    main()
