"""Export a complete, inspectable v9.8 network/weight kit for implementation.

All model tensors are compared directly with the original NPZ arrays. The
output contains portable Python source and converted weights, not the release
EXE, native extensions, or extracted shaders. Existing output is never replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproduction'))

from jsr_repro.model import REFERENCE_AMPLITUDE
from jsr_repro.network_bundle import describe_layers, export_pretrained_bundle, load_pretrained_bundle
from jsr_repro.recovered_checkpoint import load_recovered_checkpoint, save_recovered_checkpoint
from jsr_repro.recovered_frontend import FRONTEND_VERSION
from jsr_repro.recovered_model import load_recovered_jsr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def shape_description(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [shape_description(item) for item in value]
    if isinstance(value, dict):
        return {name: shape_description(item) for name, item in value.items()}
    return type(value).__name__


def weight_catalog(source: Path, output: Path, manifest: dict) -> dict:
    """Independently compare every exported tensor and write its source mapping."""
    columns = ['artifact', 'role', 'source_npz', 'source_npz_sha256', 'artifact_sha256',
               'source_npz_key', 'exported_tensor_key', 'full_state_key', 'tensor_role',
               'shape', 'source_dtype', 'pytorch_dtype', 'numel', 'tensor_sha256',
               'verification', 'metadata_value']
    rows = []
    tensors = scalars = metadata_scalars = 0
    for relative, record in sorted(manifest['files'].items()):
        artifact = output / 'weights' / relative
        source_path = source / record['source_npz']
        if sha256(artifact) != record['sha256'] or sha256(source_path) != record['source_npz_sha256']:
            raise ValueError(f'weight artifact/source checksum mismatch: {relative}')
        state = torch.load(artifact, map_location='cpu', weights_only=True)['state_dict']
        seen = set()
        with np.load(source_path, allow_pickle=False) as archive:
            for source_key in archive.files:
                array = np.asarray(archive[source_key])
                common = {'artifact': 'weights/' + relative, 'role': record['role'],
                          'source_npz': record['source_npz'], 'source_npz_sha256': record['source_npz_sha256'],
                          'artifact_sha256': record['sha256'], 'source_npz_key': source_key,
                          'shape': json.dumps(list(array.shape)), 'source_dtype': str(array.dtype),
                          'numel': int(array.size), 'tensor_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}
                if record['role'] == 'static_affine' and source_key == 'calibration_ids':
                    if array.tolist() != list(range(16)):
                        raise ValueError('static affine calibration IDs differ from the released schema')
                    rows.append(dict(common, exported_tensor_key='', full_state_key='metadata.calibration_ids',
                                     tensor_role='metadata; not a neural parameter', pytorch_dtype='',
                                     verification='released schema metadata', metadata_value=json.dumps(array.tolist())))
                    metadata_scalars += int(array.size)
                    continue
                key = source_key[4:] if record['role'] == 'controller' and source_key.startswith('net.') else source_key
                if key not in state:
                    raise ValueError(f'NPZ tensor has no exported counterpart: {record["source_npz"]}:{source_key}')
                tensor = state[key]
                if tensor.dtype != torch.float32 or array.dtype != np.float32 or not np.array_equal(array, tensor.numpy()):
                    raise ValueError(f'exported PT tensor differs from original NPZ: {record["source_npz"]}:{source_key}')
                full_key = ({'mean': 'feature_mean', 'std': 'feature_std'}[key]
                            if record['role'] == 'static_affine' else record['role'] + '.' + key)
                rows.append(dict(common, exported_tensor_key=key, full_state_key=full_key,
                                 tensor_role='buffer' if record['role'] == 'static_affine' else 'parameter',
                                 pytorch_dtype=str(tensor.dtype), verification='exact', metadata_value=''))
                seen.add(key)
                tensors += 1
                scalars += tensor.numel()
        if seen != set(state):
            raise ValueError(f'exported component has unmapped tensors: {relative}')
    target = output / 'catalog' / 'weight_catalog.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return {'weight_artifacts': len(manifest['files']), 'exported_tensors': tensors,
            'exported_scalars': scalars, 'catalog_rows': len(rows),
            'calibration_metadata_scalars': metadata_scalars, 'all_exported_tensors_equal_original_npz': True}


def architecture_catalog(model, manifest: dict) -> tuple[dict, dict, tuple]:
    """Record the actual full-core leaf forward trace, with 64x64 output."""
    trace, handles = [], []
    for name, module in model.named_modules():
        if not name or list(module.children()):
            continue
        def hook(layer, args, result, name=name):
            trace.append({'index': len(trace), 'name': name, 'type': type(layer).__name__,
                          'input': shape_description(args), 'output': shape_description(result)})
        handles.append(module.register_forward_hook(hook))
    frames = torch.full((1, 4, 32, 32), .1)
    flow = torch.zeros((1, 4, 32, 32, 2))
    covariance = torch.tensor([1.1, .1, .9]).view(1, 3, 1, 1).expand(1, 3, 64, 64)
    try:
        with torch.no_grad():
            result = model(frames, flow, covariance, diagnostics=True)
    finally:
        for handle in handles:
            handle.remove()
    layers = {}
    for role, module in (('controller', model.controller), ('refinenet', model.refinenet)):
        layers[role] = []
        for layer in describe_layers(module):
            row = dict(layer, name=role + '.' + layer['name'])
            row['full_tensor_keys'] = [row['name'] + '.' + key for key in layer['tensors']]
            layers[role].append(row)
    routes = {}
    for k, route in manifest['routes'].items():
        controller = manifest['files'][route['controller']]
        refine = manifest['files'][route['refinenet']]
        routes[k] = {'controller': 'weights/' + route['controller'], 'refinenet': 'weights/' + route['refinenet'],
                     'controller_parameters': controller['scalar_count'], 'refinenet_parameters': refine['scalar_count'],
                     'trainable_parameters': controller['scalar_count'] + refine['scalar_count'],
                     'affine_buffers': 302}
    architecture = {
        'format': 'jsr-v98-architecture-catalog-v1', 'release': 'v9.8',
        'frontend_version': FRONTEND_VERSION, 'reference_amplitude': REFERENCE_AMPLITUDE,
        'input_contract': {'frames': '[B,K,Hframe,Wframe] black-subtracted linear CFA values',
                           'flow': '[B,K,Hreference,Wreference,2] reference-to-frame native (dx,dy)',
                           'covariance': '[B,3,Hout,Wout] inverse covariance (xx,xy,yy), native coordinate units'},
        'output_contract': {'rgb': '[B,3,Hout,Wout] final linear RGB',
                            'learned': '[B,3,Hout,Wout]', 'legacy': '[B,3,Hout,Wout]'},
        'components': {
            'controller': {'class': 'Controller', 'input_channels': 151, 'output_channels': 36,
                           'width': model.controller.enc[0][0].out_channels, 'depth': len(model.controller.enc),
                           'parameter_count': sum(p.numel() for p in model.controller.parameters()),
                           'parameter_layers': layers['controller'],
                           'functional_operations': ['Each encoder/bottleneck/decoder block has two zero-padded 3x3 Conv2d layers, each followed by exact-erf GELU.',
                                                     'Three floor AvgPool2d(kernel=2,stride=2) reductions.',
                                                     'Decoder resize is bilinear to the corresponding skip size, align_corners=False.',
                                                     'Concatenation order is [upsampled decoder, encoder skip].',
                                                     'Final 1x1 convolution produces 12 continuous kernel coefficients per RGB channel.']},
            'refinenet': {'class': 'RefineNet', 'input_channels': 6, 'output_channels': 3,
                          'width': model.refinenet.inp.out_channels, 'residual_blocks': len(model.refinenet.body),
                          'parameter_count': sum(p.numel() for p in model.refinenet.parameters()),
                          'input_order': '[learned R,G,B, legacy R,G,B]',
                          'guide_channels': [1, 4], 'residual_order': ['dG', 'd(R-G)', 'd(B-G)'],
                          'parameter_layers': layers['refinenet'],
                          'functional_operations': ['h = Conv3x3(input6); guide = Conv3x3(input[:,[1,4]])',
                                                    'For each block: h = h + Conv3x3(GELU(Conv3x3(GELU(h))))',
                                                    'Then h = h*(1+0.1*tanh(gamma_i(guide))) + beta_i(guide)',
                                                    'gamma_i and beta_i are separate 1x1 convolutions for each block.',
                                                    'A final 1x1 convolution returns color-difference residuals.',
                                                    'Before the network, both RGB branches are divided by amplitude/reference_amplitude.',
                                                    'Green and color-difference residuals anchor to the first RGB branch; normalized RGB is clamped to [0,1], then amplitude scale is restored.']},
        },
        'nonparameterized_steps': ['4x4 RAW gather and weighted count/mean/population-standard-deviation statistics in 3 colors x 16 bins x 3 fields.',
                                  '151 features = 144 phase statistics + 3 legacy RGB + 3 smoothed structure-tensor fields + 1 zero-noise field.',
                                  'Local amplitude: sqrt(Gaussian(radius16,sigma4)(mean_RGB(legacy^2))).',
                                  'Feature channels divided by amplitude to their released degree (0,1,2), then frozen mean/std affine.',
                                  'RAW continuous kernel correction with 12 basis coefficients per color and cap*tanh(dot/cap), default cap5.',
                                  'Optional fixed-coefficient global radial LCA applies to both RGB branches before RefineNet and recomputes amplitude.'],
        'routes': routes,
        'forward_trace': {'scope': 'One complete K4 core invocation, supplied registration/covariance, LCA disabled; constant numerical shape probe, not a quality experiment.',
                          'input_shapes': {'frames': list(frames.shape), 'flow': list(flow.shape), 'covariance': list(covariance.shape)},
                          'outputs': shape_description(result), 'leaf_modules': trace},
        'remaining_external_inputs': ['registration', 'RAW-to-guide construction', 'covariance output-grid scheduling', 'automatic LCA estimation'],
        'version_boundary': 'These original weights belong to v9.8/v9.7 core. They are not author weights for the independent speech-inspired Transformer or v17 Stormer/Fast paths.',
    }
    return architecture, result, (frames, flow, covariance)


def copy_portable_source(output: Path) -> int:
    source = output / 'source'
    package = ROOT / 'reproduction' / 'jsr_repro'
    for path in sorted(package.rglob('*.py')):
        target = source / 'reproduction' / 'jsr_repro' / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    assets = package / 'spectral_assets'
    if assets.exists():
        for path in sorted(assets.rglob('*')):
            if path.is_file():
                target = source / 'reproduction' / 'jsr_repro' / path.relative_to(package)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    shutil.copyfile(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    for path in sorted(ROOT.glob('*LICENSE*')):
        if path.is_file():
            shutil.copyfile(path, source / path.name)
    (source / 'README.md').write_text('# Portable JSR Python source\n\nRun `python -m pip install -e .` in this directory. The network kit weights and examples are one directory above; see [kit README](../README.md) and [architecture guide](../docs/NETWORK_KIT.md).\n', encoding='utf-8')
    return sum(path.is_file() for path in source.rglob('*'))


def verify_all_routes(source: Path, output: Path) -> dict:
    """Check requested K and every complete model tensor against original NPZ."""
    reports = {}
    for k in range(1, 15):
        original = load_recovered_jsr(source, k)
        restored = load_pretrained_bundle(output / 'weights', k)
        before, after = original.state_dict(), restored.state_dict()
        exact = before.keys() == after.keys() and all(torch.equal(value, after[name]) for name, value in before.items())
        if not exact or restored.expected_frames != k:
            raise RuntimeError(f'original NPZ versus exported complete K={k} route mismatch')
        reports[str(k)] = {'requested_k': k, 'loaded_k': restored.expected_frames,
                           'state_tensors': len(before), 'all_tensors_exact': True}
    return reports


def export_kit(weights_dir: str | Path, output_dir: str | Path, *, example: str | Path | None = None) -> dict:
    source, output = Path(weights_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError('network kit output already exists; select a new directory')
    doc = ROOT / 'docs' / 'NETWORK_KIT.md'
    diagrams = ROOT / 'docs' / 'network-kit'
    if not doc.is_file() or not diagrams.is_dir() or not any(diagrams.glob('*.svg')):
        raise ValueError('NETWORK_KIT.md and its architecture diagrams must exist before exporting')
    if example is not None and not Path(example).is_file():
        raise ValueError('engineering example does not exist')
    torch.set_num_threads(2)
    bundle = export_pretrained_bundle(source, output / 'weights')
    shutil.copyfile(source / 'manifest.json', output / 'weights' / 'source_manifest.json')
    catalog = weight_catalog(source, output, bundle)
    verified_routes = verify_all_routes(source, output)
    model = load_pretrained_bundle(output / 'weights', 4)
    architecture, reference, inputs = architecture_catalog(model, bundle)
    write_json(output / 'catalog' / 'architecture.json', architecture)
    (output / 'catalog' / 'model_repr.txt').write_text(str(model) + '\n', encoding='utf-8')
    route = bundle['routes']['4']
    provenance = {'kind': 'original_release', 'release': 'v9.8', 'weight_family': bundle['weight_family'], 'k': 4,
                  'source_manifest_sha256': bundle['source_manifest_sha256'],
                  'original_weights': {role: {'file': bundle['files'][path]['source_npz'],
                                              'sha256': bundle['files'][path]['source_npz_sha256']}
                                       for role, path in route.items()},
                  'static_affine_npz_sha256': bundle['files'][bundle['static_affine']]['source_npz_sha256'],
                  'conversion': 'exact FP32 tensor import; no training or quantization'}
    checkpoint = output / 'checkpoints' / 'pretrained_k04.pt'
    metadata = save_recovered_checkpoint(model, checkpoint, provenance=provenance)
    restored = load_recovered_checkpoint(checkpoint)
    if model.state_dict().keys() != restored.state_dict().keys() or any(
            not torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()):
        raise RuntimeError('complete K4 checkpoint tensor reload mismatch')
    with torch.no_grad():
        reloaded_rgb = restored(*inputs)['rgb']
    checkpoint_error = float((reference['rgb'] - reloaded_rgb).abs().max())
    if checkpoint_error != 0:
        raise RuntimeError('complete K4 checkpoint output reload mismatch')
    write_json(output / 'checkpoints' / 'pretrained_k04.json', metadata)
    source_files = copy_portable_source(output)
    docs = output / 'docs'
    docs.mkdir()
    shutil.copyfile(doc, docs / doc.name)
    recovery = ROOT / 'docs' / 'REA_RECOVERY.md'
    if recovery.exists():
        shutil.copyfile(recovery, docs / recovery.name)
    for path in sorted(diagrams.rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.svg', '.png', '.md'):
            target = docs / 'network-kit' / path.relative_to(diagrams)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    # The guide links these next to its diagrams. Copy this export's freshly
    # generated catalogs so a clean run has working, current links as well.
    for name in ('architecture.json', 'weight_catalog.csv', 'model_repr.txt'):
        target = docs / 'network-kit' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / 'catalog' / name, target)
    example_file = None
    if example is not None:
        examples = output / 'examples'
        examples.mkdir()
        shutil.copyfile(Path(example), examples / 'engineering_k4.npz')
        (examples / 'README.md').write_text('# Engineering example\n\nThis fixed K4 boundary sample has `target = original K4 output * 0.98 + 0.001`. It exercises optimizer/save/reload behavior. It is a procedural target perturbation, not independent scene ground truth or a photo-quality result.\n', encoding='utf-8')
        example_file = 'examples/engineering_k4.npz'
    sample = example_file or 'YOUR_BOUNDARY_SAMPLE.npz'
    (output / 'README.md').write_text(
        '# JSR v9.8 网络结构与原权重工具包\n\n'
        '从本目录运行 `python -m pip install -e ./source`。完整说明见 [网络、张量与微调指南](docs/NETWORK_KIT.md)，取证边界见 [逆向记录](docs/REA_RECOVERY.md)。\n\n'
        '包含 K=1..14 的 13 个 Controller、8 个 RefineNet（共享组件去重）及静态 affine；所有 PT 张量与对应原 NPZ 逐值相等。`weights/manifest.json` 给出 K 路由与 SHA-256；`catalog/weight_catalog.csv` 给出每个原 NPZ key 到 PyTorch full state key 的映射。calibration_ids 是元数据，不是神经参数。\n\n'
        '`catalog/architecture.json` 包含逐层超参数及 64×64 输出的实际完整网络 forward leaf-shape trace；`catalog/model_repr.txt` 可直接查看模块树。\n\n'
        '```python\nfrom jsr_repro.network_bundle import load_pretrained_bundle\nmodel = load_pretrained_bundle("weights", k=4)\n# result = model(frames, flow, covariance)\n```\n\n'
        '完整 K4 checkpoint 也可以直接加载：\n\n'
        '```python\nfrom jsr_repro.recovered_checkpoint import load_recovered_checkpoint\nmodel = load_recovered_checkpoint("checkpoints/pretrained_k04.pt")\n```\n\n'
        '最小微调命令（原坐标 boundary NPZ：frames、flow、covariance、线性 RGB target）：\n\n'
        f'```bash\npython -m jsr_repro.finetune_recovered --bundle-dir weights --k 4 --samples {sample} --output runs/my_first_finetune --stage joint --steps 1 --lr 1e-6\n```\n\n'
        '微调阶段可选 controller/refine/joint。新 checkpoint 记录 fine_tuned 来源；`--initialize` 是模型 warm start，会重置 optimizer。工程样例的 target 为原输出扰动，不是真实 GT。\n\n'
        '当前原权重属于 v9.8/v9.7 core。source 中保留的历史 speech-inspired Transformer 是独立假设实现，没有对应作者权重；此包不把它或 v17 三模式归为已恢复网络。配准、RAW 导引、covariance 网格调度与自动 LCA 估计仍是外部边界。\n', encoding='utf-8')
    (output / 'requirements.txt').write_text('-e ./source\n', encoding='utf-8')
    summary = {'format': 'jsr-v98-network-kit-v1', 'release': 'v9.8', 'ks': list(range(1, 15)),
               'unique_controllers': sum(record['role'] == 'controller' for record in bundle['files'].values()),
               'unique_refinenets': sum(record['role'] == 'refinenet' for record in bundle['files'].values()),
               'unique_networks': bundle['unique_networks'], 'catalog': catalog,
               'parameters_per_route': {k: record['trainable_parameters'] for k, record in architecture['routes'].items()},
               'verified_routes': verified_routes,
               'source_files': source_files, 'example': example_file,
               'full_k4_checkpoint': {'file': 'checkpoints/pretrained_k04.pt', 'sha256': sha256(checkpoint),
                                      'state_tensors': len(model.state_dict()),
                                      'state_equal': True, 'reloaded_output_max_abs': checkpoint_error},
               'source_manifest_sha256': bundle['source_manifest_sha256'],
               'numpy': np.__version__, 'torch': str(torch.__version__),
               'scope': 'Recovered v9.8 reconstruction core with supplied registration/covariance; original training and v17 paths remain outside this export.'}
    write_json(output / 'export_summary.json', summary)
    files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha256(path)}
             for path in sorted(output.rglob('*')) if path.is_file()}
    write_json(output / 'kit_manifest.json', {'format': 'jsr-v98-network-kit-file-manifest-v1',
                                             'self_excluded': ['kit_manifest.json', 'SHA256SUMS.txt'], 'files': files})
    all_files = [path for path in sorted(output.rglob('*')) if path.is_file()]
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(path)}  {path.relative_to(output).as_posix()}\n' for path in all_files), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--example', type=Path, help='Optional engineering-only supervised boundary NPZ to copy')
    args = parser.parse_args()
    summary = export_kit(args.weights_dir, args.output, example=args.example)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
