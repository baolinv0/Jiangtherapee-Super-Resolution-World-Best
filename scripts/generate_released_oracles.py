"""Regenerate numeric test oracles from the digest-pinned official v9.8 EXE.

Requires Windows x64 CPython 3.10 and requirements-released-oracle.txt.
The EXE is never launched. After complete release verification this script
imports three extracted native modules in the current Python process.
Original modules remain in the chosen local output/_modules directory; do not
redistribute that directory. The three NPZ fixtures contain numbers only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import zlib

from extract_released_shaders import archive_table, extract, read_at

ROOT = Path(__file__).resolve().parents[1]
MODULE_HASHES = {
    'bsr_steer': '20acf56aa47dd483759bf43b2d948a8bf1459936050b65f3fcd96737bafedbe4',
    'v82_lca': '5def15dc2da130048718dfa4cf3d385d40ff337f1541c3d4b8c1276cbb2bccc8',
    'local_v7_frontend': '55a86032a7f040eb24f605fea2d64773d926c34a629e5ba1de00579913eef284',
}
FIXTURE_HASHES = {
    'recovered_covariance_oracle.npz': '8f941fbd1e1848a02d2544c5207994d6942ddd0841886d3943d749d4e3a6da63',
    'recovered_lca_warp_oracle.npz': '64fd3636609ed273c988ca3e0cad9f56bef72d040652016b6a9faba1d96079c6',
    'recovered_frontend_oracle.npz': '16fb10b6a3e966b3d3ac8e3af19007b7871fbeee3cd8f99bf534ab933048c35a',
}
AFFINE_HASH = '606f16a60a7738a5193de765d8408e4de810c684cddb2286aadd7cb7556854f3'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_runtime() -> dict:
    if (platform.system() != 'Windows' or platform.python_implementation() != 'CPython'
            or sys.version_info[:2] != (3, 10) or sys.maxsize <= 2 ** 32):
        raise RuntimeError('Use Windows x64 CPython 3.10 with requirements-released-oracle.txt')
    versions = {name: importlib.metadata.version(name)
                for name in ('numpy', 'opencv-python-headless')}
    if versions != {'numpy': '2.2.6', 'opencv-python-headless': '5.0.0.93'}:
        raise RuntimeError(f'Probe dependencies must match the release: {versions}')
    return {'python': sys.version, 'platform': platform.platform(), **versions}


def extract_modules(executable: Path, output: Path, release: dict) -> tuple[dict, dict]:
    target = output / '_modules'
    target.mkdir()
    modules = {}
    with executable.open('rb') as source:
        start, table, _ = archive_table(source, release['backend_resource'])

        def unpack(name: str) -> tuple[bytes, dict]:
            offset, packed_size, size, compressed, kind = table[name]
            packed = read_at(source, start + offset, packed_size)
            data = zlib.decompress(packed) if compressed else packed
            if len(data) != size:
                raise ValueError(f'Archive entry size mismatch: {name}')
            return data, {'archive_name': name, 'archive_offset': start + offset,
                          'compressed_bytes': packed_size, 'bytes': size,
                          'sha256': hashlib.sha256(data).hexdigest()}

        for name, expected in MODULE_HASHES.items():
            filename = name + '.cp310-win_amd64.pyd'
            data, info = unpack(filename)
            if info['sha256'] != expected:
                raise ValueError(f'Released module SHA-256 mismatch: {name}')
            (target / filename).write_bytes(data)
            modules[name] = dict(info, local_file='_modules/' + filename)

        cv2_data, cv2_info = unpack('cv2\\version.py')
        numpy_data, numpy_info = unpack('numpy-2.2.6.dist-info\\METADATA')
        cv2_version = re.search(r'opencv_version\s*=\s*"([^"]+)"', cv2_data.decode('utf-8')).group(1)
        numpy_version = re.search(r'^Version:\s*(.+)$', numpy_data.decode('utf-8'), re.MULTILINE).group(1).strip()
        if (numpy_version, cv2_version) != ('2.2.6', '5.0.0.93'):
            raise ValueError('Bundled dependency versions differ from the pinned probe environment')
        dependencies = {'numpy': dict(numpy_info, version=numpy_version),
                        'opencv-python-headless': dict(cv2_info, version=cv2_version)}
    return modules, dependencies


def covariance_fixture(np, original, path: Path) -> dict:
    rng = np.random.default_rng(20261026)
    tensors, names = [], []
    for scale in [1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
        basis = rng.normal(size=(8, 16, 2, 2)).astype(np.float32)
        j = np.einsum('...ij,...kj->...ik', basis, basis) * np.float32(scale)
        tensors.append(np.stack((j[..., 0, 0], j[..., 0, 1], j[..., 1, 1])))
        names.append('random_psd_' + str(scale))
    for name, xx, xy, yy in [('flat', 0, 0, 0), ('isotropic', 1, 0, 1),
                            ('rank_one_x', 1, 0, 0), ('rank_one_y', 0, 0, 1),
                            ('rank_one_rotated', .5, .5, .5), ('diagonal', 4, 0, 1)]:
        tensors.append(np.tile(np.array([xx, xy, yy], np.float32)[:, None, None], (1, 8, 16)))
        names.append(name)
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False, dtype=np.float32).reshape(8, 16)
    cos, sin = np.cos(angles), np.sin(angles)
    tensors.append(np.stack((4 * cos * cos + sin * sin, 3 * cos * sin, 4 * sin * sin + cos * cos)))
    names.append('rotating_axes')
    tensor = np.stack(tensors)
    parameters = np.array([[.12, 2], [1, 2], [1, 1], [1, 4], [.45, 3]], np.float64)
    covariances = []
    for kd, shrink in parameters:
        covariances.append(np.stack([
            original.cov_i_from_tensor(*j, k_detail=float(kd), k_shrink=float(shrink)).transpose(2, 0, 1)
            for j in tensor]))
    guide = rng.uniform(0, 1, (2, 1, 9, 11)).astype(np.float32)
    guide[1, 0] = np.linspace(0, 1, 11, dtype=np.float32)[None, :]
    structure = np.stack([np.stack(original.structure_tensor(item[0])) for item in guide])
    guide_covariance = np.stack([original.cov_i(item[0]).transpose(2, 0, 1) for item in guide])
    np.savez_compressed(path, tensor=tensor, parameters=parameters, covariance=np.stack(covariances),
                        guide=guide, structure=structure, guide_covariance=guide_covariance)
    return {'seed': 20261026, 'tensor_cases': names, 'parameter_pairs': parameters.tolist(),
            'tensor_count': 1664, 'original_covariance_evaluations': 8320}


def lca_fixture(np, original, path: Path) -> dict:
    data = {}
    rng = np.random.default_rng(1349)
    cases = []
    for h, w in ((13, 16), (9, 11), (1, 5), (5, 1)):
        source = rng.uniform(-.15, 1.1, (h, w, 3)).astype(np.float32)
        for rb in ((0., 0.), (.017, -.023), (.137, -.091)):
            for applied in (False, True):
                for tile in ((0, 0, h, w), (h // 3, w // 3, max(1, h // 2), max(1, w // 2))):
                    index = len(cases)
                    record = original.estimate_radial_lca(source, (1., 1.), mode='off')
                    record['applied'] = applied
                    record['coefficients'] = {'r': rb[0], 'g': 0., 'b': rb[1]}
                    y0, x0, th, tw = tile
                    expected = original.warp_rgb_tile(source, record, y0=y0, x0=x0, height=th, width=tw)
                    data[f'source_{index}'] = source.transpose(2, 0, 1)[None].copy()
                    data[f'expected_{index}'] = expected.transpose(2, 0, 1)[None].copy()
                    data[f'rb_{index}'] = np.array(rb, np.float32)
                    data[f'tile_{index}'] = np.array(tile, np.int64)
                    data[f'applied_{index}'] = np.array(applied)
                    cases.append({'shape': [h, w], 'rb': rb, 'applied': applied, 'tile': tile})
    data['case_count'] = np.array(len(cases))
    np.savez_compressed(path, **data)
    return {'seed': 1349, 'cases': cases}


def frontend_fixture(np, original, affine: Path, path: Path) -> dict:
    rng = np.random.default_rng(825)
    data = {}
    for i, gain in enumerate((0., 1e-4, .2, 1., 7.)):
        phase = rng.uniform(0., 1., (3, 48, 9, 11)).astype(np.float32)
        phase[:, 16:] *= np.float32(gain)
        phase = phase.reshape(144, 9, 11)
        legacy = rng.uniform(0., .5, (9, 11, 3)).astype(np.float32) * np.float32(gain)
        feature, amplitude = original.make_controller_input(phase.copy(), legacy.copy(), affine)
        data[f'phase_{i}'] = phase[None].copy()
        data[f'legacy_{i}'] = legacy.transpose(2, 0, 1)[None].copy()
        data[f'feature_{i}'] = feature[None].copy()
        data[f'amplitude_{i}'] = amplitude[None].copy()
    data['case_count'] = np.array(5)
    np.savez_compressed(path, **data)
    return {'seed': 825, 'gains': [0., 1e-4, .2, 1., 7.], 'cases': 5}


def generate(executable: Path, output: Path, weights_dir: Path) -> dict:
    executable, output, weights_dir = executable.resolve(), output.resolve(), weights_dir.resolve()
    if output.exists():
        raise ValueError('Output directory already exists; select a new directory')
    runtime = check_runtime()
    affine = weights_dir / 'local_v7_static_affine_v1.npz'
    if sha256(affine) != AFFINE_HASH:
        raise ValueError('Published static affine SHA-256 mismatch')
    if any(name in sys.modules for name in MODULE_HASHES):
        raise RuntimeError('Run in a fresh Python process before importing released modules')
    release = extract(executable, output / 'shaders')
    modules, dependencies = extract_modules(executable, output, release)
    # Imports occur only after the complete EXE, all modules, and affine are pinned.
    sys.path.insert(0, str(output / '_modules'))
    np = importlib.import_module('numpy')
    cv2 = importlib.import_module('cv2')
    original = {name: importlib.import_module(name) for name in MODULE_HASHES}
    provenance = {
        'recovered_covariance_oracle.npz': covariance_fixture(np, original['bsr_steer'], output / 'recovered_covariance_oracle.npz'),
        'recovered_lca_warp_oracle.npz': lca_fixture(np, original['v82_lca'], output / 'recovered_lca_warp_oracle.npz'),
        'recovered_frontend_oracle.npz': frontend_fixture(np, original['local_v7_frontend'], affine, output / 'recovered_frontend_oracle.npz'),
    }
    fixtures = {}
    for filename, details in provenance.items():
        path = output / filename
        actual = sha256(path)
        fixtures[filename] = dict(details, bytes=path.stat().st_size, sha256=actual,
                                  expected_sha256=FIXTURE_HASHES[filename],
                                  byte_identical=actual == FIXTURE_HASHES[filename])
    passed = all(value['byte_identical'] for value in fixtures.values())
    manifest = {'status': 'passed' if passed else 'failed', 'release': 'v9.8',
                'release_sha256': release['executable_sha256'], 'runtime': runtime,
                'cv2_version_attribute': cv2.__version__, 'bundled_dependencies': dependencies,
                'modules': modules, 'affine_sha256': AFFINE_HASH, 'fixtures': fixtures,
                'execution': 'Imported hash-verified native modules; did not launch the EXE or GUI.',
                'distribution': 'Only numeric NPZ fixtures are intended for redistribution; _modules contains original native binaries.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if not passed:
        raise RuntimeError(f'Generated fixture bytes differ from the accepted oracle; inspect {output / "manifest.json"}')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New local directory; must not already exist')
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'Core-Only-Source-Code/weights')
    args = parser.parse_args()
    manifest = generate(args.release_exe, args.output, args.weights_dir)
    print(json.dumps({'status': manifest['status'], 'output': str(args.output),
                      'fixtures': {name: value['sha256'] for name, value in manifest['fixtures'].items()}}, indent=2))


if __name__ == '__main__':
    main()
