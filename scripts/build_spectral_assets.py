"""Convert separately downloaded public numeric data without importing source code.

Usage: python scripts/build_spectral_assets.py --source /path/to/public_spectral_sources
Source URLs/hashes and transformation provenance are retained with each output.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((source / 'sources.json').read_text(encoding='utf-8'))
    for name, info in sources.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'source hash mismatch: {name}')
    tree = ast.parse((source / 'mallett2019.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == 'DATA_BASIS_FUNCTIONS_sRGB_MALLETT2019')
    basis = np.asarray(ast.literal_eval(node.value.args[0]), dtype=float)
    assert basis.shape == (81, 3)
    waves = np.arange(400., 701., 5.)
    xyz_source = np.loadtxt(source / 'cie_xyz.csv', delimiter=',')
    d65_source = np.loadtxt(source / 'cie_d65.csv', delimiter=',')
    color = {'schema': 'jsr-spectral-basis-v1', 'wavelengths_nm': waves.tolist(),
             'reflectance_basis': basis[4:65].tolist(),
             'cie_xyz': np.stack([np.interp(waves, xyz_source[:, 0], xyz_source[:, c]) for c in (1,2,3)], 1).tolist(),
             'illuminant_d65': np.interp(waves, d65_source[:, 0], d65_source[:, 1]).tolist(),
             'provenance': {'basis': sources['mallett2019.py'], 'xyz': sources['cie_xyz.csv'],
                            'd65': sources['cie_d65.csv'], 'basis_license': 'BSD-3-Clause',
                            'cie_license': 'CC-BY-SA-4.0',
                            'transform': 'Mallett basis cropped from 380..780/5nm to 400..700/5nm; CIE/D65 sampled at those wavelengths. Round-trip error is measured, not claimed zero.'}}
    lines = [x.strip() for x in (source / 'camspec_database.txt').read_text().splitlines() if x.strip()]
    assert len(lines) == 28 * 4
    cameras = {}
    for i in range(0, len(lines), 4):
        sensitivity = np.asarray([[float(v) for v in line.split()] for line in lines[i+1:i+4]])
        assert sensitivity.shape == (3,33)
        cameras[lines[i]] = np.stack([np.interp(waves, np.arange(400.,721.,10.), row) for row in sensitivity], 1).tolist()
    camera = {'schema': 'jsr-camera-spectra-v1', 'wavelengths_nm': waves.tolist(), 'cameras': cameras,
              'provenance': {'kind': 'published_measured_relative_response', 'source': sources['camspec_database.txt'],
                             'authors': 'Jun Jiang, Dengyu Liu, Jinwei Gu, Sabine Suesstrunk; RIT; WACV 2013',
                             'license': 'CC-BY-NC-SA-4.0',
                             'transform': 'Linear interpolation from 400..720/10nm to 400..700/5nm; original normalization preserved. NOT absolute QE, ISO/PTC, or a complete full-frame library.'}}
    for name, value in [('basis.json', color), ('cameras.json', camera), ('source_manifest.json', sources)]:
        (output / name).write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))
    for name in ('colour-LICENSE.txt', 'camspec-LICENSE.txt', 'cie_xyz_metadata.json', 'cie_d65_metadata.json'):
        shutil.copyfile(source / name, output / name)
    return {'camera_count': len(cameras), 'wavelength_count': len(waves), 'output': str(output)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'reproduction/jsr_repro/spectral_assets')
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output)))
