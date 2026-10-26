"""Portable original-weight bundles retain exact tensors and K alias semantics."""
import json
from pathlib import Path
import shutil

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'Core-Only-Source-Code/weights'


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    from jsr_repro.network_bundle import export_pretrained_bundle
    torch.set_num_threads(2)
    output = tmp_path_factory.mktemp('network-bundle') / 'pretrained'
    manifest = export_pretrained_bundle(WEIGHTS, output, ks=[4, 13, 14])
    return output, manifest


def test_bundle_has_exact_original_k4_parameters_buffers_and_forward(bundle):
    from jsr_repro.network_bundle import load_pretrained_bundle
    from jsr_repro.recovered_model import load_recovered_jsr
    path, _ = bundle
    actual, expected = load_pretrained_bundle(path, 4), load_recovered_jsr(WEIGHTS, 4)
    assert set(actual.state_dict()) == set(expected.state_dict())
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    frames = torch.linspace(.01, .3, 4 * 6 * 8).reshape(1, 4, 6, 8)
    flow = torch.zeros(1, 4, 6, 8, 2)
    covariance = torch.tensor([1., .1, 1.1]).reshape(1, 3, 1, 1).expand(1, 3, 8, 12)
    with torch.no_grad():
        a, e = actual(frames, flow, covariance), expected(frames, flow, covariance)
    for name in e:
        torch.testing.assert_close(a[name], e[name], rtol=0, atol=0)


def test_deduplicated_k13_route_preserves_requested_frame_count(bundle):
    from jsr_repro.network_bundle import load_pretrained_bundle
    path, manifest = bundle
    assert manifest['routes']['13'] == manifest['routes']['14']
    model = load_pretrained_bundle(path, 13)
    assert model.expected_frames == 13
    with pytest.raises(ValueError, match='K=13'):
        model(torch.ones(1, 14, 4, 4), torch.zeros(1, 14, 4, 4, 2), torch.ones(1, 3, 8, 8))


def test_bundle_rejects_corrupted_component_before_loading(bundle, tmp_path):
    from jsr_repro.network_bundle import load_pretrained_bundle
    path, manifest = bundle
    copy = tmp_path / 'corrupted'
    shutil.copytree(path, copy)
    file = copy / manifest['static_affine']
    file.write_bytes(file.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='checksum|size'):
        load_pretrained_bundle(copy, 4)


def test_bundle_rejects_unexported_k_and_overwrite(bundle):
    from jsr_repro.network_bundle import export_pretrained_bundle, load_pretrained_bundle
    path, _ = bundle
    with pytest.raises(ValueError, match='route'):
        load_pretrained_bundle(path, 1)
    with pytest.raises(ValueError, match='exists'):
        export_pretrained_bundle(WEIGHTS, path, ks=[4])


def test_bundle_manifest_cannot_escape_component_directory(bundle, tmp_path):
    from jsr_repro.network_bundle import load_pretrained_bundle
    path, manifest = bundle
    forged = tmp_path / 'forged'
    forged.mkdir()
    data = json.loads(json.dumps(manifest))
    data['static_affine'] = '../outside.pt'
    (forged / 'manifest.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='path|component'):
        load_pretrained_bundle(forged, 4)
