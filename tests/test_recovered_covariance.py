"""Frozen numeric probes of the official v9.8 covariance extension.

The fixture contains numbers only. Original binaries are not needed for tests.
"""
from pathlib import Path

import numpy as np
import pytest
import torch


ORACLE = Path(__file__).parent / 'data' / 'recovered_covariance_oracle.npz'


@pytest.mark.parametrize('parameter_index', range(5))
def test_covariance_matches_released_module_psd_rotations_and_degeneracies(parameter_index):
    from jsr_repro.recovered_covariance import covariance_from_tensor
    with np.load(ORACLE, allow_pickle=False) as fixture:
        tensor = torch.from_numpy(fixture['tensor'])
        kd, shrink = fixture['parameters'][parameter_index]
        expected = torch.from_numpy(fixture['covariance'][parameter_index])
    actual = covariance_from_tensor(tensor, k_detail=float(kd), k_shrink=float(shrink))
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-6)


def test_structure_tensor_matches_released_sobel_and_box_filter():
    from jsr_repro.recovered_covariance import structure_tensor
    with np.load(ORACLE, allow_pickle=False) as fixture:
        guide = torch.from_numpy(fixture['guide'])
        expected = torch.from_numpy(fixture['structure'])
    torch.testing.assert_close(structure_tensor(guide), expected, atol=2e-6, rtol=2e-6)


def test_complete_guide_stage_matches_released_covariance_and_has_finite_gradients():
    from jsr_repro.recovered_covariance import steering_covariance
    with np.load(ORACLE, allow_pickle=False) as fixture:
        guide = torch.from_numpy(fixture['guide']).requires_grad_()
        expected = torch.from_numpy(fixture['guide_covariance'])
    actual = steering_covariance(guide)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-6)
    actual.square().mean().backward()
    assert guide.grad is not None and torch.isfinite(guide.grad).all()
    assert guide.grad.abs().sum() > 0


def test_flat_and_one_pixel_guides_preserve_released_degenerate_orientation():
    from jsr_repro.recovered_covariance import steering_covariance
    guide = torch.zeros(1, 1, 1, 1, requires_grad=True)
    actual = steering_covariance(guide)
    expected = torch.tensor([5.4774250984, 0., 24.9999980927]).view(1, 3, 1, 1)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    actual.sum().backward()
    assert torch.isfinite(guide.grad).all()


def test_covariance_rejects_unsupported_release_parameters_and_invalid_tensors():
    from jsr_repro.recovered_covariance import covariance_from_tensor, structure_tensor
    value = torch.zeros(1, 3, 2, 2)
    for parameters in ({'min_k': .21}, {'k_stretch': 3.}, {'k_detail': 0.}, {'k_shrink': float('nan')}):
        with pytest.raises(ValueError):
            covariance_from_tensor(value, **parameters)
    with pytest.raises(ValueError):
        covariance_from_tensor(torch.zeros(1, 2, 2, 2))
    with pytest.raises(ValueError):
        covariance_from_tensor(torch.full_like(value, float('nan')))
    with pytest.raises(ValueError):
        structure_tensor(torch.zeros(1, 3, 2, 2))
