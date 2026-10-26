"""Frozen original-extension probes, including clipping and global tile origins."""
from pathlib import Path

import numpy as np
import pytest
import torch

ORACLE = Path(__file__).parent / 'data/recovered_lca_warp_oracle.npz'


def test_fixed_radial_lca_matches_released_warp_for_all_boundary_probes():
    from jsr_repro.recovered_lca import warp_rgb_tile
    with np.load(ORACLE, allow_pickle=False) as fixture:
        for i in range(int(fixture['case_count'])):
            source = torch.from_numpy(fixture[f'source_{i}'])
            rb = torch.from_numpy(fixture[f'rb_{i}'])
            y0, x0, height, width = map(int, fixture[f'tile_{i}'])
            result = warp_rgb_tile(source, rb, y0=y0, x0=x0, height=height, width=width,
                                   applied=bool(fixture[f'applied_{i}']))
            torch.testing.assert_close(result, torch.from_numpy(fixture[f'expected_{i}']), atol=2e-6, rtol=2e-6,
                                       msg=lambda message: f'case {i}: {message}')


def test_fixed_lca_keeps_pixel_gradients_and_validates_bounds():
    from jsr_repro.recovered_lca import warp_rgb_tile
    source = torch.ones(1, 3, 7, 9, requires_grad=True)
    result = warp_rgb_tile(source, [.02, -.01], y0=1, x0=2, height=4, width=5)
    result.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all() and source.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        warp_rgb_tile(source, [0., 0.], x0=8, width=2)
