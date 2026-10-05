"""Reproduce saturation-protocol-probe.json without changing repository code.

Run from the repository root:
python docs/google-target-review/diagnose_burst_support.py

Source: constant linear RGB [3,48,48] at 1.0, Mallett spectral lifting,
Canon 5DMarkII relative SRF, scene gain 1.4, seed43. The original control uses
analytic_profile ISO800 (ADC first), with transmission changed to [.5,1,.25].
Noise and quantization are both disabled. Compare equal exposure against
one-EV bracketing; production generator, integration and baseline are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from jsr_repro.calibration import analytic_profile
from jsr_repro.spectral_data import synthesize_spectral
from jsr_repro.spectral import SpectralAssets
from jsr_repro.train_transformer import batch_sample
from jsr_repro.transformer import SpeechTransformer


def main():
    torch.set_num_threads(1)
    profile = analytic_profile()
    profile['channel_transmission'] = [.5, 1., .25]
    assets = SpectralAssets()
    source = torch.ones(3, 48, 48)
    options = {
        'protocol': 'spectral-camera-v3', 'target_stage': 'post_pixel',
        'native_size': 16, 'frames': 7, 'scale': 2,
        'psf_radius': 4, 'scene_margin_hr': 16, 'max_shift': 0.,
        'pupil_samples': 32, 'fft_size': 128,
        'field_extent': 0., 'field_center': [0., 0.],
        'scene_gain': [1.4, 1.4],
        'defocus_nm': [0., 0.], 'astigmatism_nm': [0., 0.],
        'coma_nm': [0., 0.], 'lca_native': [0., 0.],
        'iso': [800], 'camera_ids': ['Canon 5DMarkII'],
        'noise': False, 'quantize': False, 'augment': False,
    }
    # f-number, pitch and fill-factor intentionally retain the generator's
    # default ranges used by the original control: [2,8], [3,5.76], [.9,1].
    # A flat field is preserved by every normalized PSF and area footprint.
    report = {}
    for label, interval in [('equal', 0.), ('bracketed', 1.)]:
        sample = synthesize_spectral(
            source, dict(options, bracket_interval_ev=[interval, interval]),
            43, profile, assets)
        color_support = {}
        for color, y, x in [('R', 0, 0), ('G1', 0, 1), ('G2', 1, 0), ('B', 1, 1)]:
            good = (sample['valid'] * (1-sample['saturation']))[:, 0, y::2, x::2]
            color_support[color] = {
                'per_frame_valid_fraction': good.flatten(1).mean(1).tolist(),
                'no_support_anywhere_in_burst': bool(not good.any()),
            }
        sample['metadata'] = json.dumps(sample['metadata'])
        model = SpeechTransformer(width=4, heads=1, window=2)
        with torch.no_grad():
            result = model(batch_sample(sample, 'cpu'))
        # Report only baseline/support, neither of which depends on the newly
        # instantiated Transformer's random learned parameters.
        report[label] = {
            'target_min': sample['target'].min().item(),
            'target_max': sample['target'].max().item(),
            'exposure': sample['exposure'].tolist(),
            'true_signal_saturation_fraction': sample['signal_saturation'].mean().item(),
            'support': color_support,
            'merged_green_mean': result['baseline'][:, 1].mean().item(),
            'all_saturated_or_invalid_packed_fraction': result['all_saturated_or_invalid_packed'].mean().item(),
        }
    path = Path(__file__).with_name('saturation-protocol-probe.json')
    if path.exists():
        original = json.loads(path.read_text())
        if original != report:
            raise AssertionError('regenerated control differs from the original recorded evidence')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'evidence': str(path), 'matches_original': True,
                      'control_iso': 800, 'noise': False, 'quantize': False}, indent=2))


if __name__ == '__main__':
    main()
