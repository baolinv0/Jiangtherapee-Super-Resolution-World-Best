"""Saturation-mask diagnostic; writes evidence beside this script.

Run from the repository after installing its reproduction/test dependencies:
python docs/google-target-review/diagnose_full_well_mask.py

The diagnostic truth mask is used for evaluation only. Every guard-band
candidate uses only observed DN and camera calibration, never clean signal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from jsr_repro.calibration import analytic_profile
from jsr_repro.physical_sensor import capture_sensor
from jsr_repro.transformer import FIELDS, SpeechTransformer


def gaussian_cdf(value):
    return .5 * (1 + math.erf(value / math.sqrt(2)))


def green_planes(value):
    return torch.cat((value[:, 0, 0::2, 1::2].reshape(-1),
                      value[:, 0, 1::2, 0::2].reshape(-1)))


def observed_guard_mask(observed_dn, camera, quantize, sigma_multiple):
    gain = float(camera['gain_e_per_dn'])
    ceiling_dn = min(float(camera['white_dn']),
                     float(camera['black_dn']) + float(camera['full_well_e']) / gain)
    read_sigma_dn = float(camera['read_noise_e']) / gain
    half_code = .5 if quantize else 0.
    # Candidate policy only: conservative rejection of measurements near the
    # calibrated ceiling. This deliberately trades usable near-white samples
    # for reduced false acceptance of full-well-clipped measurements.
    threshold = ceiling_dn - half_code - sigma_multiple * read_sigma_dn
    return observed_dn >= threshold, threshold


def main():
    torch.set_num_threads(1)
    profile = analytic_profile()
    size, frames, seed = 128, 7, 1189
    transmission = torch.tensor([.5, 1., .25])
    output = {'seed': seed, 'native_size': size, 'frames': frames,
              'transmission': transmission.tolist(),
              'scope': 'Synthetic calibrated-observation mask diagnostic, not real sensor validation',
              'candidates': [], 'near_ceiling_unsaturated_controls': []}
    for iso in (100, 200):
        camera = profile['iso'][str(iso)]
        gain, read = camera['gain_e_per_dn'], camera['read_noise_e']
        span = camera['white_dn'] - camera['black_dn']
        ceiling_dn = min(camera['white_dn'], camera['black_dn'] + camera['full_well_e'] / gain)
        regime = 'full-well-first' if camera['full_well_e'] < span * gain else 'ADC-first'
        for quantize in (False, True):
            sample = capture_sensor(torch.full((frames, 3, size, size), 1.4),
                                    torch.ones(frames), transmission, camera,
                                    torch.Generator().manual_seed(seed),
                                    noise=True, quantize=quantize)
            truth = green_planes(sample['signal_saturation']).bool()
            base = green_planes(sample['saturation']).bool()
            candidates = []
            for multiple in (0., 2., 3., 4., 5.):
                mask, threshold = observed_guard_mask(sample['observed_dn'], camera, quantize, multiple)
                green_mask = green_planes(mask)
                # Deeply overexposed FW-first pixels have negligible chance of
                # unclipped Poisson counts; their post-FW scatter is read noise.
                if regime == 'full-well-first':
                    boundary = math.ceil(threshold) - .5 if quantize else threshold
                    theory = gaussian_cdf((ceiling_dn - boundary) / (read / gain))
                else:
                    theory = None  # ADC clipping occurs long above its ceiling here.
                candidates.append({'read_sigma_multiple': multiple,
                                   'observed_dn_threshold': threshold,
                                   'green_observed_saturated_fraction': green_mask.float().mean().item(),
                                   'true_clipped_green_reaccepted_fraction': (truth & ~green_mask).float().mean().item(),
                                   'any_green_support_remains_in_complete_input': bool((~green_mask).any()),
                                   'conditional_FW_Gaussian_prediction': theory})
            output['candidates'].append({'iso': iso, 'regime': regime, 'quantize': quantize,
                                         'camera': camera, 'green_truth_saturation_fraction': truth.float().mean().item(),
                                         'base_green_observed_saturation_fraction': base.float().mean().item(),
                                         'green_observed_dn_min': green_planes(sample['observed_dn']).min().item(),
                                         'green_observed_dn_max': green_planes(sample['observed_dn']).max().item(),
                                         'masks': candidates})
    # Quantify the guard band's tradeoff on a signal below full-well: exclude
    # shot noise to isolate how read-noise censoring itself changes the mask.
    # This is NOT the Poisson capture model used in the preceding cases.
    camera = profile['iso']['100']
    ceiling_dn = camera['black_dn'] + camera['full_well_e'] / camera['gain_e_per_dn']
    read_sigma_dn = camera['read_noise_e'] / camera['gain_e_per_dn']
    read_sample = torch.randn(200000, generator=torch.Generator().manual_seed(seed)) * read_sigma_dn
    for distance_sigma in (2., 4., 8.):
        observed = (ceiling_dn - distance_sigma * read_sigma_dn + read_sample).round()
        row = {'clean_signal_below_ceiling_read_sigma': distance_sigma}
        for multiple in (0., 3., 5.):
            mask, _ = observed_guard_mask(observed, camera, True, multiple)
            row[f'read_sigma_guard_{multiple:g}_rejected_fraction'] = mask.float().mean().item()
        output['near_ceiling_unsaturated_controls'].append(row)
    # Show the downstream implication on the unchanged Transformer baseline:
    # clipped values below the threshold are accepted as linear measurements.
    camera = profile['iso']['100']
    sample = capture_sensor(torch.full((7, 3, 32, 32), 1.4),
                            torch.ones(7), transmission, camera,
                            torch.Generator().manual_seed(seed),
                            noise=True, quantize=True)
    sample['shifts'] = torch.zeros(7, 2)
    model = SpeechTransformer(width=4, heads=1, window=2)
    with torch.no_grad():
        result = model({key: sample[key][None] for key in FIELDS})
    output['downstream_original_baseline'] = {
        'native_size': 32,
        'green_truth_saturated_fraction': green_planes(sample['signal_saturation']).mean().item(),
        'all_saturated_or_invalid_packed_fraction': result['all_saturated_or_invalid_packed'].mean().item(),
        'packed_green_positions_with_any_observed_valid_frame_fraction': 1-result['all_saturated_or_invalid_packed'][:, 1:3].mean().item(),
        'baseline_green_mean': result['baseline'][:, 1].mean().item(),
        'target_green': 1.4,
    }
    path = Path(__file__).with_name('full_well_mask_independent.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'evidence': str(path),
                      'base_masks': [{key: row[key] for key in ('iso', 'regime', 'quantize', 'green_truth_saturation_fraction', 'base_green_observed_saturation_fraction')}
                                     for row in output['candidates']],
                      'guard_summary': [{'iso': row['iso'], 'quantize': row['quantize'],
                                         '3_sigma': row['masks'][2]['green_observed_saturated_fraction'],
                                         '5_sigma': row['masks'][4]['green_observed_saturated_fraction']}
                                        for row in output['candidates']]}, indent=2))


if __name__ == '__main__':
    main()
