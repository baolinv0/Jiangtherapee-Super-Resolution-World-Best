import json

import numpy as np
import pytest
import torch

from jsr_repro.calibrate_stacks import calibrate
from jsr_repro.calibration import fit_profile
from jsr_repro.spectral_data import synthesize_spectral


def test_pair_ptc_fit_and_bound_empirical_read_bank(tmp_path):
    # Precisely known pair-difference variance with stationary nonuniformity.
    rng = np.random.default_rng(61)
    h=w=48
    pattern = rng.normal(size=(h,w))
    pattern = (pattern-pattern.mean())/pattern.std(ddof=1)
    levels = [1000.,2000.,4000.,7000.,11000.]
    pairs = []
    for mean in levels:
        residual = pattern*np.sqrt((mean/2+2.25+1/12)/2)
        pairs.append(np.stack([512+mean+residual,512+mean-residual]))
    dark = 512+rng.normal(0,np.sqrt(2.25+1/12),(128,h,w))
    np.save(tmp_path/'flats.npy',np.stack(pairs))
    np.save(tmp_path/'dark.npy',dark)
    description = {'name':'synthetic-calibration-test','provenance':'known numerical fixture, NOT measured',
                   'channel_transmission':[.7,1,.55],
                   'captures':[{'iso':100,'flat_pairs':'flats.npy','dark':'dark.npy','white_dn':16383,'full_well_e':60000}]}
    (tmp_path/'input.json').write_text(json.dumps(description))
    fitted = calibrate(tmp_path/'input.json',tmp_path/'calibrated')
    assert fitted['iso']['100']['gain_e_per_dn']==pytest.approx(2.,abs=1e-4)
    assert fitted['iso']['100']['read_noise_e']==pytest.approx(3.,abs=.04)
    assert fitted['provenance']['kind']=='fitted_input'
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        options={'native_size':16,'psf_radius':4,'scene_margin_hr':16,'max_shift':0,
                 'pupil_samples':32,'fft_size':128,'field_extent':0.,'iso':[100],
                 'read_noise_bank':str(tmp_path/'calibrated/read-bank-iso100.npz')}
        sample=synthesize_spectral(torch.ones(3,48,48)*.1,options,43,profile=fitted)
        assert torch.isfinite(sample['variance']).all()
        wrong=json.loads(json.dumps(fitted))
        wrong['iso']['100']['read_noise_e']+=1
        with pytest.raises(ValueError,match='bound'):
            synthesize_spectral(torch.ones(3,48,48)*.1,options,43,profile=wrong)
    finally:
        torch.set_num_threads(previous)


def test_random_poisson_flat_pairs_use_measured_dark_intercept(tmp_path):
    """Realistic finite sampling can make the extrapolated intercept negative.

    These are independently sampled Poisson/Gaussian captures, unlike the
    exact-variance fixture above. ADC rounding is included in both stacks.
    """
    rng = np.random.default_rng(1)
    levels = [1000., 2000., 4000., 7000., 11000.]
    pairs = np.stack([
        np.rint(512 + rng.poisson(mean * 2, (2, 48, 48)) / 2
                + rng.normal(0, 1.5, (2, 48, 48)))
        for mean in levels
    ])
    dark = np.rint(512 + rng.normal(0, 1.5, (128, 48, 48)))
    means = pairs.mean((1, 2, 3)) - dark.mean()
    variances = (pairs[:, 0] - pairs[:, 1]).var((1, 2), ddof=1) / 2
    assert np.polyfit(means, variances, 1)[1] < 0  # Previously rejected.
    np.save(tmp_path / 'flats.npy', pairs)
    np.save(tmp_path / 'dark.npy', dark)
    manifest = {
        'name': 'independent-random-numerical-fixture',
        'provenance': 'synthetic independent Poisson/read/quantization draws; not measured',
        'channel_transmission': [.7, 1., .55],
        'captures': [{'iso': 100, 'flat_pairs': 'flats.npy', 'dark': 'dark.npy',
                      'white_dn': 16383, 'full_well_e': 60000}],
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    result = calibrate(tmp_path / 'manifest.json', tmp_path / 'calibration')
    sensor = result['iso']['100']
    diagnostic = result['fit_diagnostics']['100']
    assert sensor['gain_e_per_dn'] == pytest.approx(2., rel=.04)
    assert sensor['read_noise_e'] == pytest.approx(3., rel=.04)
    assert diagnostic['method'] == 'dark_fixed_intercept_feasible_wls'
    assert diagnostic['unconstrained_intercept_dn2'] < 0
    assert diagnostic['intercept_dn2'] == pytest.approx(dark.var(0, ddof=1).mean())
    assert diagnostic['rmse_dn2'] > 0
    assert 0 < diagnostic['relative_rmse'] < .1


def test_without_dark_negative_ptc_intercept_is_still_rejected(tmp_path):
    csv = tmp_path / 'ptc.csv'
    csv.write_text('iso,mean_dn,variance_dn2\n100,10,4\n100,100,49\n100,1000,499\n')
    with pytest.raises(ValueError, match='nonnegative intercept'):
        fit_profile(csv, tmp_path / 'profile.json')


def test_dark_quantization_floor_does_not_create_negative_read_noise(tmp_path):
    dark = tmp_path / 'dark.npy'
    np.save(dark, np.full((4, 8, 8), 512., np.float32))
    csv = tmp_path / 'ptc.csv'
    csv.write_text('iso,mean_dn,variance_dn2\n' + ''.join(
        f'100,{m},{m/2+1/12}\n' for m in (10, 100, 1000)))
    result = fit_profile(csv, tmp_path / 'profile.json', dark=dark)
    assert result['iso']['100']['gain_e_per_dn'] == pytest.approx(2.)
    assert result['iso']['100']['read_noise_e'] == 0
    assert result['fit_diagnostics']['100']['quantization_floor_applied']
