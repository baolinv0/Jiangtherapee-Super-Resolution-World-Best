"""Fit PTC/black/read profiles from user-supplied repeated RAW-DN captures.

Manifest JSON: {name, provenance, channel_transmission:[r,g,b], captures:[
{iso, flat_pairs:'Lx2xHxW.npy', dark:'TxHxW.npy', white_dn, full_well_e}]}.
Dark and flats must share ISO/readout/temperature. Flat pairs cancel stationary
illumination/PRNU in variance estimates. Units stay DN until fitted conversion.
The linear unsaturated interval is limited to 5..70% of the black-white range.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .calibration import fit_profile, profile_identity, validate_profile
from .utils import save_json, sha256


def calibrate(manifest,output):
    manifest,output = Path(manifest).resolve(),Path(output).resolve()
    description = json.loads(manifest.read_text(encoding='utf-8'))
    if not description.get('provenance') or not description.get('captures'):
        raise ValueError('calibration manifest needs provenance and capture records')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('calibration output must be empty')
    profile = {'schema':'jsr-camera-v1','name':description.get('name','fitted-raw-stacks'),
               'provenance':{'kind':'fitted_input','manifest_sha256':sha256(manifest),
                             'input_provenance':description['provenance'],
                             'note':'Capture provenance is supplied by the user; not independently certified or the JSR author library.'},
               'channel_transmission':description['channel_transmission'],'iso':{},'fit_diagnostics':{}}
    banks = []
    for capture in description['captures']:
        iso = int(capture['iso'])
        if str(iso) in profile['iso']:
            raise ValueError('one capture record per ISO required')
        flat_path,dark_path = [manifest.parent/Path(capture[k]) for k in ('flat_pairs','dark')]
        flats = np.load(flat_path,allow_pickle=False).astype(np.float64)
        dark = np.load(dark_path,allow_pickle=False).astype(np.float64)
        if flats.ndim!=4 or flats.shape[1]!=2 or dark.ndim!=3 or dark.shape[0]<3 or flats.shape[2:]!=dark.shape[1:] or not np.isfinite(flats).all() or not np.isfinite(dark).all():
            raise ValueError('flats[L,2,H,W] and dark[T>=3,H,W] must be finite and share dimensions')
        black = float(dark.mean())
        white = float(capture['white_dn'])
        if white<=black:
            raise ValueError('white_dn must exceed dark mean')
        means = flats.mean((1,2,3))-black
        variance = (flats[:,0]-flats[:,1]).var((1,2),ddof=1)/2
        mask = (means>=(white-black)*.05)&(means<=(white-black)*.70)&(flats.max((1,2,3))<white)
        if mask.sum()<3:
            raise ValueError('need >=3 unsaturated flat levels inside 5..70% reference range')
        csv_path = output/f'ptc-iso{iso}.csv'
        with csv_path.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['iso','mean_dn','variance_dn2'])
            writer.writerows((iso,float(m),float(v)) for m,v in zip(means[mask],variance[mask]))
        fitted = fit_profile(csv_path,output/f'fit-iso{iso}.json',dark=dark_path,black_dn=black,
                             white_dn=white,full_well_e=float(capture['full_well_e']),
                             transmission=description['channel_transmission'])
        profile['iso'][str(iso)] = fitted['iso'][str(iso)]
        profile['fit_diagnostics'][str(iso)] = dict(fitted['fit_diagnostics'][str(iso)],
             flat_sha256=sha256(flat_path),dark_sha256=sha256(dark_path),
             accepted_levels=int(mask.sum()),rejected_levels=int((~mask).sum()),
             full_well_source=capture.get('full_well_source','user-specified; not estimated from the linear PTC'))
        residual = dark-dark.mean(0,keepdims=True)
        # Correct mean-estimation loss and approximately remove ADC variance
        # before this observed residual is re-quantized in the simulator.
        residual *= np.sqrt(len(dark)/(len(dark)-1))
        var = residual.var(0)
        residual *= np.sqrt(np.maximum(var-1/12,0)/np.maximum(var,1e-12))[None]
        banks.append((iso,(residual*profile['iso'][str(iso)]['gain_e_per_dn']).astype(np.float32)))
    validate_profile(profile)
    save_json(output/'profile.json',profile)
    for iso,bank in banks:
        metadata = {'schema':'jsr-read-bank-v1','iso':iso,'profile_sha256':profile_identity(profile),
                    'unit':'electron','provenance':description['provenance'],
                    'processing':'temporal mean removed; finite-stack correction; approximate ADC variance removal'}
        np.savez_compressed(output/f'read-bank-iso{iso}.npz',read_noise_e=bank,metadata=np.asarray(json.dumps(metadata)))
    return profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    result = calibrate(args.manifest,args.output)
    print(json.dumps({'profile':str(Path(args.output)/'profile.json'),'iso':list(result['iso'])}))


if __name__=='__main__':
    main()
