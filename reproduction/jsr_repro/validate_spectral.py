"""Offline construction validation, sample export and short training evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .bracket_data import save_burst
from .data import linear_to_srgb
from .evaluate_transformer import evaluate
from .infer_transformer import infer
from .prepare import build_manifest, procedural_image
from .spectral_data import SpectralBurstDataset, synthesize_spectral, PROTOCOL
from .train_transformer import run_training
from .utils import load_config, save_json, seed_all


def _preview(rgb,white=4):
    encoded = linear_to_srgb(rgb/white).permute(1,2,0).numpy()
    return Image.fromarray(np.round(encoded.clip(0,1)*255).astype(np.uint8))


def validate(output,config='configs/spectral_smoke.yaml'):
    seed_all(1234,2)
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('use a new validation output directory')
    root.mkdir(parents=True,exist_ok=True)
    manifest = build_manifest(None,root/'fixtures',procedural=12)
    cfg = load_config(config)
    if cfg['data']['options'].get('psf_library'):
        raise ValueError('this optical-intervention validation requires generated PSFs; validate a fixed external PSF with its own calibration protocol')
    cfg['data']['manifest'],cfg['output'] = str(manifest),str(root/'training')
    training = run_training(cfg)
    checkpoint = root/'training/last.pt'
    evaluations = {mode:evaluate(checkpoint,manifest,root/f'evaluation_{mode}.json',alignment=mode)
                   for mode in ('oracle','estimated')}
    dataset = SpectralBurstDataset(manifest,'test',cfg['data']['options'],cfg['seed']+2,cfg['data'].get('profile'))
    profile,assets = dataset.profile,dataset.assets
    sample = dataset[0]
    save_burst(root/'burst.npz',sample)
    # Full training pair includes GT and diagnostic tensors, not model inputs.
    np.savez_compressed(root/'training_pair.npz',**{k:v.numpy() for k,v in sample.items() if isinstance(v,torch.Tensor)},
                        metadata=np.asarray(sample['metadata']))
    inference = infer(checkpoint,root/'burst.npz',root/'inference',alignment='provided',radiance_white=4.)
    if not inference['calibration_domain_match']:
        raise AssertionError('validation inference did not use the training calibration')
    source = torch.from_numpy(procedural_image(78,160)).permute(2,0,1)
    options = dict(cfg['data']['options'],augment=False)
    base = synthesize_spectral(source,options,41,profile,assets)
    interventions = {}
    for name,value in [('f_number',[8.,8.]),('pitch_um',[3.,3.]),('fill_factor',[.8,.8]),('psf_radius',6),('field_center',[.6,.5])]:
        changed = synthesize_spectral(source,dict(options,**{name:value}),41,profile,assets)
        interventions[name] = {'gt_max_abs':(base['target']-changed['target']).abs().max().item(),
                               'raw_mean_abs_change':(base['raw']-changed['raw']).abs().mean().item()}
        if interventions[name]['gt_max_abs']>1e-6:
            raise AssertionError(f'GT changed under {name}')
    assets_report = assets.roundtrip_report()
    if assets_report['max_abs_linear_srgb']>.005:
        raise AssertionError('spectral round-trip error exceeded documented truncation tolerance')
    # Deliberately cover endpoints and each ISO/selected CMF, not random claims.
    coverage = []
    cameras = options.get('camera_ids',list(assets.cameras))
    isos = options.get('iso',[100,200,400,800])
    for i in range(max(len(isos),len(cameras))):
        iso,camera = isos[i%len(isos)],cameras[i%len(cameras)]
        opts = dict(options,iso=[iso],camera_ids=[camera],f_number=[2.,2.] if i%2==0 else [8.,8.],
                    pitch_um=[3.,3.] if i<2 else [5.76,5.76],scene_gain=[2.,2.],noise=False,quantize=False)
        probe = synthesize_spectral(torch.ones(3,128,128),opts,240+i,profile,assets)
        metadata = probe['metadata']
        coverage.append({k:metadata[k] for k in ('iso','camera_response_id','f_number','pitch_um','signal_saturation_fraction')})
    # The same scene is evaluated independently of its target; observed masks
    # are visualized, and no display gain is fitted per model.
    tiles = [('CLEAR GT / white=4',_preview(sample['target'])),
             ('Reference RAW / white=1',_preview(sample['raw'][0].expand(3,-1,-1),1.)),
             ('Longest RAW / white=1',_preview(sample['raw'][-1].expand(3,-1,-1),1.)),
             ('Observed saturation',_preview(sample['saturation'][-1].expand(3,-1,-1),1.)),
             ('Hypothetical spectrum RGB',_preview(base['target'])),
             ('Trained output / white=4',Image.open(root/'inference/preview.png').copy())]
    sheet = Image.new('RGB',(3*260,2*282),'#f5f3ec')
    draw = ImageDraw.Draw(sheet)
    for i,(label,img) in enumerate(tiles):
        x,y=(i%3)*260,(i//3)*282
        draw.text((x+8,y+6),label,fill='black')
        sheet.paste(img.resize((244,244),Image.Resampling.NEAREST),(x+8,y+28))
    sheet.save(root/'construction.png')
    report = {'protocol':PROTOCOL,'training':training,'asset_identity':assets.identity,
              'bundled_camera_shapes':len(assets.cameras),'wavelength_count':len(assets.wavelengths),
              'spectral_roundtrip':assets_report,'gt_optical_interventions':interventions,
              'explicit_endpoint_coverage':coverage,'inference':inference,
              'evaluation':{k:v['mean_per_image_metrics'] for k,v in evaluations.items()},
              'scope':'Procedural fixtures and short CPU optimization only. Default PTC and optical aberrations are assumptions; camera spectral shapes are public measurements. No original-author quality claim.',
              'sample_metadata':json.loads(sample['metadata'])}
    save_json(root/'validation.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default='configs/spectral_smoke.yaml')
    args=parser.parse_args()
    result=validate(args.output,args.config)
    print(json.dumps({'protocol':result['protocol'],'steps':result['training']['steps_completed'],
                      'camera_shapes':result['bundled_camera_shapes'],'wavelengths':result['wavelength_count']}))
