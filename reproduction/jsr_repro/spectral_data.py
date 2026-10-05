"""Complete, explicitly assumed spectral-camera RAW burst construction.

Clean GT and every input share camera spectral coordinates and reference
exposure. v3 defaults to reference optics and native pixel-area integration
sampled on a dense 2x grid. GT never includes capture noise or clipping.
The legacy v2 target remains before optics. Scene/optics/noise use independent
deterministic RNGs, with the v2 namespace retained for paired interventions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .bracket_data import bracket_exposures
from .calibration import load_profile, profile_identity
from .data import load_rgb, read_manifest, verify_manifest, _uniform
from .optics import sensor_integrate
from .physical_optics import prepare_spectral_kernels, apply_spectral_kernels, load_psf_library
from .physical_sensor import capture_sensor
from .spectral import SpectralAssets, load_spectral_scene
from .utils import save_json, sha256, load_config

PROTOCOL = 'spectral-camera-v3'
LEGACY_PROTOCOL = 'spectral-camera-v2'
DEFAULT_CAMERAS = ['Canon 5DMarkII','Nikon D3X','Nikon D3','Nikon D700']


def resolve_target_stage(options):
    """Validate the protocol and resolve its default reconstruction target."""
    protocol = options.get('protocol', PROTOCOL)
    if protocol not in (PROTOCOL, LEGACY_PROTOCOL):
        raise ValueError(f'unknown spectral data protocol: {protocol}')
    stage = options.get('target_stage', 'pre_optics' if protocol == LEGACY_PROTOCOL else 'post_pixel')
    if stage not in ('pre_optics', 'post_optics', 'post_pixel'):
        raise ValueError('target_stage must be pre_optics, post_optics or post_pixel')
    if protocol == LEGACY_PROTOCOL and stage != 'pre_optics':
        raise ValueError('spectral-camera-v2 requires target_stage=pre_optics')
    return stage


def _rng(seed, name):
    # Protocol/target comparisons must share the original scene and RAW burst.
    digest = hashlib.sha256(f'{LEGACY_PROTOCOL}/{seed}/{name}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8],'little') % (2**63-1))


def _range(options,key,default,low,high):
    value = options.get(key,default)
    if not isinstance(value,(list,tuple)) or len(value)!=2 or not all(math.isfinite(float(x)) for x in value) or not low<=value[0]<=value[1]<=high:
        raise ValueError(f'{key} requires finite ordered bounds within [{low},{high}]')
    return value


def _warp(spectrum,shift,scale):
    h,w = spectrum.shape[-2:]
    yy,xx = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    coords = torch.stack((xx,yy),-1).float()+shift*scale
    grid = 2*(coords+.5)/torch.tensor([w,h]) - 1
    return F.grid_sample(spectrum[None],grid[None],mode='bilinear',padding_mode='border',align_corners=False)[0]


def synthesize_spectral(source,options,seed,profile=None,assets=None):
    target_stage = resolve_target_stage(options)
    protocol = options.get('protocol', PROTOCOL)
    assets = SpectralAssets() if assets is None else assets
    profile = load_profile() if profile is None else profile
    size = int(options.get('native_size',16))
    if size<16 or size%2 or int(options.get('scale',2))!=2 or int(options.get('frames',7))!=7:
        raise ValueError('spectral Transformer path requires even native_size>=16, scale2 and K7')
    if source.ndim!=3 or source.shape[0] not in (3,61) or not torch.isfinite(source).all() or (source<0).any():
        raise ValueError('source must be nonnegative RGB[3,H,W] or radiance[61,H,W]')
    scene_rng,optics_rng,noise_rng = (_rng(seed,n) for n in ('scene','optics','noise'))
    radius = int(options.get('psf_radius',12))
    max_shift = float(options.get('max_shift',2.))
    if not 2<=radius<=48 or not math.isfinite(max_shift) or not 0<=max_shift<=4:
        raise ValueError('PSF radius2..48 and max_shift0..4 required')
    # Fixed source margin independent of the sampled PSF and its radius.
    # This makes the source crop invariant under optics interventions.
    margin = int(options.get('scene_margin_hr',56))
    if margin < radius+math.ceil(2*max_shift)+2:
        raise ValueError('scene_margin_hr must cover PSF and motion')
    hr = size*2
    resized = min(source.shape[-2:]) < hr
    if resized:
        factor = hr/min(source.shape[-2:])
        source = F.interpolate(source[None],size=[math.ceil(d*factor) for d in source.shape[-2:]],mode='bilinear',align_corners=False)[0]
    oy = int(torch.randint(source.shape[-2]-hr+1,(),generator=scene_rng))
    ox = int(torch.randint(source.shape[-1]-hr+1,(),generator=scene_rng))
    # Pad source only to obtain out-of-scene context; report this explicitly.
    # Replicate padding supports tiny fixtures without arbitrary hidden resize.
    padded = F.pad(source[None],(margin,)*4,mode='replicate')[0]
    scene = padded[:,oy:oy+hr+2*margin,ox:ox+hr+2*margin].clone()
    padded_context = oy<margin or ox<margin or oy+hr+margin>source.shape[-2] or ox+hr+margin>source.shape[-1]
    if options.get('augment',True):
        scene = torch.rot90(scene,int(torch.randint(4,(),generator=scene_rng)),(-2,-1))
        if torch.rand((),generator=scene_rng)>.5:
            scene = scene.flip(-1)
    gain_range = _range(options,'scene_gain',[.125,8.],1e-6,1e4)
    gain = 2**_uniform(scene_rng,*[math.log2(x) for x in gain_range])
    camera_ids = options.get('camera_ids',DEFAULT_CAMERAS)
    if not camera_ids or any(x not in assets.cameras for x in camera_ids):
        raise ValueError('camera_ids must select entries from bundled measured spectral curves')
    camera_id = camera_ids[int(torch.randint(len(camera_ids),(),generator=scene_rng))]
    spectrum = assets.lift(scene) if scene.shape[0]==3 else scene
    spectrum = spectrum*gain
    if target_stage == 'pre_optics':
        clear_rgb = assets.camera_rgb(spectrum,camera_id)
        target = clear_rgb[:,margin:margin+hr,margin:margin+hr].clone()
    shifts = (torch.rand(7,2,generator=scene_rng)*2-1)*max_shift
    shifts[0] = 0
    interval = _uniform(scene_rng,*_range(options,'bracket_interval_ev',[0.,1.],0.,1.))
    exposure = bracket_exposures(interval)
    f_number = _uniform(optics_rng,*_range(options,'f_number',[2.,8.],2.,8.))
    pitch = _uniform(optics_rng,*_range(options,'pitch_um',[3.,5.76],3.,5.76))
    fill = _uniform(optics_rng,*_range(options,'fill_factor',[.9,1.],.8,1.))
    field = options.get('field_center')
    field = (torch.rand(2,generator=optics_rng)*1.6-.8).tolist() if field is None else list(field)
    if len(field)!=2 or not all(math.isfinite(float(x)) and abs(x)<=1 for x in field):
        raise ValueError('field_center must be two finite normalized coordinates in [-1,1]')
    extent = float(options.get('field_extent',.1))
    aberrations = {name:_uniform(optics_rng,*_range(options,name+'_nm',default,-500.,500.))
                   for name,default in [('defocus',[-20.,20.]),('astigmatism',[0.,60.]),('coma',[0.,80.])]}
    if 'spherical_nm' in options:
        aberrations['spherical'] = _uniform(optics_rng,*_range(options,'spherical_nm',[0.,0.],-500.,500.))
    lca = _uniform(optics_rng,*_range(options,'lca_native',[0.,.5],0.,2.))
    pupil_samples,fft_size = int(options.get('pupil_samples',64)),int(options.get('fft_size',256))
    psf_provenance = 'assumed circular-pupil scalar diffraction with parametric OPD; not measured prime-lens library'
    if options.get('psf_library'):
        library = load_psf_library(options['psf_library'],assets.wavelengths,pitch)
        if not math.isclose(float(library['metadata'].get('f_number',float('nan'))),f_number,rel_tol=1e-6):
            raise ValueError('external PSF f_number must match the fixed configured f_number')
        expected_fields = torch.tensor([field] if extent==0 else [[field[0]+dx*extent,field[1]+dy*extent] for dy in (-1,1) for dx in (-1,1)])
        if library['field_xy'].shape!=expected_fields.shape or not torch.allclose(library['field_xy'],expected_fields,atol=1e-6,rtol=0):
            raise ValueError('external PSF nodes must match configured center/corners in TL,TR,BL,BR order')
        kernels = library['kernels']
        if kernels.shape[-1]!=2*radius+1:
            raise ValueError('external PSF support must match psf_radius')
        psf_provenance = library['metadata']['provenance']
    else:
        kernels = prepare_spectral_kernels(assets.wavelengths,f_number,pitch,radius=radius,
                      field_center=field,field_extent=extent,aberrations_nm=aberrations,lca_native=lca,
                      pupil_samples=pupil_samples,fft_size=fft_size)
    native_frames = []
    for frame, shift in enumerate(shifts):
        moved = _warp(spectrum,shift,2)
        blurred = apply_spectral_kernels(moved,kernels)
        sensor_rgb = assets.camera_rgb(blurred,camera_id)
        native_frames.append(sensor_integrate(sensor_rgb,torch.zeros(1,2),size,margin,fill)[0])
        # Frame zero is the unshifted reference. Reuse its actual spectral PSF
        # and SRF result; target construction must not redraw or reblur optics.
        if frame == 0:
            if target_stage == 'post_optics':
                target = sensor_rgb[:,margin:margin+hr,margin:margin+hr].clone()
            elif target_stage == 'post_pixel':
                target = sensor_integrate(sensor_rgb,torch.zeros(1,2),size,margin,fill,
                                          output_scale=2)[0]
    native = torch.stack(native_frames)
    choices = [str(int(x)) for x in options.get('iso',[100,200,400,800])]
    if not choices or any(x not in profile['iso'] for x in choices):
        raise ValueError('every requested ISO needs a supplied camera profile entry')
    iso = choices[int(torch.randint(len(choices),(),generator=noise_rng))]
    transmission = torch.tensor(profile['channel_transmission'],dtype=torch.float32)
    bank = None
    if options.get('read_noise_bank'):
        with np.load(options['read_noise_bank'],allow_pickle=False) as archive:
            bank = torch.from_numpy(np.asarray(archive['read_noise_e'],dtype=np.float32))
            bank_metadata = json.loads(str(archive['metadata'].item()))
        if bank_metadata.get('schema')!='jsr-read-bank-v1' or bank_metadata.get('unit')!='electron' or int(bank_metadata.get('iso',0))!=int(iso) or bank_metadata.get('profile_sha256')!=profile_identity(profile):
            raise ValueError('read noise bank must use electron units and be bound to the selected ISO and exact camera profile')
        if bank.ndim!=3 or min(bank.shape[-2:])<size:
            raise ValueError('read noise bank must be electron residuals[T,H>=native,W>=native]')
        by = int(torch.randint(bank.shape[-2]-size+1,(),generator=noise_rng))
        bx = int(torch.randint(bank.shape[-1]-size+1,(),generator=noise_rng))
        bank = bank[:,by:by+size,bx:bx+size]
    captured = capture_sensor(native,exposure,transmission,profile['iso'][iso],noise_rng,
                     noise=bool(options.get('noise',True)),quantize=bool(options.get('quantize',True)),read_noise_bank=bank)
    captured.update(shifts=shifts,exposure=exposure,transmission=transmission,target=target)
    target_descriptions = {
        'pre_optics': 'before added optics',
        'post_optics': 'after reference optics, before pixel aperture',
        'post_pixel': 'after reference optics and native pixel aperture, sampled on a dense 2x grid',
    }
    captured['metadata'] = {'protocol':protocol,'sample_seed':int(seed),'spectral_input':source.shape[0]==61,
        'asset_identity':assets.identity,'camera_response_id':camera_id,
        'profile_sha256':profile_identity(profile),'profile_provenance':profile['provenance'],
        'spectral_curve_provenance':assets.provenance['cameras']['kind'],
        'target_stage':target_stage,
        'target_space':'D65-normalized camera RGB, reference exposure, '+target_descriptions[target_stage],
        'pixel_aperture':'retained' if target_stage == 'post_pixel' else 'excluded',
        'area_quadrature':4,'output_sample_pitch_native':.5,
        'sampling_phase':'dense center=(j+0.5)*2/2-0.5+margin; native center=(i+0.5)*2-0.5+margin; shifts in native pixels',
        'spectral_prior':'provided spectral radiance; provenance user-supplied' if source.shape[0]==61 else 'Mallett2019 public primary basis; cropped 400..700nm',
        'wavelengths_nm':assets.wavelengths.tolist(),'scene_gain':gain,'crop_xy':[ox,oy],
        'source_resized':resized,'replicated_boundary_context':padded_context,
        'iso':int(iso),'camera':profile['iso'][iso],'interval_ev':interval,'exposure_ratios':exposure.tolist(),
        'channel_transmission':transmission.tolist(),'f_number':f_number,'pitch_um':pitch,'fill_factor':fill,
        'psf':psf_provenance,
        'exposure_convention':'reference-white-normalized per ISO; not fixed absolute photons across ISO',
        'electrons_per_reference_unit':(profile['iso'][iso]['white_dn']-profile['iso'][iso]['black_dn'])*profile['iso'][iso]['gain_e_per_dn'],
        'aberrations_nm':aberrations,'lca_native':lca,'field_center':field,'field_extent':extent,
        'field_extent_region':'half-width of full simulated support including scene margin',
        'psf_radius_hr':radius,'pupil_samples':pupil_samples,'fft_size':fft_size,
        'signal_saturation_fraction':captured['signal_saturation'].float().mean().item(),
        'observed_saturation_fraction':captured['saturation'].float().mean().item(),
        'target_above_reference_white_fraction':(target>1).float().mean().item(),
        'noise_prior':'observed plug-in variance; no clean signal or true saturation fed to model',
        'cfa':'RGGB','scale_native':2,'shifts_xy_native':shifts.tolist()}
    return captured


class SpectralBurstDataset(Dataset):
    def __init__(self,manifest,split,options,seed=1234,profile=None):
        target_stage = resolve_target_stage(options)
        protocol = options.get('protocol', PROTOCOL)
        self.rows = read_manifest(manifest,split)
        self.options,self.seed,self.epoch = dict(options),int(seed),0
        self.profile = load_profile(profile)
        self.assets = SpectralAssets(options.get('asset_root'))
        self.samples_per_scene = int(options.get('samples_per_scene',1))
        if self.samples_per_scene<1:
            raise ValueError('samples_per_scene must be positive')
        # v2 identity must remain field-for-field compatible with checkpoints.
        self.identity = {'spectral_assets':self.assets.identity,'protocol':protocol}
        if protocol == PROTOCOL:
            self.identity['target_stage'] = target_stage
        if options.get('read_noise_bank'):
            self.identity['read_noise_bank_sha256'] = sha256(options['read_noise_bank'])
        if options.get('psf_library'):
            self.identity['psf_library_sha256'] = sha256(options['psf_library'])

    def __len__(self):
        return len(self.rows)*self.samples_per_scene

    def set_epoch(self,epoch):
        self.epoch = int(epoch)

    def __getitem__(self,index):
        if not 0<=index<len(self):
            raise IndexError(index)
        row = self.rows[index%len(self.rows)]
        token = f'{self.seed}/{self.epoch}/{index}/{row["sha256"]}'.encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8],'little')%(2**63-1)
        source = (load_spectral_scene(row['resolved_path'],self.assets.wavelengths)
                  if row['encoding']=='spectral_radiance' else load_rgb(row['resolved_path'],row['encoding']))
        result = synthesize_spectral(source,self.options,seed,self.profile,self.assets)
        if self.options.get('capture_order') is not None:
            result['capture_order'] = torch.as_tensor(self.options['capture_order'])
        result['metadata'].update(source_sha256=row['sha256'],scene_id=row['scene_id'],split=row['split'])
        result['metadata'] = json.dumps(result['metadata'],sort_keys=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=['train','val','test'],default='train')
    parser.add_argument('--count',type=int,default=1)
    args = parser.parse_args()
    if args.count<1:
        parser.error('--count must be positive')
    cfg = load_config(args.config)
    torch.set_num_threads(int(cfg.get('threads',2)))
    data = cfg['data']
    verify_manifest(data['manifest'])
    dataset = SpectralBurstDataset(data['manifest'],args.split,data['options'],cfg.get('seed',1234),data.get('profile'))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('use an empty output directory for a generated dataset')
    records = []
    for i in range(min(args.count,len(dataset))):
        sample = dataset[i]
        name = f'{args.split}_{i:06d}.npz'
        arrays = {k:v.numpy() for k,v in sample.items() if isinstance(v,torch.Tensor)}
        np.savez_compressed(output/name,**arrays,metadata=np.asarray(sample['metadata']))
        records.append({'file':name,'sha256':sha256(output/name),'metadata':json.loads(sample['metadata'])})
    save_json(output/'manifest.json',{'protocol':dataset.identity['protocol'],'identity':dataset.identity,'samples':records})
    print(json.dumps({'samples':len(records),'output':str(output)}))


if __name__ == '__main__':
    main()
