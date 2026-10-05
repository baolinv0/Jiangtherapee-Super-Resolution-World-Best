"""Real RAW reference protocol. No fitted gain; missing files => NOT_RUN.
Uses existing RAW adapter. CPU, RGGB camera color units, static scenes only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .import_raw import import_files
from .utils import sha256
from .geometry import estimate_geometry
from .frontend import phase_splat

def validate_metadata(meta):
    for key in ('camera_identity','cfa','black_dn','white_dn','iso','exposure_seconds','transmission','independent_long_reference','reference_exposure_seconds','reference_iso','reference_transmission','color_units'):
        if key not in meta: raise ValueError('missing real-reference metadata: '+key)
    if meta['cfa']!='RGGB' or meta['independent_long_reference'] is not True or not meta['camera_identity'] or meta['color_units']!='linear camera RGB, reference-short exposure units':
        raise ValueError('declare RGGB, independent reference, camera identity and linear camera RGB units')
    if meta['reference_iso']!=meta['iso']:
        raise ValueError('equal measured ISO required; no invented ISO gain conversion')
    if not 2<=len(meta['exposure_seconds'])<=14 or any(not np.isfinite(v) or v<=0 for v in meta['exposure_seconds']):
        raise ValueError('2..14 positive measured exposure seconds required')
    if not np.isfinite(meta['reference_exposure_seconds']) or meta['reference_exposure_seconds']<=0:
        raise ValueError('positive independent reference exposure required')
    for name in ('transmission','reference_transmission'):
        a=np.asarray(meta[name]);
        if a.shape!=(3,) or not np.isfinite(a).all() or (a<=0).any() or (a>1).any(): raise ValueError('measured RGB transmission in (0,1] required')
    black=np.asarray(meta['black_dn']);white=np.asarray(meta['white_dn'])
    if not np.isfinite(black).all() or not np.isfinite(white).all() or (white<=black).any(): raise ValueError('finite black/white levels required')
    return meta['reference_exposure_seconds']/meta['exposure_seconds'][0]

def run(spec,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    request=json.loads(Path(spec).read_text(encoding='utf-8-sig'))
    files=[Path(p) for p in request.get('burst',[])]; reference=Path(request.get('long_reference','MISSING_LONG_REFERENCE'))
    missing=[str(p) for p in files+[reference] if not p.is_file()]
    if not files: missing.append('RAW burst list')
    report=dict(tier='C',status='NOT_RUN',missing=missing,nine_stop_claim=False)
    if not missing:
        torch.set_num_threads(2)
        if reference.resolve() in [p.resolve() for p in files] or sha256(reference) in [sha256(p) for p in files]:
            raise ValueError('long reference must be an independent file, not reused burst data')
        meta=request['metadata'];ratio=validate_metadata(meta)
        if len(files)!=len(meta['exposure_seconds']): raise ValueError('exposure list differs from burst')
        import_files(files,root/'burst.npz',request.get('crop'))
        import_files([reference],root/'long.npz',request.get('crop'))
        with np.load(root/'burst.npz',allow_pickle=False) as a:
            raw=torch.tensor(a['raw'])[None];burst_meta=json.loads(str(a['metadata'].item()))
        with np.load(root/'long.npz',allow_pickle=False) as a:
            long=torch.tensor(a['raw'])[None];long_meta=json.loads(str(a['metadata'].item()))
        for frame in burst_meta['frames']+long_meta['frames']:
            if not np.allclose(frame['black_per_channel'],meta['black_dn']) or not np.allclose(frame['white_per_channel'],meta['white_dn']):
                raise ValueError('declared black/white differ from RAW decoder; matching ADC units required')
        # Decoder values are (ADC-black)/(white-black), before exposure scaling.
        short_valid=torch.isfinite(raw)&(raw>0)&(raw<1)
        long_valid=torch.isfinite(long)&(long>0)&(long<1)
        raw=torch.nan_to_num(raw);long=torch.nan_to_num(long)
        exposure=torch.tensor(meta['exposure_seconds']);raw=raw/(exposure/exposure[0])[None,:,None,None,None]
        geometry=estimate_geometry(raw)
        short=phase_splat(raw,torch.zeros(1,len(files),2),1,geometry,validity=short_valid)
        merged=short['sum']/short['count'].clamp_min(1e-8)
        total_short=phase_splat(torch.ones_like(raw),torch.zeros(1,len(files),2),1,geometry)['count']
        short_support=(short['count']>1e-6)&(short['count']>=total_short*(1-1e-6))
        # Register long exposure independently to short reference. No target-derived gain.
        combined=torch.cat((raw[:,:1],long/ratio),1)
        reg=estimate_geometry(combined)
        if reg.reports[0][1]['status']!='estimated':
            report.update(missing=['reliable long-reference spatial registration'],registration=reg.reports)
        else:
            both=phase_splat(combined,torch.zeros(1,2,2),1,reg)
            first=phase_splat(combined[:,:1],torch.zeros(1,1,2),1)
            count=both['count']-first['count']
            truth=(both['sum']-first['sum'])/count.clamp_min(1e-8)
            valid_raw=torch.cat((torch.zeros_like(raw[:,:1]),long_valid.to(long)),1)
            valid_sum=phase_splat(valid_raw,torch.zeros(1,2,2),1,reg)['sum']
            spatial_support=(count>1e-6)&(valid_sum>=count*(1-1e-6))
            short_t=torch.tensor(meta['transmission'])[None,:,None,None];long_t=torch.tensor(meta['reference_transmission'])[None,:,None,None]
            merged/=short_t;truth/=long_t
            border=int(request.get('border_native',8));p=merged[...,border:-border,border:-border];t=truth[...,border:-border,border:-border]
            if p.numel()==0: raise ValueError('border excludes whole crop')
            # Gate saturated/black long reference using independently declared ADC units.
            support=(spatial_support&short_support)[...,border:-border,border:-border]&(t>0)&(t<1/ratio/long_t)
            if not support.any(): raise ValueError('no joint valid unsaturated short/long reconstruction support')
            report.update(status='RUN',file_hashes={str(p):sha256(p) for p in files+[reference]},exposure_ratio=ratio,metadata=meta,registration=reg.reports,burst_alignment=geometry.reports,valid_fraction=float(support.float().mean()),short_adc_invalid_fraction=float((~short_valid).float().mean()),support_policy='positive accepted short counts; all short and registered long contributors valid in pre-exposure ADC units',rmse_camera_radiance=float((p-t)[support].square().mean().sqrt()),method='independent corrected merge; no fitted gain; static scene only',border_native=border)
    (root/'real_status.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();run(a.spec,a.output)
