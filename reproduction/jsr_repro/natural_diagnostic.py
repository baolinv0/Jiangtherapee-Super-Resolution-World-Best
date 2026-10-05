"""Tier B natural linear RGB / 61-band spectral file evaluation, explicit provenance.
No bundled natural source exists. Input is local; no automatic downloads.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .spectral_data import synthesize_spectral
from .spectral import SpectralAssets
from .calibration import load_profile
from .transformer import SpeechTransformer
from .train_transformer import batch_sample
from .utils import sha256

def run(spec,output):
    request=json.loads(Path(spec).read_text(encoding='utf-8-sig'))
    source=Path(request['source']);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    report=dict(tier='B',status='NOT_RUN',missing=[str(source)])
    if source.is_file():
        if request.get('encoding') not in ('linear','spectral_radiance','srgb') or not request.get('provenance') or not request.get('units'):
            raise ValueError('explicit linear/spectral encoding, source provenance and units required')
        torch.set_num_threads(2);torch.manual_seed(214)
        from .spectral import load_spectral_scene
        assets=SpectralAssets()
        if request['encoding']=='spectral_radiance':
            scene=load_spectral_scene(source,assets.wavelengths)
        elif request['encoding']=='srgb':
            from .data import load_rgb
            scene=load_rgb(source,'srgb')
        else:
            a=np.load(source,allow_pickle=False)
            if a.ndim!=3 or a.shape[-1]!=3: raise ValueError('linear RGB input must be HWC npy')
            scene=torch.tensor(a,dtype=torch.float32).permute(2,0,1)
        options=dict(protocol='spectral-camera-v2',native_size=16,frames=7,scale=2,noise=False,quantize=False,augment=False)
        sample=synthesize_spectral(scene,options,214,load_profile(),assets)
        with torch.no_grad():result=SpeechTransformer(width=4)(batch_sample(sample,'cpu'))
        from .metrics import image_metrics
        report=dict(tier='B',status='RUN',source_sha256=sha256(source),provenance=request['provenance'],units=request['units'],encoding=request['encoding'],checkpoint='random seed214',construction=sample['metadata'],metrics={k:image_metrics(result[k],sample['target'][None],4) for k in ('baseline','rgb')})
    (root/'natural_status.json').write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.spec,a.output)
