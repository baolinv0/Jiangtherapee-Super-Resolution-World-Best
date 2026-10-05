"""Independent Tier A fixtures; DN normalization white=16383; uint16 storage maximum=65535, no author geometry claim."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .model import JSRModel
from .transformer import SpeechTransformer
from .data import mosaic_rggb
from .physical_optics import spectral_psfs
from .geometry import correct_lca, estimate_geometry

DN_VALUES=[0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,4096,8192,10240,12288,14336,16383]

DN_WHITE=16383
MIDPOINTS=(-.375,-.125,.125,.375)

def ring_masks(size=64):
    """INFERRED fixed physical plateau interiors; 8-pixel border excluded."""
    y,x=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    r=((x-(size-1)/2)**2+(y-(size-1)/2)**2).sqrt()
    interior=(x>=8)&(x<size-8)&(y>=8)&(y<size-8)
    return {'disk':interior&(r<=6), 'annulus':interior&(r>=15)&(r<=18)}

def fixture(kind,size=64,displacements=True,dn=4096):
    """INFERRED fixed disk r<=9 and annulus 12<=r<=21; DN-valued plateaus.
    White16383 gap (9,12), black exterior; 29 separate intensity cases.
    Stripe period=8 native pixels; true chroma stripe amplitude .1, mean .3.
    Channel observation shifts -0.5/0/+0.5 native x pixels.
    """
    y,x=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    rgb=torch.zeros(3,size,size)
    for dy in MIDPOINTS:
        for dx in MIDPOINTS:
            for c,shift in enumerate((-.5,0,.5) if displacements else (0,0,0)):
                xx=x+dx+shift; yy=y+dy
                if kind=='rings':
                    radius=((xx-(size-1)/2)**2+(yy-(size-1)/2)**2).sqrt()
                    plateau=(radius<=9)|((radius>=12)&(radius<=21))
                    value=torch.where(plateau,dn/DN_WHITE,torch.where((radius>9)&(radius<12),1.,0.))
                else:
                    axis=xx if kind.endswith('x') else yy
                    value=.3+.1*torch.sin(axis*2*torch.pi/8)
                    if kind.startswith('chroma'): value=.3+(1 if c==0 else -1 if c==2 else 0)*.1*torch.sin(axis*2*torch.pi/8)
                rgb[c]+=value/16
    return rgb

def error_metrics(rgb,target,border=8,axis='x'):
    p=rgb[...,border:-border,border:-border]; t=target[...,border:-border,border:-border]
    error=p-t
    chroma=torch.stack((p[0]-p[1],p[2]-p[1])); truth=torch.stack((t[0]-t[1],t[2]-t[1]))
    y,x=torch.meshgrid(torch.arange(p.shape[-2]),torch.arange(p.shape[-1]),indexing='ij')
    # Cross-axis error is RMS of residual differences orthogonal to stripe direction.
    cross=error.diff(dim=-2 if axis=='x' else -1).square().mean().sqrt()
    checker=(error*((-1.)**(x+y))).mean((-2,-1)).square().mean().sqrt()
    denom=truth.square().sum()
    return dict(rmse=float(error.square().mean().sqrt()),cross_axis_residual_rms=float(cross),checkerboard_residual_projection=float(checker),false_chroma_rms=float((chroma-truth).square().mean().sqrt()),true_chroma_projection=float((chroma*truth).sum()/denom) if denom>1e-10 else None)

def run(output,natural_spec=None):
    torch.set_num_threads(2); torch.manual_seed(214)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    model=JSRModel(scale=1,controller_width=4,refine_width=4,refine_blocks=1,lca_offsets_native=[[-.5,0],[0,0],[.5,0]]).eval()
    transformer=SpeechTransformer(width=4).eval()
    records=[]; tiles=[]
    with torch.no_grad():
        for kind in ('rings','stripe_x','stripe_y','chroma_x','chroma_y'):
            target=fixture(kind,displacements=False); observed=fixture(kind); raw=mosaic_rggb(observed[None].expand(7,-1,-1,-1))[None]
            result=model(raw,torch.zeros(1,7,2),diagnostics=True)
            for name in ('initial_legacy','legacy','learned','rgb'):
                records.append(dict(fixture=kind,method=name,**error_metrics(result[name][0],target,axis=kind[-1])))
            corrected=correct_lca(result['initial_legacy'],[[-.5,0],[0,0],[.5,0]],1)[0]
            records.append(dict(fixture=kind,method='LCA_corrected',**error_metrics(corrected,target,axis=kind[-1])))
            s=dict(raw=raw,shifts=torch.zeros(1,7,2),exposure=torch.ones(1,7),transmission=torch.ones(1,3),variance=torch.zeros_like(raw),saturation=torch.zeros_like(raw),black_invalid=torch.zeros_like(raw),valid=torch.ones_like(raw))
            tr=transformer(s)
            t2=torch.nn.functional.interpolate(target[None],scale_factor=2,mode='bilinear',align_corners=False)[0]
            for name in ('baseline','rgb'):
                records.append(dict(fixture=kind,method='transformer_'+name,**error_metrics(tr[name][0],t2,border=16,axis=kind[-1])))
            for name,v in ((kind+' target',target),(kind+' legacy',result['legacy'][0]),(kind+' random final',result['rgb'][0])):
                tiles.append((name,Image.fromarray(np.uint8(v.permute(1,2,0).clamp(0,1).numpy()*255))))
        curve=[]
        # Identical deterministic scene, adding independently sampled native subpixel shifts.
        target=fixture('stripe_x'); y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
        shifts=torch.rand(1,14,2)*2-1;shifts[:,0]=0
        raws=[]
        for shift in shifts[0]:
            grid=torch.stack((2*(x+shift[0]+.5)/64-1,2*(y+shift[1]+.5)/64-1),-1)
            moved=torch.nn.functional.grid_sample(target[None],grid[None],align_corners=False,padding_mode='border')
            raws.append(mosaic_rggb(moved))
        burst=torch.stack(raws,1)
        from .frontend import phase_splat
        for k in range(2,15):
            evidence=phase_splat(burst[:,:k],shifts[:,:k],1)
            pred=model(burst[:,:k],shifts[:,:k])
            curve.append(dict(K=k,coverage_fraction=float((evidence['phase'].reshape(1,3,48,64,64)[:,:,:16,8:-8,8:-8]>1e-5).float().mean()),legacy_rmse=error_metrics(pred['legacy'][0],target)['rmse'],final_rmse=error_metrics(pred['rgb'][0],target)['rmse']))
    optical=[]
    from .spectral import SpectralAssets
    from .physical_optics import apply_spectral_kernels
    assets=SpectralAssets();camera_id=sorted(assets.cameras)[0]
    spectrum=assets.lift(fixture('stripe_x',displacements=False))
    clear_camera=torch.nn.functional.avg_pool2d(assets.camera_rgb(spectrum,camera_id)[None],2)[0]
    saved_lca=model.lca_offsets_native;model.lca_offsets_native=None
    for condition in ('defocus','coma','astigmatism','spherical'):
        psf=spectral_psfs(torch.linspace(400,700,61),2,3,radius=6,field_xy=(.7,.4),aberrations_nm={condition:100},pupil_samples=32,fft_size=128)
        optical.append(dict(condition=condition,coefficient_nm=100,wavelengths=61,min=float(psf.min()),sum_range=[float(psf.sum((-2,-1)).min()),float(psf.sum((-2,-1)).max())],center_range=[float(psf[:,6,6].min()),float(psf[:,6,6].max())]))
        blurred=apply_spectral_kernels(spectrum,psf[None])
        camera_native=torch.nn.functional.avg_pool2d(assets.camera_rgb(blurred,camera_id)[None],2)[0]
        optical_raw=mosaic_rggb(camera_native[None].expand(7,-1,-1,-1))[None]
        with torch.no_grad():prediction=model(optical_raw,torch.zeros(1,7,2))
        optical[-1]['camera_response_id']=camera_id
        optical[-1]['clear_target_units']='relative D65-normalized camera RGB, linear white1'
        optical[-1]['reconstruction_metrics']={name:error_metrics(prediction[name][0],clear_camera,border=8) for name in ('legacy','learned','rgb')}
        tiles.append((condition+' final',Image.fromarray(np.uint8(prediction['rgb'][0].permute(1,2,0).clamp(0,1).detach().numpy()*255))))
        tiles.append((condition+' 550nm' ,Image.fromarray(np.uint8(psf[30].numpy()/psf[30].max().item()*255)).convert('RGB')))
    model.lca_offsets_native=saved_lca
    sheet=Image.new('RGB',(3*192,((len(tiles)+2)//3)*216),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,img) in enumerate(tiles):
        px,py=i%3*192,i//3*216;draw.text((px+4,py+4),name,fill='black');sheet.paste(img.resize((184,184)),(px+4,py+24))
    sheet.save(root/'fixtures.png')
    for name,rows in (('metrics',records),('k_curve',curve)):
        with (root/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    # Identical geometry and physical sampling areas for every intensity case.
    masks=ring_masks(); step_rows=[]; response={}
    methods=('initial_legacy','legacy','rgb')
    for dn in DN_VALUES:
        raw=mosaic_rggb(fixture('rings',dn=dn)[None].expand(7,-1,-1,-1))[None]
        with torch.no_grad(): out=model(raw,torch.zeros(1,7,2),diagnostics=True)
        row=dict(DN=dn,samples=sum(int(m.sum()) for m in masks.values()),disk_samples=int(masks['disk'].sum()),annulus_samples=int(masks['annulus'].sum()))
        mask=masks['disk']|masks['annulus']
        for name in methods:
            row[name+'_mean_dn']=float(out[name][0,:,mask].mean()*DN_WHITE)
        step_rows.append(row)
    for name in methods:
        truth=np.asarray(DN_VALUES,dtype=float);pred=np.asarray([v[name+'_mean_dn'] for v in step_rows])
        slope,intercept=np.polyfit(truth,pred,1);residual=pred-(slope*truth+intercept);dark=truth<=512
        response[name]=dict(slope=float(slope),intercept_dn=float(intercept),affine_response_residual_rmse_dn=float(np.sqrt(np.mean(residual**2))),identity_response_rmse_dn=float(np.sqrt(np.mean((pred-truth)**2))),dark_max_dn=512,dark_cases=int(dark.sum()),dark_identity_response_rmse_dn=float(np.sqrt(np.mean((pred[dark]-truth[dark])**2))),dark_affine_response_residual_rmse_dn=float(np.sqrt(np.mean(residual[dark]**2))))
    # Alignment evidence uses independently generated smooth coordinates solely for scoring.
    import cv2
    rng=np.random.default_rng(28)
    scene=cv2.GaussianBlur(rng.random((128,128)).astype('float32'),(5,5),0)
    yy,xx=np.mgrid[:128,:128].astype('float32')
    sx=xx+1+.0001*(xx-64)**2+.002*yy;sy=yy-.7+.00007*(yy-64)**2
    moved=cv2.remap(scene,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    geo=estimate_geometry(torch.tensor(np.stack((scene,moved,scene)))[None,:,None])
    alignment=dict(reports=geo.reports,forward_rmse_native=float((geo.forward[0,1,20:-20,20:-20]-torch.tensor(np.stack((sx,sy),-1))[20:-20,20:-20]).square().mean().sqrt()),oracle_used_by_estimator=False,flat_reports=estimate_geometry(torch.zeros(1,3,1,32,32)).reports)
    with (root/'stepped_response.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=step_rows[0].keys());writer.writeheader();writer.writerows(step_rows)
    # Portable plots, fixed axes and no fitted display gain.
    chart=Image.new('RGB',(720,360),'white');d=ImageDraw.Draw(chart)
    d.text((10,8),'K curve: legacy blue / random final red; RMSE linear white1',fill='black')
    maximum=max(max(v['legacy_rmse'],v['final_rmse']) for v in curve)*1.1
    d.line((45,30,45,320,690,320),fill='black')
    for field,color in (('legacy_rmse','blue'),('final_rmse','red')):
        d.line([(45+(v['K']-2)*645/12,320-v[field]/maximum*280) for v in curve],fill=color,width=2)
    for v in curve:d.text((40+(v['K']-2)*645/12,325),str(v['K']),fill='black')
    d.text((5,30),f'{maximum:.3f}',fill='black');d.text((20,305),'0',fill='black');chart.save(root/'k_curve.png')
    chart=Image.new('RGB',(720,360),'white');d=ImageDraw.Draw(chart)
    d.text((10,8),'Stepped response: log1p DN x-axis; observed mean DN y-axis / white16383',fill='black')
    d.line((45,30,45,320,690,320),fill='black')
    for field,color in (('initial_legacy_mean_dn','blue'),('rgb_mean_dn','red')):
        points=[(45+np.log1p(v['DN'])/np.log1p(DN_WHITE)*645,320-v[field]/DN_WHITE*280) for v in step_rows if v[field] is not None]
        d.line(points,fill=color,width=2)
    d.text((50,325),'0 DN',fill='black');d.text((630,325),'16383 DN',fill='black');chart.save(root/'stepped_response.png')
    report=dict(optical_asset_identity=assets.identity,coverage_definition='fraction of 48 color/phase bins with smoothed count/14 > 1e-5, excluding 8 native border',alignment=alignment,tier_A='RUN' ,checkpoint='random seed214; no trained quality claim',DN_white=DN_WHITE,uint16_storage_max=65535,fixture_definition='INFERRED independent geometry and DN list',DN_values=DN_VALUES,ring_geometry='fixed center (31.5,31.5); disk r<=9, annulus 12<=r<=21 at swept DN; gap white16383, exterior black; masks disk r<=6, annulus 15<=r<=18; border8',plateau_sample_counts={k:int(v.sum()) for k,v in masks.items()},response_definition='unweighted affine fit of 29 RGB plateau means in DN; identity RMSE to input DN; dark subset DN<=512; same combined physical mask every case',integration='16 midpoint samples per native pixel',channel_observation_offsets_native=[-.5,0,.5],border_native=8,metrics=records,k_curve=curve,optical=optical,stepped_response=response,tier_B=dict(status='NOT_RUN',reason='no natural input passed; procedural fixtures are Tier A'),tier_C=dict(status='NOT_RUN',missing=['S5M2 RAW burst','independent long-exposure RAW','measured exposure/ISO/transmission metadata'],nine_stop_claim=False))
    if natural_spec:
        from .natural_diagnostic import run as natural_run
        report['tier_B']=natural_run(natural_spec,root/'natural')
    (root/'diagnostics.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--natural-spec');a=parser.parse_args();run(a.output,a.natural_spec)
