import numpy as np
import pytest
import torch
from jsr_repro.geometry import correct_lca, fit_residual, estimate_geometry
from jsr_repro.model import JSRModel
from jsr_repro.frontend import local_amplitude, phase_splat
from jsr_repro.transformer import SpeechTransformer, align_planes
from jsr_repro.physical_optics import spectral_psfs

def sample():
    raw=torch.rand(1,7,1,32,32)
    return dict(raw=raw,shifts=torch.zeros(1,7,2),exposure=torch.ones(1,7),transmission=torch.ones(1,3),variance=torch.ones_like(raw)*.001,saturation=torch.zeros_like(raw),valid=torch.ones_like(raw),black_invalid=torch.zeros_like(raw))

def test_lca_sign_amplitude_and_homogeneity():
    x=torch.arange(32).float()[None,None,None].expand(1,3,32,32)
    offsets=[[-.5,0],[0,0],[.5,0]]
    observed=x+torch.tensor([-.5,0,.5])[None,:,None,None]*2
    corrected=correct_lca(observed,offsets,2)
    torch.testing.assert_close(corrected[...,2:-2],x[...,2:-2])
    torch.testing.assert_close(correct_lca(observed*3,offsets),corrected*3)
    model=JSRModel(controller_width=2,refine_width=2,refine_blocks=1,lca_offsets_native=offsets)
    result=model(torch.rand(1,2,1,16,16),torch.zeros(1,2,2),diagnostics=True)
    torch.testing.assert_close(result['learned'],correct_lca(result['pre_lca_learned'],offsets))
    torch.testing.assert_close(result['controller_amplitude'],local_amplitude(result['initial_legacy']))
    torch.testing.assert_close(result['refine_amplitude'],local_amplitude(result['legacy']))
    torch.testing.assert_close(result['legacy'],correct_lca(result['initial_legacy'],offsets))

def test_polynomial_fit_and_fallback():
    rng=np.random.default_rng(3); p=rng.uniform(0,63,(100,2)); residual=np.stack((.001*p[:,0]**2,.002*p[:,1]),1)
    degree,coef=fit_residual(p,residual,64,64)
    from jsr_repro.geometry import basis
    assert degree==4
    np.testing.assert_allclose(basis(p,degree,64,64)@coef,residual,atol=1e-8)
    assert fit_residual(p[:1],residual[:1],64,64)[0]==-1
    geometry=estimate_geometry(torch.zeros(1,3,1,32,32))
    assert geometry.reports[0][1]['status']=='fallback'
    shifts=torch.zeros(1,3,2)
    raw=torch.ones(1,3,1,32,32)
    a=phase_splat(raw,shifts); b=phase_splat(raw,shifts,geometry=geometry)
    torch.testing.assert_close(a['legacy'],b['legacy'])

def test_estimated_translation_and_dense_inverse():
    import cv2
    rng=np.random.default_rng(12)
    im=cv2.GaussianBlur(rng.random((96,96)).astype('float32'),(5,5),0)
    shifted=cv2.warpAffine(im,np.float32([[1,0,-2],[0,1,1]]),(96,96),borderMode=cv2.BORDER_REFLECT)
    raw=torch.tensor(np.stack([im,shifted,im]))[None,:,None]
    g=estimate_geometry(raw)
    assert g.reports[0][1]['status']=='estimated'
    yy,xx=torch.meshgrid(torch.arange(96),torch.arange(96),indexing='ij')
    delta=g.forward[0,1]-torch.stack((xx,yy),-1)
    torch.testing.assert_close(delta[20:-20,20:-20].mean((0,1)),torch.tensor([2.,-1.]),atol=.4,rtol=0)
    planes=torch.rand(1,3,4,48,48); zeros=torch.zeros_like(planes); ones=torch.ones_like(planes)
    aligned=align_planes(planes,zeros,zeros,ones,zeros,torch.zeros(1,3,2),g)
    assert all(torch.isfinite(v).all() for v in aligned)
    # Dense translation must equal the existing oracle path, including support/variance.
    from jsr_repro.geometry import Geometry
    shifts=torch.tensor([[[0.,0.],[2.,-2.],[0.,0.]]])
    identity=torch.stack((xx,yy),-1).float()[None,None].expand(1,3,-1,-1,-1)
    dense=Geometry(identity+shifts[:,:,None,None],identity-shifts[:,:,None,None],[])
    baseline=align_planes(planes,ones,zeros,ones,zeros,shifts)
    mapped=align_planes(planes,ones,zeros,ones,zeros,shifts,dense)
    for expected,actual in zip(baseline,mapped):torch.testing.assert_close(expected,actual)
    oracle=phase_splat(raw,shifts)
    reconstructed=phase_splat(raw,torch.zeros_like(shifts),geometry=dense)
    for key in ('legacy','count','sum','phase'):torch.testing.assert_close(oracle[key],reconstructed[key])


def test_capture_storage_permutation_and_legacy():
    torch.manual_seed(3); s=sample(); s['capture_order']=torch.tensor([[3,0,1,2,4,5,6]])
    model=SpeechTransformer(width=4,capture_order_mode='explicit-ranks-v1').eval()
    perm=torch.tensor([0,4,2,6,1,5,3]); other={k:(v[:,perm] if k!='transmission' else v) for k,v in s.items()}
    with torch.no_grad():
        torch.testing.assert_close(model(s)['rgb'],model(other)['rgb'],atol=2e-6,rtol=2e-6)
    s['capture_order'][0,0]=0
    with pytest.raises(ValueError): model(s)
    del s['capture_order']
    with pytest.raises(ValueError): model(s)
    assert torch.isfinite(SpeechTransformer(width=4)(s)['rgb']).all()

def test_spherical_spectral_psf():
    kw=dict(f_number=2,pitch_um=3,radius=4,pupil_samples=32,fft_size=64)
    base=spectral_psfs([400,550,700],**kw)
    psf=spectral_psfs([400,550,700],aberrations_nm={'spherical':150},**kw)
    assert (psf>=0).all() and torch.isfinite(psf).all()
    torch.testing.assert_close(psf.sum((-2,-1)),torch.ones(3))
    assert not torch.allclose(base,psf)

def test_reference_metadata_gating():
    from jsr_repro.real_reference import validate_metadata
    with pytest.raises(ValueError): validate_metadata({})
    m=dict(camera_identity='test fixture only',cfa='RGGB',black_dn=512,white_dn=16383,iso=100,exposure_seconds=[.001,.002],transmission=[1,1,1],independent_long_reference=True,reference_exposure_seconds=.512,reference_iso=100,reference_transmission=[1,1,1],color_units='linear camera RGB, reference-short exposure units')
    assert validate_metadata(m)==512
    m['reference_iso']=200
    with pytest.raises(ValueError): validate_metadata(m)

def test_diagnostic_metrics_separate_artifacts_from_chroma():
    from jsr_repro.revision_diagnostics import fixture,error_metrics,DN_VALUES
    assert len(DN_VALUES)==29 and DN_VALUES[:5]==[0,1,2,3,4]
    truth=fixture('chroma_x',displacements=False)
    assert error_metrics(truth,truth)['true_chroma_projection']==pytest.approx(1)
    gray=truth.mean(0,keepdim=True).expand_as(truth)
    assert error_metrics(gray,truth)['true_chroma_projection']==pytest.approx(0,abs=1e-5)
    y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    checker=truth+.02*((-1.)**(x+y))
    assert error_metrics(checker,truth)['checkerboard_residual_projection']>.019

def test_homography_with_smooth_residual_and_reference_guard():
    import cv2
    rng=np.random.default_rng(28)
    im=cv2.GaussianBlur(rng.random((128,128)).astype('float32'),(5,5),0)
    y,x=np.mgrid[:128,:128].astype('float32')
    sx=x+1+.0001*(x-64)**2+.002*y; sy=y-.7+.00007*(y-64)**2
    moved=cv2.remap(im,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    raw=torch.tensor(np.stack([im,moved,im]))[None,:,None]
    g=estimate_geometry(raw)
    assert g.reports[0][1]['status']=='estimated'
    assert g.reports[0][1]['cross_frame_mean_px'] is not None
    assert g.reports[0][1]['degree']>=1
    truth=torch.tensor(np.stack((sx,sy),-1))
    assert (g.forward[0,1,20:-20,20:-20]-truth[20:-20,20:-20]).square().mean().sqrt()<.5
    g.forward[:,0]+=1
    with pytest.raises(ValueError): phase_splat(raw,torch.zeros(1,3,2),geometry=g)

def test_natural_protocol_runnable_local_linear(tmp_path):
    import json
    from jsr_repro.natural_diagnostic import run
    source=tmp_path/'fixture.npy';np.save(source,np.ones((96,96,3),np.float32)*.3)
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(source=str(source),encoding='linear',units='synthetic linear white1',provenance='unit test; NOT natural evidence')))
    assert run(spec,tmp_path/'output')['status']=='RUN'

def test_real_protocol_registered_synthetic_adapter(tmp_path,monkeypatch):
    # Synthetic RAW-decoder stub checks runnable registration/normalization only.
    import json,cv2
    import jsr_repro.real_reference as protocol
    rng=np.random.default_rng(48)
    im=cv2.GaussianBlur(rng.random((64,64)).astype('float32'),(5,5),0)*.1
    files=[tmp_path/'short0.RW2',tmp_path/'short1.RW2'];long=tmp_path/'long.RW2'
    for path in files+[long]: path.write_bytes(('synthetic test only '+path.name).encode())
    def adapter(paths,output,crop):
        frames=np.stack([im[None]*(2 if p==long else 1) for p in paths])
        metadata={'frames':[dict(black_per_channel=[512]*4,white_per_channel=[16383]*4) for p in paths]}
        np.savez(output,raw=frames,metadata=json.dumps(metadata))
    monkeypatch.setattr(protocol,'import_files',adapter)
    m=dict(camera_identity='unit test only',cfa='RGGB',black_dn=512,white_dn=16383,iso=100,exposure_seconds=[.001,.001],transmission=[1,1,1],independent_long_reference=True,reference_exposure_seconds=.002,reference_iso=100,reference_transmission=[1,1,1],color_units='linear camera RGB, reference-short exposure units')
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(burst=[str(p) for p in files],long_reference=str(long),metadata=m)))
    report=protocol.run(spec,tmp_path/'output')
    assert report['status']=='RUN' and report['rmse_camera_radiance']<1e-4
    assert report['nine_stop_claim'] is False


def test_projective_homography_correspondence():
    import cv2
    rng=np.random.default_rng(51)
    scene=cv2.GaussianBlur(rng.random((128,128)).astype('float32'),(5,5),0)
    hom=np.array([[1.004,.003,1.2],[-.002,.999,-.6],[.00004,-.00003,1.]])
    moved=cv2.warpPerspective(scene,hom,(128,128),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT)
    raw=torch.tensor(np.stack((scene,moved,scene)))[None,:,None]
    geo=estimate_geometry(raw)
    assert geo.reports[0][1]['status']=='estimated'
    y,x=np.mgrid[:128,:128];points=np.stack((x,y),-1).reshape(1,-1,2).astype('float64')
    truth=torch.tensor(cv2.perspectiveTransform(points,hom).reshape(128,128,2)).float()
    assert (geo.forward[0,1,20:-20,20:-20]-truth[20:-20,20:-20]).square().mean().sqrt()<.5


def test_capture_order_archive_roundtrip_and_fractional_rejection(tmp_path):
    import json
    from jsr_repro.bracket_data import save_burst
    from jsr_repro.spectral_data import synthesize_spectral
    from jsr_repro.calibration import load_profile,profile_identity
    from jsr_repro.infer_transformer import read_burst
    from jsr_repro.train_transformer import batch_sample
    s=synthesize_spectral(torch.ones(3,96,96)*.3,dict(native_size=16,noise=False,quantize=False),12)
    s['capture_order']=torch.tensor([3,0,1,2,4,5,6])
    path=tmp_path/'burst.npz';save_burst(path,s)
    loaded,meta=read_burst(path,'cpu')
    torch.testing.assert_close(loaded['capture_order'],s['capture_order'][None])
    assert meta['capture_order_provenance']=='explicit ranks'
    assert 'capture_order' in batch_sample(s,'cpu')
    loaded['capture_order']=loaded['capture_order'].float()+.1
    from jsr_repro.transformer import validate_inputs
    with pytest.raises(ValueError):validate_inputs(loaded)


def test_ring_sweep_fixed_physical_masks_and_quadrature():
    from jsr_repro.revision_diagnostics import fixture,ring_masks,DN_VALUES,DN_WHITE,MIDPOINTS
    masks=ring_masks();union=masks['disk']|masks['annulus']
    assert len(MIDPOINTS)**2==16 and DN_VALUES[-1]==16383
    assert all(m.any() for m in masks.values())
    baseline=fixture('rings',dn=0,displacements=False)
    for dn in DN_VALUES:
        for displaced in (False,True):
            image=fixture('rings',dn=dn,displacements=displaced)
            assert torch.equal(ring_masks()['disk'],masks['disk'])
            assert torch.equal(ring_masks()['annulus'],masks['annulus'])
            torch.testing.assert_close(image[:,union],torch.full_like(image[:,union],dn/DN_WHITE))
        # Surroundings remain identical, independent of swept intensity.
        image=fixture('rings',dn=dn,displacements=False)
        torch.testing.assert_close(image[:,0,:],baseline[:,0,:])
        assert image[:,31,42].min()>.9


@pytest.mark.parametrize('mixed',[False,True])
def test_saturated_short_valid_long_actual_registration(tmp_path,monkeypatch,mixed):
    import json,cv2
    import jsr_repro.real_reference as protocol
    im=cv2.GaussianBlur(np.random.default_rng(48).random((64,64)).astype('float32'),(5,5),0)
    files=[tmp_path/'s0.RW2',tmp_path/'s1.RW2'];long=tmp_path/'long.RW2'
    for p in files+[long]:p.write_bytes(('synthetic decoder test '+p.name).encode())
    shorts=[np.minimum(2*im,1),np.minimum((4 if mixed else 2)*im,1)]
    def adapter(paths,output,crop):
        values=[.04*im if p==long else shorts[files.index(p)] for p in paths]
        np.savez(output,raw=np.stack(values)[:,None],metadata=json.dumps({'frames':[dict(black_per_channel=[512]*4,white_per_channel=[16383]*4) for p in paths]}))
    monkeypatch.setattr(protocol,'import_files',adapter)
    meta=dict(camera_identity='SYNTHETIC decoder stub only',cfa='RGGB',black_dn=512,white_dn=16383,iso=100,exposure_seconds=[.001,.002 if mixed else .001],transmission=[1]*3,independent_long_reference=True,reference_exposure_seconds=.002,reference_iso=100,reference_transmission=[.01]*3,color_units='linear camera RGB, reference-short exposure units')
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(burst=list(map(str,files)),long_reference=str(long),metadata=meta)))
    if mixed:
        with pytest.raises(ValueError,match='no joint valid unsaturated'):
            protocol.run(spec,tmp_path/'out')
        return
    result=protocol.run(spec,tmp_path/'out')
    assert result['status']=='RUN' and result['registration'][0][1]['status']=='estimated'
    assert result['short_adc_invalid_fraction']>.5
    assert 0<result['valid_fraction']<.49
    assert result['exposure_ratio']==2 and result['nine_stop_claim'] is False


def test_masked_splat_does_not_count_invalid_zero_as_observation():
    raw=torch.ones(1,2,1,16,16);valid=torch.ones_like(raw,dtype=torch.bool);valid[:,1]=False
    masked=phase_splat(raw,torch.zeros(1,2,2),1,validity=valid)
    single=phase_splat(raw[:,:1],torch.zeros(1,1,2),1)
    for key in ('sum','count','legacy'):torch.testing.assert_close(masked[key],single[key])
    empty=phase_splat(raw,torch.zeros(1,2,2),1,validity=torch.zeros_like(valid))
    assert not empty['count'].any() and not empty['sum'].any()
