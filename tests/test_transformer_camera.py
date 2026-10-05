import copy
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from jsr_repro.bracket_data import BracketBurstDataset, bracket_exposures, save_burst, synthesize_bracket
from jsr_repro.calibration import analytic_profile, fit_profile, load_profile, profile_identity, validate_profile
from jsr_repro.data import verify_manifest
from jsr_repro.evaluate_transformer import detailed_metrics
from jsr_repro.infer_transformer import export_linear, infer, read_burst
from jsr_repro.optics import diffraction_psf, sensor_integrate
from jsr_repro.prepare import build_manifest
from jsr_repro.train_transformer import batch_sample, load_checkpoint, run_training
from jsr_repro.transformer import SpeechTransformer, align_planes, edge_amplitude, planes_to_rgb
from jsr_repro.utils import load_config


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def sample(**overrides):
    options = dict(native_size=16,scene_gain=[1,1],bracket_interval_ev=[.3,.3],iso=[100],max_shift=0,noise=False,quantize=False,augment=False)
    options.update(overrides)
    return synthesize_bracket(torch.full((3,100,100),.25),options,torch.Generator().manual_seed(88))


def test_camera_profile_units_provenance_and_ptc_recovery(tmp_path):
    profile = load_profile()
    assert profile["provenance"]["kind"] == "analytic_inferred"
    assert set(profile["iso"]) == {"100","200","400","800"}
    csv_path = tmp_path/"ptc.csv"
    csv_path.write_text("iso,mean_dn,variance_dn2\n"+"".join(f"{iso},{m},{m/g+(r/g)**2+1/12}\n" for iso,g,r in ((100,4,3),(200,2,4),(400,1,3),(800,.5,2)) for m in (10,100,1000)),encoding="utf-8")
    fitted = fit_profile(csv_path,tmp_path/"fitted.json")
    assert fitted["provenance"]["kind"] == "fitted_input"
    for iso,g,r in ((100,4,3),(200,2,4),(400,1,3),(800,.5,2)):
        assert fitted["iso"][str(iso)]["gain_e_per_dn"] == pytest.approx(g,abs=1e-10)
        assert fitted["iso"][str(iso)]["read_noise_e"] == pytest.approx(r,abs=1e-10)
    changed = copy.deepcopy(profile)
    changed["iso"]["100"]["gain_e_per_dn"] = 0
    with pytest.raises(ValueError,match="units"):
        validate_profile(changed)
    changed = copy.deepcopy(profile)
    changed["provenance"] = {}
    with pytest.raises(ValueError,match="provenance"):
        validate_profile(changed)


def test_dark_stack_uses_temporal_not_spatial_variance(tmp_path):
    rng = np.random.default_rng(8)
    dsnu = rng.normal(0,25,(32,32))
    stack = 512 + dsnu[None] + rng.normal(0,2,(128,32,32))
    np.save(tmp_path/"dark.npy",stack)
    csv_path = tmp_path/"ptc.csv"
    csv_path.write_text("iso,mean_dn,variance_dn2\n100,10,9\n100,100,54\n100,1000,504\n")
    fitted = fit_profile(csv_path,tmp_path/"fit.json",dark=tmp_path/"dark.npy")
    assert fitted["iso"]["100"]["read_noise_e"] == pytest.approx(np.sqrt(4-1/12)*2,rel=.02)
    assert fitted["provenance"]["dark"]["dsnu_std_dn"]>20
    assert fitted["provenance"]["dark"]["temporal_variance_dn2"]<5


@pytest.mark.parametrize("interval",[0,.3,1])
def test_bracket_interval_and_reference(interval):
    exposures = bracket_exposures(interval)
    assert exposures[0] == 1
    assert torch.log2(exposures.max()/exposures.min()).item() == pytest.approx(interval*6,abs=1e-6)
    if interval==0:
        assert torch.equal(exposures,torch.ones(7))
    with pytest.raises(ValueError):
        bracket_exposures(1.1)


def test_true_channel_saturation_and_unclipped_hdr_target():
    source = sample(scene_gain=[8,8])
    assert source["target"].min()>1
    assert source["saturation"].any() and not source["saturation"].all()
    # At shortest exposure (.536), green saturates but blue does not.
    assert source["saturation"][1,0,0,1] == 1
    assert source["saturation"][1,0,1,1] == 0
    assert source["metadata"]["signal_saturation_fraction"]>.1
    assert source["metadata"]["profile_provenance"]["kind"] == "analytic_inferred"


def test_noiseless_signal_units_and_noise_variance():
    source = sample()
    transmission = source["transmission"]
    expected = .25*source["exposure"]
    torch.testing.assert_close(source["raw"][:,0,0,0],expected*transmission[0],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(source["raw"][:,0,0,1],expected*transmission[1],atol=1e-6,rtol=1e-5)
    source = synthesize_bracket(torch.full((3,320,320),.2),dict(native_size=128,scene_gain=[1,1],max_shift=0,
                              bracket_interval_ev=[0,0],iso=[100],noise=True,quantize=False,augment=False),torch.Generator().manual_seed(44))
    plane = source["raw"][:,0,0::2,1::2]
    entry = analytic_profile()["iso"]["100"]
    electrons = (entry["white_dn"]-entry["black_dn"])*entry["gain_e_per_dn"]
    expected_var = .2/electrons + (entry["read_noise_e"]/electrons)**2
    assert plane.mean().item() == pytest.approx(.2,abs=5e-5)
    assert plane.var().item() == pytest.approx(expected_var,rel=.03)
    assert source["variance"][:,0,0::2,1::2].mean().item() == pytest.approx(expected_var,rel=.005)


def test_psf_units_wavelength_and_pixel_footprint():
    sharp = diffraction_psf(2,5.76,450)
    broad = diffraction_psf(8,3,650)
    assert sharp.sum() == pytest.approx(1,abs=1e-6)
    assert broad.sum() == pytest.approx(1,abs=1e-6)
    assert sharp.min()>=0 and broad.min()>=0
    assert sharp[12,12]>broad[12,12]
    assert not torch.equal(diffraction_psf(5,4,450),diffraction_psf(5,4,650))
    yy,xx = torch.meshgrid(torch.arange(100.),torch.arange(100.),indexing="ij")
    scene = (xx/100)[None].expand(3,-1,-1)
    shift = torch.tensor([[0.,0.],[.5,-.25]])
    integrated = sensor_integrate(scene,shift,16,20,.95)
    assert integrated[0,0,5,6].item() == pytest.approx((20+(6+.5)*2-.5)/100,abs=1e-6)
    assert (integrated[1]-integrated[0]).mean().item() == pytest.approx(.01,abs=1e-6)
    constant = sensor_integrate(torch.ones(3,100,100),shift,16,20,.8)
    torch.testing.assert_close(constant,torch.ones_like(constant))
    stripe = (.5+.4*torch.cos(xx*torch.pi/2))[None].expand(3,-1,-1)
    assert (sensor_integrate(stripe,shift,16,20,.8)-sensor_integrate(stripe,shift,16,20,1)).abs().max()>1e-4


def test_plane_alignment_sign_variance_and_boundaries():
    yy,xx = torch.meshgrid(torch.arange(8.),torch.arange(8.),indexing="ij")
    planes = xx[None,None,None].expand(1,7,4,-1,-1).clone()
    shifts = torch.zeros(1,7,2)
    shifts[:,1,0] = 1  # inverse packed shift is-.5
    zeros,ones = torch.zeros_like(planes),torch.ones_like(planes)
    moved,variance,sat,valid,black = align_planes(planes,ones,zeros,ones,zeros,shifts)
    assert moved[0,1,0,4,4] == 3.5
    assert variance[0,1,0,4,4] == .5
    assert valid[0,1,0,4,0] == 0
    assert valid[0,0,0,0,0] == 1
    saturated = zeros.clone()
    saturated[0,1,0,4,3]=1
    assert align_planes(planes,ones,saturated,ones,zeros,shifts)[2][0,1,0,4,4] == 1
    # RGB plane positions, not four arbitrarily co-located channels.
    native_ramp = torch.stack((2*xx,2*xx+1,2*xx,2*xx+1))[None]
    rgb = planes_to_rgb(native_ramp)
    torch.testing.assert_close(rgb[0,0,8,8:20],rgb[0,2,8,8:20])


def test_deterministic_dataset_scene_leak_and_profile_consumption(tmp_path):
    manifest = build_manifest(None,tmp_path/"data",procedural=6)
    options = dict(native_size=16)
    dataset = BracketBurstDataset(manifest,"train",options)
    torch.testing.assert_close(dataset[0]["raw"],dataset[0]["raw"],atol=0,rtol=0)
    dataset.set_epoch(1)
    assert not torch.equal(dataset[0]["raw"],BracketBurstDataset(manifest,"train",options)[0]["raw"])
    profile = analytic_profile()
    profile["channel_transmission"][0]=.2
    path=tmp_path/"camera.json"
    path.write_text(json.dumps(profile))
    custom = BracketBurstDataset(manifest,"train",options,profile=path)
    assert custom[0]["transmission"][0] == .2
    assert profile_identity(profile)!=profile_identity(analytic_profile())
    rows=[json.loads(row) for row in manifest.read_text().splitlines()]
    with manifest.open("a") as stream:
        stream.write(json.dumps(dict(rows[0],split="test" if rows[0]["split"]!="test" else "train"))+"\n")
    with pytest.raises(ValueError,match="leakage"):
        verify_manifest(manifest)


def test_retained_frames_shared_phases_gradient_and_metadata():
    torch.manual_seed(2)
    model=SpeechTransformer(width=8)
    inputs=batch_sample(sample(max_shift=1,noise=True),"cpu")
    output=model(inputs,trace=True)
    for name,shape in output["trace"].items():
        if name.startswith(("embedding","encoder","decoder")):
            assert shape[1]==7
    assert [output["trace"][f"late_pair{i}"][1] for i in range(3)]==[4,2,1]
    assert output["trace"]["shared_phase_latent"][-2:]==[32,32]
    assert isinstance(model.upsample[1],torch.nn.PixelShuffle) and isinstance(model.upsample[4],torch.nn.PixelShuffle)
    assert model.rgb_projection.out_channels==3
    output["rgb"].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.parameters())
    for key in ("exposure","variance","saturation"):
        changed={k:v.clone() for k,v in inputs.items()}
        if key=="exposure":
            changed[key][:,1:]*=2
        elif key=="variance":
            changed[key]*=1000
        else:
            changed[key][:,1:]=1
        assert not torch.allclose(model(changed)["rgb"],output["rgb"],atol=1e-8,rtol=0),key


def test_zero_gain_edge_preservation_and_hdr_output():
    torch.manual_seed(2)
    model=SpeechTransformer(width=8)
    inputs=batch_sample(sample(scene_gain=[8,8],bracket_interval_ev=[1,1]),"cpu")
    original=model(inputs)["rgb"]
    assert original.max()>1
    scaled={k:v.clone() for k,v in inputs.items()}
    scaled["raw"]*=.125
    scaled["variance"]*=.125**2
    torch.testing.assert_close(model(scaled)["rgb"],original*.125,atol=1e-6,rtol=1e-5)
    scaled["raw"].zero_()
    scaled["variance"].zero_()
    assert model(scaled)["rgb"].abs().max()==0
    step=torch.ones(1,4,8,8)
    step[:,:,:,:4]=.1
    amplitude=edge_amplitude(step)
    assert amplitude[0,0,4,3]<.2 and amplitude[0,0,4,4]>.9


@pytest.mark.parametrize("failure",["missing","wrong_k","exposure","negative_variance","nan","mask"])
def test_actionable_input_errors(failure):
    inputs=batch_sample(sample(),"cpu")
    if failure=="missing":
        del inputs["variance"]
    elif failure=="wrong_k":
        inputs["raw"]=inputs["raw"][:,:6]
    elif failure=="exposure":
        inputs["exposure"][0,1]=0
    elif failure=="negative_variance":
        inputs["variance"][0,0,0,0,0]=-1
    elif failure=="nan":
        inputs["transmission"][0,0]=float("nan")
    else:
        inputs["valid"][0,0,0,0,0]=2
    with pytest.raises(ValueError):
        SpeechTransformer(width=8)(inputs)


def test_uint16_linear_export_roundtrip_and_highlight_metrics(tmp_path):
    rgb=np.array([[[0.,1.,2.],[3.,4.,9.]]],dtype=np.float32)
    result=export_linear(tmp_path,rgb,white=8.)
    data=(tmp_path/"linear_rgb16.tiff").read_bytes()
    assert data[:4]==b"II*\x00"
    # Independent TIFF IFD parser reads the declared RGB16 storage.
    ifd=struct.unpack_from("<I",data,4)[0]
    count=struct.unpack_from("<H",data,ifd)[0]
    tags={}
    for i in range(count):
        tag,kind,n,value=struct.unpack_from("<HHII",data,ifd+2+i*12)
        tags[tag]=(kind,n,value)
    assert tags[277][2]==3 and tags[262][2]==2
    assert struct.unpack_from("<HHH",data,tags[258][2])==(16,16,16)
    restored=np.frombuffer(data,dtype="<u2",offset=tags[273][2]).reshape(rgb.shape).astype(np.float32)/65535*8
    np.testing.assert_allclose(restored,np.clip(rgb,0,8),atol=8/65535)
    import cv2
    decoded=cv2.imread(str(tmp_path/"linear_rgb16.tiff"),cv2.IMREAD_UNCHANGED)
    assert decoded.dtype==np.uint16 and decoded.shape==rgb.shape
    np.testing.assert_allclose(decoded[...,::-1].astype(np.float32)/65535*8,np.clip(rgb,0,8),atol=8/65535)
    assert result["clipped_fraction"]==pytest.approx(1/6)
    t=torch.full((1,3,16,16),2.)
    assert detailed_metrics(t+.1,t,4)["highlight_rmse"]==pytest.approx(.1,abs=1e-6)


def test_transformer_resume_checkpoint_and_inference_contract(tmp_path):
    manifest=build_manifest(None,tmp_path/"data",procedural=6)
    cfg=load_config(Path(__file__).resolve().parents[1]/"configs/transformer_smoke.yaml")
    cfg["train"]["steps"],cfg["train"]["validate_every"]=4,2
    cfg["data"]["manifest"]=str(manifest)
    profile_path=tmp_path/"camera.json"
    profile_path.write_text(json.dumps(analytic_profile()))
    cfg["data"]["profile"]=str(profile_path)
    cfg["output"]=str(tmp_path/"full")
    run_training(cfg)
    partial=copy.deepcopy(cfg)
    partial["output"]=str(tmp_path/"part")
    run_training(partial,stop_after=2)
    run_training(partial,resume=tmp_path/"part/last.pt")
    a,sa=load_checkpoint(tmp_path/"full/last.pt")
    b,sb=load_checkpoint(tmp_path/"part/last.pt")
    assert sa["scheduler"]==sb["scheduler"] and sa["step"]==sb["step"]==4
    assert all(torch.equal(v,b.state_dict()[name]) for name,v in a.state_dict().items())
    wrong=copy.deepcopy(partial)
    wrong["data"]["options"]["scene_gain"]=[1,2]
    with pytest.raises(ValueError,match="configuration changed"):
        run_training(wrong,resume=tmp_path/"part/last.pt")
    source=BracketBurstDataset(manifest,"test",cfg["data"]["options"],profile=profile_path)[0]
    save_burst(tmp_path/"burst.npz",source)
    inferred=infer(tmp_path/"full/last.pt",tmp_path/"burst.npz",tmp_path/"infer",alignment="provided",radiance_white=4)
    assert inferred["output_shape_hwc"]==[32,32,3]
    with pytest.raises(ValueError,match="max-native-pixels"):
        read_burst(tmp_path/"burst.npz","cpu",max_native_pixels=10)
    np.savez(tmp_path/"missing.npz",raw=source["raw"].numpy())
    with pytest.raises(ValueError,match="requires"):
        read_burst(tmp_path/"missing.npz","cpu")
    state=torch.load(tmp_path/"full/last.pt",weights_only=True)
    state["implementation"]="inferred-jsr-v1"
    torch.save(state,tmp_path/"wrong.pt")
    with pytest.raises(ValueError,match="checkpoint"):
        load_checkpoint(tmp_path/"wrong.pt")
    profile=analytic_profile()
    profile["iso"]["100"]["read_noise_e"]=4
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match="identity changed"):
        run_training(partial,resume=tmp_path/"part/last.pt")
