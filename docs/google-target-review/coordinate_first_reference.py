"""Coordinate-first, discrete-field camera reference; never a physical GT claim.

For each field-node PSF, convolve the original spectral samples once to obtain
C_i. At each sensor footprint quadrature position q, evaluate C_i(q + 2*u),
blend with sensor-fixed weights w_i(q), integrate SRF and then the pixel area.
This removes the intermediate moved-scene resampling. The source and optical
fields still use bilinear reconstruction, finite-support discrete PSFs, and
midpoint area quadrature. It is an implementation reference, not measured optics
or a continuous-scene truth. No repository production files are modified.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from jsr_repro.optics import sensor_integrate
from jsr_repro.physical_optics import (
    _convolve, apply_spectral_kernels, prepare_spectral_kernels, spectral_psfs,
)
from jsr_repro.spectral_data import _warp
from jsr_repro.spectral import SpectralAssets


def prepare_fields(spectrum, kernels):
    """Prepare [nodes,L,H,W] convolution fields from the original scene."""
    if kernels.shape[0] not in (1, 4) or kernels.shape[1] != spectrum.shape[0]:
        raise ValueError('one or four PSF nodes with matching spectral channels required')
    return torch.stack([_convolve(spectrum, node) for node in kernels])


def prepare_camera_fields(spectrum, kernels, srf):
    """Memory-efficient equivalent: spectral convolution BEFORE SRF.

    The common spatial interpolation, node weights and footprint do not depend on
    wavelength, so SRF can commute past those linear operators. This is not a
    three-channel PSF approximation: all L physical kernels are still applied.
    Only node-local convolution results are projected to camera RGB before caching.
    """
    return torch.stack([torch.einsum('cl,lhw->chw', srf.to(spectrum),
                                   _convolve(spectrum, node)) for node in kernels])


def sensor_queries(size, margin, fill_factor=1., quadrature=4,
                   output_scale=1, sampling_origin_hr=(0., 0.), dtype=torch.float32):
    """Return [N,N,Q²,xy] sensor coordinates independent of scene motion.

    output_scale changes the spacing of centers, never the footprint. The optional
    sampling origin is for testing arbitrary center lattices, not a new GT phase.
    """
    if output_scale not in (1, 2) or not 0 < fill_factor <= 1:
        raise ValueError('output scale1/2 and positive fill<=1 required')
    if not isinstance(quadrature, int) or quadrature < 2:
        raise ValueError('integer midpoint quadrature>=2 required')
    n = size * output_scale
    yy, xx = torch.meshgrid(torch.arange(n, dtype=dtype),
                           torch.arange(n, dtype=dtype), indexing='ij')
    centers = torch.stack(((xx + .5)*2/output_scale - .5 + margin + sampling_origin_hr[0],
                           (yy + .5)*2/output_scale - .5 + margin + sampling_origin_hr[1]), -1)
    offsets = ((torch.arange(quadrature, dtype=dtype)+.5)/quadrature-.5)*2*math.sqrt(fill_factor)
    oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
    return centers[:, :, None] + torch.stack((ox.flatten(), oy.flatten()), -1)


def fixed_field_weights(sensor_q, height, width, nodes):
    """Weights are evaluated at sensor q, never q+scene displacement."""
    if nodes == 1:
        return torch.ones((1, *sensor_q.shape[:-1]), dtype=sensor_q.dtype)
    tx, ty = sensor_q[..., 0]/(width-1), sensor_q[..., 1]/(height-1)
    return torch.stack(((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty))


def query_fields(fields, q):
    """One bilinear lookup in each precomputed convolution field."""
    nodes, bands, height, width = fields.shape
    n, _, count, _ = q.shape
    if q[..., 0].min() < 0 or q[..., 0].max() > width-1 or q[..., 1].min() < 0 or q[..., 1].max() > height-1:
        raise ValueError('increase scene margin: query exits precomputed field')
    grid = 2*(q+.5)/q.new_tensor([width, height])-1
    sampled = F.grid_sample(fields, grid.reshape(1,n,n*count,2).expand(nodes,-1,-1,-1),
                            mode='bilinear', padding_mode='border', align_corners=False)
    return sampled.reshape(nodes,bands,n,n,count)


def capture_from_fields(fields, shifts, size, margin, *, fill_factor=1.,
                        quadrature=4, output_scale=1, srf=None,
                        sampling_origin_hr=(0.,0.)):
    """Noise-free [frames,C,N,N] integrated reference; shifts are native xy.

    SRF is an optional [C,L] linear response matrix, already normalized to the
    caller's target units. Transmission, exposure, CFA, noise and clipping belong
    downstream. A matching call with output_scale=2, shifts=[[0,0]] provides
    post_pixel GT using exactly the same optical fields and native footprint.
    """
    nodes, bands, height, width = fields.shape
    sensor_q = sensor_queries(size, margin, fill_factor, quadrature,
                             output_scale, sampling_origin_hr, fields.dtype)
    weights = fixed_field_weights(sensor_q, height, width, nodes)
    outputs = []
    for shift in shifts.to(fields):
        sampled = query_fields(fields, sensor_q + shift*2)
        # Field weights stay fixed in SENSOR coordinates, including each
        # footprint integration position. They are not moved with the scene.
        spectrum = (sampled*weights[:,None]).sum(0)
        channels = spectrum if srf is None else torch.einsum('cl,lhwq->chwq', srf.to(fields), spectrum)
        outputs.append(channels.mean(-1))
    return torch.stack(outputs)


def current_capture(spectrum, kernels, shifts, size, margin, fill_factor=1., output_scale=1):
    return torch.stack([sensor_integrate(apply_spectral_kernels(_warp(spectrum, shift, 2), kernels),
                                        torch.zeros(1,2), size, margin, fill_factor,
                                        output_scale=output_scale)[0] for shift in shifts])


def run_probes():
    torch.set_num_threads(1)
    size, margin = 32, 24
    extent = 2*size+2*margin
    yy, xx = torch.meshgrid(torch.arange(extent).float(), torch.arange(extent).float(), indexing='ij')
    results = {'scope':'discrete bilinear-field reference; not continuous physical truth'}

    # 1. One stationary pupil: preserve zero-displacement native capture and
    # resolve the fractional-motion attenuation from the earlier review.
    field = (.5+.3*torch.cos(2*torch.pi*.25*xx+.23))[None].expand(3,-1,-1).clone()
    kernels = spectral_psfs([550]*3,2.,5.76,radius=8,pupil_samples=64,fft_size=256)[None]
    fields = prepare_fields(field,kernels)
    shifts = torch.tensor([[0.,0.],[.25,0.],[.5,0.]])
    reference = capture_from_fields(fields,shifts,size,margin)
    current = current_capture(field,kernels,shifts,size,margin)
    direct = sensor_integrate(fields[0],shifts,size,margin,fill_factor=1.)
    assert torch.allclose(reference,direct,atol=2e-7,rtol=1e-6)
    assert torch.allclose(reference[0],current[0],atol=2e-7,rtol=1e-6)
    ratios=[]
    for old,new in zip(current[:,0,16],reference[:,0,16]):
        a,b=new-new.mean(),old-old.mean()
        ratios.append(float((b@a)/(a@a)))
    results['stationary_motion']={'shifts_native':shifts.tolist(),
        'coordinate_vs_direct_max':float((reference-direct).abs().max()),
        'zero_shift_vs_current_max':float((reference[0]-current[0]).abs().max()),
        'current_amplitude_ratios':ratios}

    # 2. Native-area dense GT, including 4-node actual pupil fields. Center-lattice
    # offset .5HR makes dense even coordinates identical to native centers. This
    # is center placement, not scene motion; w_i(q) therefore also matches.
    source=torch.stack((.45+.18*torch.cos(.7*xx)+.1*torch.sin(.5*yy),
                        .5+.2*torch.cos(.4*xx+.3*yy),
                        .4+.12*torch.sin(.6*xx-.4*yy)))
    four=prepare_spectral_kernels([450.,550.,650.],f_number=2.8,pitch_um=4.43,radius=8,
                                  field_center=(.4,.3),field_extent=.3,
                                  aberrations_nm={'defocus':40.,'astigmatism':120.,'coma':100.},
                                  lca_native=.8,pupil_samples=64,fft_size=256)
    fields4=prepare_fields(source,four)
    zero=torch.zeros(1,2)
    native=capture_from_fields(fields4,zero,size,margin)
    aligned_dense=capture_from_fields(fields4,zero,size,margin,output_scale=2,sampling_origin_hr=(.5,.5))
    dense=capture_from_fields(fields4,zero,size,margin,output_scale=2)
    narrow=capture_from_fields(fields4,zero,size,margin,output_scale=2,fill_factor=.25)
    assert torch.equal(aligned_dense[...,::2,::2],native)
    assert (dense-narrow).abs().max()>1e-3
    # A direct zero-query reconstruction agrees with its own target, rather than
    # claiming exact legacy 4-node parity for different interpolation policies.
    q=sensor_queries(size,margin,output_scale=2)
    weights=fixed_field_weights(q,extent,extent,4)
    manual=(query_fields(fields4,q)*weights[:,None]).sum(0).mean(-1)
    assert torch.equal(manual,dense[0])
    results['shared_native_footprint']={'native_aligned_dense_max':float((aligned_dense[...,::2,::2]-native).abs().max()),
        'dense_direct_zero_query_max':float((manual-dense[0]).abs().max()),
        'native_footprint_vs_half_width_dense_max':float((dense-narrow).abs().max()),
        'dense_shape':list(dense.shape),'native_shape':list(native.shape)}

    # 3. Explicitly test sensor-fixed field interpolation using linear ramps and
    # displaced PSFs. Exact linearity means the motion difference is 2*u/extent,
    # irrespective of the PSF node and its field weight. Querying a previously
    # mixed field instead wrongly translates the weights and fails this test.
    ramp=torch.stack((xx/extent,yy/extent,torch.ones_like(xx)))
    delayed=torch.zeros(4,3,9,9)
    for node,(dx,dy) in enumerate([(-3,-3),(3,-3),(-3,3),(3,3)]):
        delayed[node,:,4+dy,4+dx]=1
    ramp_fields=prepare_fields(ramp,delayed)
    motion=torch.tensor([[0.,0.],[2.,1.]])
    fixed=capture_from_fields(ramp_fields,motion,size,margin)
    expected=torch.tensor([4/extent,2/extent,0.])[:,None,None].expand(3,size,size)
    fixed_error=float((fixed[1]-fixed[0]-expected).abs().max())
    assert fixed_error<3e-7
    mixed=apply_spectral_kernels(ramp,delayed)
    wrong=sensor_integrate(mixed,motion,size,margin,fill_factor=1.)
    wrong_error=float((wrong[1]-wrong[0]-expected).abs().max())
    assert wrong_error>1e-4
    q=sensor_queries(size,margin)
    w=fixed_field_weights(q,extent,extent,4)
    wrong_w=fixed_field_weights(q+motion[1]*2,extent,extent,4)
    results['sensor_fixed_fields']={'linear_ramp_motion_error':fixed_error,
        'moving_already_mixed_field_error':wrong_error,
        'weight_change_if_wrongly_moved_max':float((w-wrong_w).abs().max()),
        'correct_weight_change_between_frames':0.0}

    # 4. A realistic four-pupil-node exercise with noise-free three-band SRF.
    # SRF and area integration commute because every band has the same footprint.
    motion4=torch.tensor([[0.,0.],[.25,-.125],[.6,.2]])
    srf=torch.tensor([[.7,.2,.1],[.1,.8,.1],[.1,.2,.7]])
    actual=capture_from_fields(fields4,motion4,size,margin,srf=srf)
    spectral=capture_from_fields(fields4,motion4,size,margin)
    projected=torch.einsum('cl,flhw->fchw',srf,spectral)
    assert torch.allclose(actual,projected,atol=2e-7,rtol=1e-6)
    legacy=current_capture(source,four,motion4,size,margin)
    results['four_pupil_nodes']={'nodes':4,'wavelengths_nm':[450,550,650],
        'srf_integral_commutation_max':float((actual-projected).abs().max()),
        'zero_motion_discrete_mixing_policy_difference_max':float((spectral[0]-legacy[0]).abs().max()),
        'nonzero_motion_vs_current_max':[float((spectral[i]-legacy[i]).abs().max()) for i in [1,2]]}

    # This is a numerical convergence comparison, not a declaration that Q64 is
    # a physical ground truth. Keep separate from source-grid/PSF convergence.
    q16=capture_from_fields(fields4,motion4,size,margin,quadrature=16)
    q64=capture_from_fields(fields4,motion4,size,margin,quadrature=64)
    results['area_convergence']={'q4_vs_q64_max':float((spectral-q64).abs().max()),
        'q16_vs_q64_max':float((q16-q64).abs().max())}

    # 5. Exercise the repository's full 61-band response integration and K7,
    # rather than relying only on a toy RGB bank. This is the forward operator
    # before exposure/transmission/CFA/noise; it does not run network training.
    assets=SpectralAssets()
    full_spectrum=assets.lift(source)
    full_kernels=prepare_spectral_kernels(assets.wavelengths,f_number=2.8,pitch_um=4.43,
        radius=8,field_center=(.4,.3),field_extent=.3,
        aberrations_nm={'defocus':40.,'astigmatism':120.,'coma':100.},
        lca_native=.8,pupil_samples=64,fft_size=256)
    full_fields=prepare_fields(full_spectrum,full_kernels)
    seven=torch.tensor([[0.,0.],[.25,-.125],[.6,.2],[-.3,.4],[.1,-.2],[-.4,-.35],[.5,.5]])
    matrix=assets.camera_matrix('Canon 5DMarkII')
    full_native=capture_from_fields(full_fields,seven,16,margin,srf=matrix)
    full_dense=capture_from_fields(full_fields,zero,16,margin,srf=matrix,output_scale=2)
    full_aligned=capture_from_fields(full_fields,zero,16,margin,srf=matrix,output_scale=2,
                                    sampling_origin_hr=(.5,.5))
    camera_fields=prepare_camera_fields(full_spectrum,full_kernels,matrix)
    cached_camera_native=capture_from_fields(camera_fields,seven,16,margin)
    cache_error=float((cached_camera_native-full_native).abs().max())
    assert cache_error<4e-7
    assert torch.equal(full_aligned[...,::2,::2],full_native[:1])
    assert torch.isfinite(full_native).all() and (full_native>=0).all()
    assert torch.isfinite(full_dense).all() and (full_dense>=0).all()
    assert torch.isfinite(full_aligned).all() and (full_aligned>=0).all()
    results['full_spectral_k7']={'wavelength_count':61,'field_nodes':4,'frames':7,
        'camera_response':'Canon 5DMarkII','native_camera_rgb_shape':list(full_native.shape),
        'dense_post_pixel_target_shape':list(full_dense.shape),
        'reference_native_vs_aligned_dense_max':float((full_aligned[...,::2,::2]-full_native[:1]).abs().max()),
        'finite_nonnegative':True,'bank_convolutions_per_burst':4*61,
        'source_warps_per_burst':0,
        'project_after_convolution_rgb_cache_max_error':cache_error,
        'full_spectral_cache_values':full_fields.numel(),
        'projected_camera_cache_values':camera_fields.numel()}
    return results


if __name__=='__main__':
    results=run_probes()
    output=Path(__file__).with_name('coordinate_first_results.json')
    output.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
