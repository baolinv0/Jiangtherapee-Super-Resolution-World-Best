"""Traceable 61-band RGB lifting and measured relative camera sensitivities.

Mallett & Yuksel's public primary basis (Colour v0.4.6, BSD) supplies a
metameric prior, not measured scene spectra. Data licenses/provenance live
next to the numeric tables. CIE observer curves are NOT camera sensitivities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .utils import sha256

ASSET_ROOT = Path(__file__).parent / 'spectral_assets'
RGB_TO_XYZ = torch.tensor([[.4124564,.3575761,.1804375],
                           [.2126729,.7151522,.0721750],
                           [.0193339,.1191920,.9503041]], dtype=torch.float64)


class SpectralAssets:
    def __init__(self, root=None):
        self.root = ASSET_ROOT if root is None else Path(root)
        basis = json.loads((self.root/'basis.json').read_text(encoding='utf-8'))
        cameras = json.loads((self.root/'cameras.json').read_text(encoding='utf-8'))
        if basis.get('schema') != 'jsr-spectral-basis-v1' or cameras.get('schema') != 'jsr-camera-spectra-v1':
            raise ValueError('unsupported spectral asset schema')
        self.wavelengths = torch.tensor(basis['wavelengths_nm'], dtype=torch.float32)
        expected = torch.arange(400.,701.,5.)
        if not torch.equal(self.wavelengths, expected) or cameras['wavelengths_nm'] != basis['wavelengths_nm']:
            raise ValueError('spectral assets must share the 400..700nm/5nm axis')
        self.basis = torch.tensor(basis['reflectance_basis'], dtype=torch.float32)
        self.observer = torch.tensor(basis['cie_xyz'], dtype=torch.float32)
        self.illuminant = torch.tensor(basis['illuminant_d65'], dtype=torch.float32) / 100.
        if self.basis.shape != (61,3) or self.observer.shape != (61,3) or self.illuminant.shape != (61,):
            raise ValueError('invalid spectral table dimensions')
        for value in (self.basis, self.observer, self.illuminant):
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError('spectral tables must be finite and nonnegative')
        if (self.illuminant<=0).any() or (self.observer[:,1]*self.illuminant).sum()<=0:
            raise ValueError('D65 and observer luminance integral must be strictly positive')
        if (self.basis.sum(1)-1).abs().max() > 1e-4:
            raise ValueError('primary basis must preserve white reflectance')
        # Composite trapezoid quadrature; both endpoints are included.
        self.integration_nm = torch.full((61,),5.)
        self.integration_nm[[0,-1]] = 2.5
        self.cameras = {}
        for name, raw in cameras['cameras'].items():
            value = torch.tensor(raw, dtype=torch.float32)
            if value.shape != (61,3) or not torch.isfinite(value).all() or (value < 0).any() or (value.sum(0)<=0).any():
                raise ValueError(f'invalid spectral response for {name}')
            self.cameras[name] = value
        self.provenance = {'basis': basis['provenance'], 'cameras': cameras['provenance']}
        self.identity = {name:sha256(self.root/name) for name in ('basis.json','cameras.json')}

    def lift(self, linear_rgb):
        if linear_rgb.ndim != 3 or linear_rgb.shape[0] != 3 or not torch.isfinite(linear_rgb).all() or (linear_rgb<0).any():
            raise ValueError('lifting requires finite nonnegative linear RGB[3,H,W]')
        reflectance = torch.einsum('lc,chw->lhw',self.basis.to(linear_rgb),linear_rgb)
        return reflectance * self.illuminant.to(linear_rgb)[:,None,None]

    def camera_matrix(self, camera_id):
        if camera_id not in self.cameras:
            raise ValueError(f'unknown camera response: {camera_id}')
        response = self.cameras[camera_id].T * self.integration_nm[None]
        # Relative shape only: normalize each channel's D65 reference to one.
        # Absolute/channel sensitivity is an EXPLICIT separate transmission.
        denominator = response @ self.illuminant
        if (denominator<=0).any():
            raise ValueError('camera D65 response integral must be positive')
        return response / denominator[:,None]

    def camera_rgb(self, spectrum, camera_id):
        if spectrum.ndim != 3 or spectrum.shape[0] != 61 or not torch.isfinite(spectrum).all() or (spectrum<0).any():
            raise ValueError('expected spectral radiance[61,H,W]')
        return torch.einsum('cl,lhw->chw',self.camera_matrix(camera_id).to(spectrum),spectrum)

    def to_linear_srgb(self, spectrum):
        weights = self.observer.T * self.integration_nm[None]
        weights /= (weights @ self.illuminant)[1]
        xyz = torch.einsum('cl,lhw->chw',weights.to(spectrum),spectrum)
        return torch.einsum('ab,bhw->ahw',torch.linalg.inv(RGB_TO_XYZ).to(spectrum),xyz)

    def roundtrip_report(self):
        colors = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[.18,.18,.18]],dtype=torch.float32).T[:,:,None]
        recovered = self.to_linear_srgb(self.lift(colors))
        return {'method':'Mallett2019-truncated-400-700', 'max_abs_linear_srgb':(recovered-colors).abs().max().item(),
                'rmse_linear_srgb':(recovered-colors).square().mean().sqrt().item(),
                'note':'Truncation/quadrature introduces error; spectra are an assumed metamer, not recovered ground truth.'}


def load_spectral_scene(path, wavelengths):
    """Optional measured spectra: safe NPZ radiance[L,H,W], wavelengths_nm[L].

    Values use the same relative radiance convention as D65/100. Input must
    cover 400..700nm; no extrapolation or automatic per-channel normalization.
    """
    with np.load(path,allow_pickle=False) as archive:
        axis = np.asarray(archive['wavelengths_nm'],dtype=np.float64)
        data = np.asarray(archive['radiance'],dtype=np.float32)
    if axis.ndim != 1 or len(axis)<2 or data.ndim!=3 or data.shape[0]!=len(axis) or not np.isfinite(axis).all() or not np.isfinite(data).all() or (data<0).any() or np.any(np.diff(axis)<=0):
        raise ValueError('spectral NPZ needs finite nonnegative radiance[L,H,W] and increasing wavelengths_nm[L]')
    target = wavelengths.cpu().numpy()
    if axis[0]>target[0] or axis[-1]<target[-1]:
        raise ValueError('spectral scene does not cover 400..700nm; extrapolation is disabled')
    upper = np.searchsorted(axis,target,side='right').clip(1,len(axis)-1)
    lower = upper-1
    alpha = ((target-axis[lower])/(axis[upper]-axis[lower])).astype(np.float32)
    return torch.from_numpy(data[lower]*(1-alpha[:,None,None])+data[upper]*alpha[:,None,None])
