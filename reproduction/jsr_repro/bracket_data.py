"""Speech-inspired physical camera proxy; all default parameters are inferred."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .calibration import load_profile, profile_identity
from .data import load_rgb, mosaic_rggb, read_manifest, _uniform
from .optics import optical_blur, sensor_integrate


def bracket_exposures(interval):
    if not math.isfinite(interval) or not 0 <= interval <= 1:
        raise ValueError("bracket interval must be0..1 EV; seven-shot span is six intervals")
    ev = torch.tensor([0., -3., -2., -1., 1., 2., 3.]) * interval
    return 2 ** ev


def synthesize_bracket(source, options, generator, profile=None):
    profile = load_profile() if profile is None else profile
    size = int(options.get("native_size", 16))
    if size < 16 or size % 2 or int(options.get("frames", 7)) != 7 or int(options.get("scale", 2)) != 2:
        raise ValueError("Transformer dataset requires K7, scale2, even native_size>=16")
    for name,default,low,high in (("scene_gain",[1.,4.],0.,float("inf")),("bracket_interval_ev",[0.,1.],0.,1.),
                                  ("f_number",[2.,8.],2.,8.),("pitch_um",[3.,5.76],3.,5.76),
                                  ("fill_factor",[.9,1.],.8,1.)):
        bounds=options.get(name,default)
        if not isinstance(bounds,(list,tuple)) or len(bounds)!=2 or not all(math.isfinite(float(v)) for v in bounds) or not low<=bounds[0]<=bounds[1]<=high or (name=="scene_gain" and bounds[0]<=0):
            raise ValueError(f"invalid {name} range; use ordered finite bounds inside [{low},{high}]")
    max_shift = float(options.get("max_shift", 1.))
    if not math.isfinite(max_shift) or max_shift < 0 or max_shift > 4:
        raise ValueError("synthetic shifts must be0..4 native pixels")
    radius = int(options.get("psf_radius", 12))
    margin = radius + int(math.ceil((max_shift + 2) * 2))
    extent = size * 2 + 2 * margin
    resized = min(source.shape[-2:]) < extent
    if resized:
        factor = extent / min(source.shape[-2:])
        source = F.interpolate(source[None], size=[math.ceil(d * factor) for d in source.shape[-2:]], mode="bilinear", align_corners=False)[0]
    oy = int(torch.randint(source.shape[-2] - extent + 1, (), generator=generator))
    ox = int(torch.randint(source.shape[-1] - extent + 1, (), generator=generator))
    scene = source[:, oy:oy + extent, ox:ox + extent].clone()
    if options.get("augment", True):
        scene = torch.rot90(scene, int(torch.randint(4, (), generator=generator)), (-2, -1))
        if torch.rand((), generator=generator) > .5:
            scene = scene.flip(-1)
    gain = _uniform(generator, *options.get("scene_gain", [1., 4.]))
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("scene_gain must be positive")
    f_number = _uniform(generator, *options.get("f_number", [2., 8.]))
    pitch = _uniform(generator, *options.get("pitch_um", [3., 5.76]))
    fill = _uniform(generator, *options.get("fill_factor", [.9, 1.]))
    scene = optical_blur(scene * gain, f_number, pitch, radius)
    target = scene[:, margin:margin + 2 * size, margin:margin + 2 * size].clone()
    shifts = (torch.rand(7, 2, generator=generator) * 2 - 1) * max_shift
    shifts[0] = 0
    rgb_native = sensor_integrate(scene, shifts, size, margin, fill)
    interval = _uniform(generator, *options.get("bracket_interval_ev", [0., 1.]))
    exposure = bracket_exposures(interval)
    choices = [str(int(v)) for v in options.get("iso", [100, 200, 400, 800])]
    if not choices or any(v not in profile["iso"] for v in choices):
        raise ValueError("requested ISO is missing from camera profile")
    iso = choices[int(torch.randint(len(choices), (), generator=generator))]
    camera = profile["iso"][iso]
    transmission = torch.tensor(profile["channel_transmission"], dtype=torch.float32)
    span = camera["white_dn"] - camera["black_dn"]
    electrons_per_unit = span * camera["gain_e_per_dn"]
    clean = mosaic_rggb(rgb_native * exposure[:, None, None, None] * transmission[None, :, None, None])
    signal_e = clean.clamp_min(0) * electrons_per_unit
    clip_e = min(camera["full_well_e"], electrons_per_unit)
    saturation = signal_e >= clip_e  # physical signal, before random noise
    captured_e = signal_e.clamp_max(camera["full_well_e"])
    if options.get("noise", True):
        noisy_e = torch.poisson(captured_e, generator=generator)
        noisy_e += torch.randn(noisy_e.shape, generator=generator) * camera["read_noise_e"]
    else:
        noisy_e = captured_e.clone()
    dn = noisy_e / camera["gain_e_per_dn"] + camera["black_dn"]
    black_invalid = dn < camera["black_dn"]
    # Retain negative read excursions after black subtraction, but clip ADC top.
    dn = dn.clamp(0, camera["white_dn"])
    quantize = bool(options.get("quantize", True))
    if quantize:
        dn = dn.round()
    raw = (dn - camera["black_dn"]) / span
    # Deployment-compatible plug-in estimate from observed photoelectrons.
    observed_e = raw.clamp_min(0) * electrons_per_unit
    variance = (observed_e + camera["read_noise_e"] ** 2) / electrons_per_unit ** 2
    if quantize:
        variance += 1 / (12 * span ** 2)
    return {"raw": raw.float(), "shifts": shifts, "exposure": exposure,
            "transmission": transmission, "variance": variance.float(), "saturation": saturation.float(),
            "valid": torch.ones_like(raw), "black_invalid": black_invalid.float(), "target": target.float(),
            "metadata": {"protocol": "speech-camera-proxy-v1", "profile_sha256": profile_identity(profile),
                         "profile_provenance": profile["provenance"], "iso": int(iso), "camera": camera,
                         "scene_gain": gain, "interval_ev": interval, "exposure_ratios": exposure.tolist(),
                         "channel_transmission": transmission.tolist(), "f_number": f_number, "pitch_um": pitch,
                         "wavelength_nm_rgb": [650, 550, 450], "psf_radius_hr": radius,
                         "psf": "truncated normalized Airy, circular ideal aperture; no measured aberrations",
                         "fill_factor": fill, "area_quadrature": 4, "quantized_dn": quantize,
                         "signal_saturation_fraction": saturation.float().mean().item(),
                         "black_excursion_fraction": black_invalid.float().mean().item(),
                         "target_above_reference_white_fraction": (target > 1).float().mean().item(),
                         "cfa": "RGGB", "scale_native": 2, "crop_xy": [ox, oy], "source_resized": resized,
                         "shifts_xy_native": shifts.tolist(), "radiance_white": 1., "dark_offset_subtracted": True}}


class BracketBurstDataset(Dataset):
    def __init__(self, manifest, split, options, seed=1234, profile=None):
        self.rows = read_manifest(manifest, split)
        self.options, self.seed, self.epoch = dict(options), int(seed), 0
        self.profile = load_profile(profile)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        token = f"speech/{self.seed}/{self.epoch}/{index}/{row['sha256']}".encode()
        sample_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2 ** 63 - 1)
        sample = synthesize_bracket(load_rgb(row["resolved_path"], row["encoding"]), self.options,
                                    torch.Generator().manual_seed(sample_seed), self.profile)
        if self.options.get("capture_order") is not None:
            sample["capture_order"] = torch.as_tensor(self.options["capture_order"])
        sample["metadata"].update(source_sha256=row["sha256"], scene_id=row["scene_id"], split=row["split"], sample_seed=sample_seed)
        sample["metadata"] = json.dumps(sample["metadata"], sort_keys=True)
        return sample


def save_burst(path, sample):
    fields = ("raw", "shifts", "exposure", "transmission", "variance", "saturation", "valid", "black_invalid")
    fields = (*fields, *(("capture_order",) if "capture_order" in sample else ()))
    metadata = sample["metadata"] if isinstance(sample["metadata"], str) else json.dumps(sample["metadata"], sort_keys=True)
    np.savez_compressed(path, **{key: sample[key].cpu().numpy() for key in fields}, metadata=np.asarray(metadata))


def make_burst_dataset(manifest, split, options, seed=1234, profile=None):
    """Select a versioned data protocol; old checkpoints retain their recipe."""
    protocol = options.get('protocol', 'speech-camera-proxy-v1')
    if protocol in ('spectral-camera-v2', 'spectral-camera-v3'):
        from .spectral_data import SpectralBurstDataset
        return SpectralBurstDataset(manifest, split, options, seed, profile)
    if protocol == 'speech-camera-proxy-v1':
        return BracketBurstDataset(manifest, split, options, seed, profile)
    raise ValueError(f'unknown data protocol: {protocol}')
