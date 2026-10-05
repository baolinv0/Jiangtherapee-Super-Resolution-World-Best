"""Inferred synthetic training recipe, NOT the undisclosed JSR training data.

Coordinates: an observed sensor sample (x,y) in frame k sees the reference
at (x+dx[k], y+dy[k]), in native mosaic pixels. Output ratio is relative
to the *mosaic*, not to the half-size packed Bayer plane.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1 / 2.4) - 0.055)


def load_rgb(path, encoding):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype.kind != "f":
            raise ValueError(f"linear .npy must be floating HWC RGB: {path}")
        x = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).permute(2, 0, 1)
    else:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError(f"source must be finite normalized [0,1] RGB: {path}")
    if encoding == "srgb":
        x = srgb_to_linear(x)
    elif encoding != "linear":
        raise ValueError("source encoding must be srgb or linear")
    return x


def read_manifest(path, split=None):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty manifest")
    groups, hashes = {}, {}
    for row in rows:
        if row.get("split") not in ("train", "val", "test"):
            raise ValueError("manifest split must be train, val or test")
        for key, seen in (("scene_id", groups), ("sha256", hashes)):
            value = row[key]
            if value in seen and seen[value] != row["split"]:
                raise ValueError(f"source leakage across splits: {key}={value}")
            seen[value] = row["split"]
        source = Path(row["path"])
        row["resolved_path"] = str(source if source.is_absolute() else path.parent / source)
    selected = [r for r in rows if split is None or r["split"] == split]
    if not selected:
        raise ValueError(f"manifest has no samples for split {split}")
    return selected


def verify_manifest(path):
    from .utils import sha256
    rows = read_manifest(path)
    for row in rows:
        if sha256(row["resolved_path"]) != row["sha256"]:
            raise ValueError(f"source hash mismatch: {row['path']}")
    return rows


def mosaic_rggb(rgb):
    """Accept (...,3,H,W), return (...,1,H,W); green includes both sites."""
    if rgb.shape[-3] != 3 or any(d % 2 for d in rgb.shape[-2:]):
        raise ValueError("RGGB requires RGB and even spatial dimensions")
    raw = rgb[..., 1:2, :, :].clone()
    raw[..., 0, 0::2, 0::2] = rgb[..., 0, 0::2, 0::2]
    raw[..., 0, 1::2, 1::2] = rgb[..., 2, 1::2, 1::2]
    return raw


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    radius = max(1, int(np.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    kernel2 = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, -1, -1)
    return F.conv2d(F.pad(x[None], (radius,) * 4, mode="reflect"), kernel2, groups=3)[0]


def _uniform(generator, low, high):
    return low + (high - low) * torch.rand((), generator=generator).item()


def synthesize(source, options, generator):
    """Return one equal-exposure Bayer burst with known translation geometry."""
    size, scale = int(options["native_size"]), int(options.get("scale", 2))
    frames = int(options.get("frames", 4))
    if size < 8 or size % 2 or scale not in (1, 2, 4) or not 1 <= frames <= 14:
        raise ValueError("need even native_size>=8, scale in {1,2,4}, 1<=frames<=14")
    max_shift = float(options.get("max_shift", 1.5))
    if max_shift < 0:
        raise ValueError("max_shift must be nonnegative")
    hr = size * scale
    sigma_range = options.get("blur_sigma", [0.2, 0.8])
    sigma = _uniform(generator, *sigma_range)
    margin = int(np.ceil((max_shift + 2) * scale + 3 * sigma))
    extent = hr + 2 * margin
    if min(source.shape[-2:]) < extent:
        # Upsizing tiny smoke fixtures is explicit metadata, never hidden.
        factor = extent / min(source.shape[-2:])
        source = F.interpolate(source[None], size=[int(np.ceil(d * factor)) for d in source.shape[-2:]], mode="bilinear", align_corners=False)[0]
        resized = True
    else:
        resized = False
    oy = int(torch.randint(source.shape[-2] - extent + 1, (), generator=generator))
    ox = int(torch.randint(source.shape[-1] - extent + 1, (), generator=generator))
    scene = source[:, oy:oy + extent, ox:ox + extent].clone()
    # Random dihedral augmentation precedes the CFA, keeping its phase RGGB.
    if options.get("augment", True):
        scene = torch.rot90(scene, int(torch.randint(4, (), generator=generator)), (-2, -1))
        if torch.rand((), generator=generator) > .5:
            scene = scene.flip(-1)
    ev = _uniform(generator, *options.get("exposure_ev", [-3.0, 0.0]))
    gains_range = options.get("color_gain", [1.0, 2.0])
    color_gains = torch.tensor([_uniform(generator, *gains_range), 1., _uniform(generator, *gains_range)])
    scene = gaussian_blur(scene * (2 ** ev) / color_gains[:, None, None], sigma)
    target = scene[:, margin:margin + hr, margin:margin + hr].clone()
    shifts = (torch.rand(frames, 2, generator=generator) * 2 - 1) * max_shift
    shifts[0] = 0
    yy, xx = torch.meshgrid(torch.arange(hr), torch.arange(hr), indexing="ij")
    grids = torch.stack((xx, yy), -1)[None].float() + margin + shifts[:, None, None, :] * scale
    grids = 2 * (grids + .5) / extent - 1
    samples = F.grid_sample(scene[None].expand(frames, -1, -1, -1), grids, mode="bilinear", padding_mode="border", align_corners=False)
    # Integrate scale x scale HR cells per native sensel before CFA sampling.
    rgb_native = F.avg_pool2d(samples, scale)
    raw_clean = mosaic_rggb(rgb_native)
    shot = _uniform(generator, *options.get("shot_noise", [0.0001, 0.004]))
    read = _uniform(generator, *options.get("read_noise", [0.0001, 0.003]))
    if min(shot, read) < 0:
        raise ValueError("noise parameters must be nonnegative")
    # Poisson shot noise + Gaussian read noise; read denotes standard deviation.
    raw = torch.poisson(raw_clean.clamp_min(0) / shot, generator=generator) * shot if shot > 0 else raw_clean.clone()
    raw += torch.randn(raw.shape, generator=generator) * read
    saturation = ((raw < 0) | (raw > 1)).float().mean().item()
    if options.get("clip_sensor", True):
        raw = raw.clamp(0, 1)
    bits = int(options.get("quantization_bits", 14))
    if bits:
        if not 1 <= bits <= 24:
            raise ValueError("quantization_bits must be 0 or 1..24")
        raw = torch.round(raw * (2 ** bits - 1)) / (2 ** bits - 1)
    return {
        "raw": raw.float(), "shifts": shifts, "target": target.float(),
        "metadata": {
            "protocol": "inferred-synthetic-v1", "motion": "oracle_translation", "cfa": "RGGB",
            "scale_native": scale, "exposure_ev": ev, "color_gains": color_gains.tolist(),
            "blur_sigma_hr": sigma, "shot_coefficient": shot, "read_std": read,
            "quantization_bits": bits, "clip_sensor": bool(options.get("clip_sensor", True)),
            "out_of_range_fraction_before_clipping": saturation, "source_resized": resized,
            "crop_xy": [ox, oy], "shifts_xy_native": shifts.tolist(),
            "black_level": 0., "white_level": 1.,
        },
    }


class SyntheticBurstDataset(Dataset):
    def __init__(self, manifest, split, options, seed=1234, samples_per_scene=1):
        self.rows = read_manifest(manifest, split)
        self.options = dict(options)
        self.seed, self.epoch = int(seed), 0
        self.samples_per_scene = int(samples_per_scene)
        if self.samples_per_scene < 1:
            raise ValueError("samples_per_scene must be >=1")

    def __len__(self):
        return len(self.rows) * self.samples_per_scene

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        # Stable across workers/platforms and independent of Python's salted hash.
        token = f"{self.seed}/{self.epoch}/{index}/{row['sha256']}".encode()
        sample_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2 ** 63 - 1)
        generator = torch.Generator().manual_seed(sample_seed)
        sample = synthesize(load_rgb(row["resolved_path"], row["encoding"]), self.options, generator)
        sample["metadata"].update(source_sha256=row["sha256"], scene_id=row["scene_id"], sample_seed=sample_seed, split=row["split"])
        # JSON keeps variable metadata out of DataLoader's numeric collation.
        sample["metadata"] = json.dumps(sample["metadata"], sort_keys=True)
        return sample
