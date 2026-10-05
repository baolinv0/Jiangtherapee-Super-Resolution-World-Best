"""Calibrated seven-frame NPZ inference and explicit radiance-scaled RGB16 TIFF."""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .data import linear_to_srgb
from .evaluate_transformer import estimated_shifts
from .train_transformer import load_checkpoint
from .transformer import FIELDS, validate_inputs
from .utils import device_from, save_json, seed_all, sha256


def write_rgb16_tiff(path,rgb):
    """Minimal uncompressed baseline TIFF, RGB interleaved, little-endian uint16.

    No camera color profile implied; keep explicit radiance scale in sidecar.
    """
    if rgb.dtype != np.uint16 or rgb.ndim!=3 or rgb.shape[-1]!=3:
        raise ValueError("TIFF requires HWC uint16 RGB")
    h,w,_ = rgb.shape
    tags = [(256,4,1,w),(257,4,1,h),(258,3,3,0),(259,3,1,1),(262,3,1,2),
            (273,4,1,0),(277,3,1,3),(278,4,1,h),(279,4,1,rgb.nbytes),(284,3,1,1),(339,3,3,0)]
    extras_offset = 8+2+len(tags)*12+4
    data_offset = extras_offset+12
    with Path(path).open("wb") as stream:
        stream.write(b"II"+struct.pack("<HI",42,8))
        stream.write(struct.pack("<H",len(tags)))
        for tag,kind,count,value in tags:
            if tag==258:
                value=extras_offset
            elif tag==339:
                value=extras_offset+6
            elif tag==273:
                value=data_offset
            stream.write(struct.pack("<HHII",tag,kind,count,value))
        stream.write(struct.pack("<I",0)+struct.pack("<HHHHHH",16,16,16,1,1,1))
        stream.write(rgb.astype("<u2",copy=False).tobytes())


def export_linear(output,rgb,white=8.):
    if not np.isfinite(white) or white<=0 or not np.isfinite(rgb).all():
        raise ValueError("export requires finite RGB and positive radiance white")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/"linear_rgb.npy",rgb.astype(np.float32))
    encoded=np.rint(np.clip(rgb/white,0,1)*65535).astype(np.uint16)
    write_rgb16_tiff(output/"linear_rgb16.tiff",encoded)
    preview=linear_to_srgb(torch.from_numpy(rgb.copy()).permute(2,0,1)/white).permute(1,2,0).numpy()
    Image.fromarray(np.rint(preview*255).astype(np.uint8)).save(output/"preview.png")
    return {"radiance_white":float(white),"decode_formula":"linear_radiance = uint16 / 65535 * radiance_white",
            "clipped_fraction":float(np.mean((rgb<0)|(rgb>white))),"tiff":"linear_rgb16.tiff","linear_float":"linear_rgb.npy",
            "preview":"sRGB diagnostic at radiance_white, no camera color/WB correction",
            "precision":"16-bit storage is not proof of 16 effective image bits"}


def read_burst(path,device,max_native_pixels=65536):
    with np.load(path,allow_pickle=False) as archive:
        if any(key not in archive for key in FIELDS) or "metadata" not in archive:
            raise ValueError("calibrated archive requires all Transformer fields and JSON metadata")
        arrays={key:np.asarray(archive[key],np.float32) for key in FIELDS}
        metadata=json.loads(str(archive["metadata"].item()))
    raw=arrays["raw"]
    if raw.ndim!=4 or raw.shape[0:2]!=(7,1) or max_native_pixels<1 or raw.shape[-2]*raw.shape[-1]>max_native_pixels:
        raise ValueError("archive needs K7,1,H,W within max-native-pixels; use an even CFA crop")
    if metadata.get("cfa")!="RGGB" or metadata.get("scale_native")!=2 or not re.fullmatch("[0-9a-f]{64}",str(metadata.get("profile_sha256",""))) or metadata.get("profile_provenance",{}).get("kind") not in ("analytic_inferred","fitted_input"):
        raise ValueError("archive metadata needs RGGB, scale2, profile_sha256 and profile_provenance")
    sample={key:torch.from_numpy(arrays[key])[None].to(device) for key in FIELDS}
    validate_inputs(sample)
    return sample,metadata


@torch.no_grad()
def infer(checkpoint,burst,output,alignment="estimated",device="cpu",max_native_pixels=65536,radiance_white=8.):
    seed_all(1234,2)
    device=device_from(device)
    model,state=load_checkpoint(checkpoint,device)
    model.eval()
    sample,metadata=read_burst(burst,device,max_native_pixels)
    if alignment=="estimated":
        sample["shifts"]=estimated_shifts(sample)
    elif alignment!="provided":
        raise ValueError("alignment must be estimated or provided")
    result=model(sample,trace=True)
    rgb=result["rgb"][0].permute(1,2,0).cpu().numpy()
    export=export_linear(output,rgb,radiance_white)
    report={"implementation":state["implementation"],"checkpoint_sha256":sha256(checkpoint),"burst_sha256":sha256(burst),
            "alignment":alignment,"shifts_xy_native":sample["shifts"][0].cpu().tolist(),"input_profile_sha256":metadata["profile_sha256"],
            "training_profile_sha256":state["data_identity"]["profile_sha256"],"calibration_domain_match":metadata["profile_sha256"]==state["data_identity"]["profile_sha256"],
            "trace":result["trace"],"output_shape_hwc":list(rgb.shape),"encoding":export,
            "all_saturated_or_invalid_packed_fraction":result["all_saturated_or_invalid_packed"].mean().item()}
    save_json(Path(output)/"inference.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","burst","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--alignment",choices=["estimated","provided"],default="estimated")
    p.add_argument("--device",default="cpu")
    p.add_argument("--max-native-pixels",type=int,default=65536)
    p.add_argument("--radiance-white",type=float,default=8.)
    print(json.dumps(infer(**vars(p.parse_args())),indent=2))


if __name__=="__main__":
    main()
