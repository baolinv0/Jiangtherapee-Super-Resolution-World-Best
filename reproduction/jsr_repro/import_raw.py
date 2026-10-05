"""Optional native RAW/DNG -> normalized RGGB archive; no WB or tone mapping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .utils import sha256


def import_files(paths, output, crop=None):
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError('Install the optional RAW dependency: pip install -e ".[raw]"') from exc
    if not 1 <= len(paths) <= 14:
        raise ValueError("expected 1..14 RAW files in reference-first order")
    frames, metadata = [], []
    for path in paths:
        with rawpy.imread(str(path)) as raw:
            if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
                raise ValueError("only 2x2 Bayer RAW is supported")
            names = raw.color_desc.decode("ascii")
            pattern = np.array([[names[int(i)] for i in line] for line in raw.raw_pattern])
            positions = np.argwhere(pattern == "R")
            if len(positions) != 1:
                raise ValueError("unsupported CFA")
            oy, ox = (int(v) for v in positions[0])
            if pattern[(oy + 1) % 2, (ox + 1) % 2] != "B" or pattern[oy, (ox + 1) % 2] != "G" or pattern[(oy + 1) % 2, ox] != "G":
                raise ValueError("only RGB Bayer CFA patterns are supported")
            values = raw.raw_image_visible.astype(np.float32)
            colors = raw.raw_colors_visible
            black = np.asarray(raw.black_level_per_channel, np.float32)
            white = np.asarray(raw.camera_white_level_per_channel if raw.camera_white_level_per_channel is not None else [raw.white_level] * 4, np.float32)
            if np.any(white <= black):
                raise ValueError("white level must exceed black level")
            values = (values - black[colors]) / (white[colors] - black[colors])
            values = values[oy:oy + (values.shape[0] - oy) // 2 * 2, ox:ox + (values.shape[1] - ox) // 2 * 2]
            if crop is not None:
                cx, cy, cw, ch = crop
                if min(cx, cy) < 0 or min(cw, ch) < 8 or any(v % 2 for v in crop) or cx + cw > values.shape[1] or cy + ch > values.shape[0]:
                    raise ValueError("crop X Y WIDTH HEIGHT must be even, in bounds, width/height>=8")
                values = values[cy:cy + ch, cx:cx + cw]
            # Preserve negative read noise after black subtraction.
            frames.append(values[None])
            metadata.append({"source": str(path), "sha256": sha256(path), "black_per_channel": black.tolist(), "white_per_channel": white.tolist(), "crop_origin_yx": [oy, ox], "requested_crop_xywh": crop, "original_pattern": pattern.tolist(), "white_balance_applied": False, "orientation": "unrotated sensor coordinates"})
    if len({f.shape for f in frames}) != 1 or len({tuple(m["crop_origin_yx"]) for m in metadata}) != 1:
        raise ValueError("burst dimensions and CFA origins must match")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, raw=np.stack(frames), metadata=json.dumps({"cfa": "RGGB", "frames": metadata, "exposure": "must be equal across frames; not inferred by importer"}))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="even sensor crop, e.g. 200 200 128 128; useful for the full-image reference implementation")
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    print(import_files(args.files, args.output, args.crop))


if __name__ == "__main__":
    main()
