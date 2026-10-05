"""Build a reproducible scene-split manifest or procedural smoke fixtures."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .data import verify_manifest
from .utils import sha256, save_json


def procedural_image(seed, size=128):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size].astype(np.float32) / size
    out = np.zeros((size, size, 3), np.float32)
    for c in range(3):
        out[..., c] = .2 + .12 * x + .12 * y
        for _ in range(6):
            fx, fy = rng.uniform(0, 8, 2)
            out[..., c] += rng.uniform(.01, .06) * np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 6))
    for _ in range(7):
        x0, y0 = rng.integers(0, size - 15, 2)
        w, h = rng.integers(5, 25, 2)
        out[y0:y0 + h, x0:x0 + w] = rng.uniform(.03, .8, 3)
    return np.clip(out, 0, 1)


def build_manifest(source, output, encoding="srgb", seed=1234, group_by="file", procedural=0, layout="custom"):
    output = Path(output).resolve()
    if layout == "div2k" and group_by != "file":
        raise ValueError("DIV2K layout groups by its independent image IDs; use --group-by file")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.jsonl").exists():
        raise FileExistsError("manifest already exists; use a new output directory")
    if procedural:
        if layout != "custom":
            raise ValueError("procedural fixtures use the custom layout")
        if procedural < 6:
            raise ValueError("procedural fixtures need at least six independent scenes")
        source = output / "sources"
        source.mkdir(exist_ok=True)
        for i in range(procedural):
            np.save(source / f"scene_{i:04d}.npy", procedural_image(seed + i))
        encoding = "linear"
    elif source is None:
        raise ValueError("provide --source or --procedural")
    source = Path(source).resolve()
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
    paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in suffixes and p.is_file())
    entries, seen = [], set()
    for path in paths:
        if layout == "div2k":
            if path.parent.name not in ("DIV2K_train_HR", "DIV2K_valid_HR") or not path.stem.isdigit():
                continue
            number = int(path.stem)
            if not 1 <= number <= 900:
                continue
        digest = sha256(path)
        if digest in seen:
            continue
        seen.add(digest)
        rel = path.relative_to(source)
        scene_id = rel.as_posix() if group_by == "file" else rel.parent.as_posix()
        entries.append({"path": str(path), "sha256": digest, "scene_id": scene_id, "encoding": encoding})
    scenes = sorted({e["scene_id"] for e in entries})
    if len(scenes) < 3:
        raise ValueError("at least three independent scene groups are required")
    random.Random(seed).shuffle(scenes)
    n_val = max(1, len(scenes) // 10)
    n_test = max(1, len(scenes) // 10)
    splits = {s: "val" if i < n_val else "test" if i < n_val + n_test else "train" for i, s in enumerate(scenes)}
    for e in entries:
        if layout == "div2k":
            number = int(Path(e["path"]).stem)
            e["split"] = "train" if number <= 800 else "val" if number <= 850 else "test"
        else:
            e["split"] = splits[e["scene_id"]]
        # Fixtures are portable; external data remains an explicit local path.
        try:
            e["path"] = Path(e["path"]).relative_to(output).as_posix()
        except ValueError:
            pass
    manifest = output / "manifest.jsonl"
    if {e["split"] for e in entries} != {"train", "val", "test"}:
        raise ValueError("source must provide train, val and test groups")
    manifest.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    verify_manifest(manifest)
    save_json(output / "dataset.json", {"seed": seed, "layout": layout, "group_by": group_by, "encoding": encoding, "procedural_smoke_only": bool(procedural), "split_counts": {s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")}, "manifest_sha256": sha256(manifest)})
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoding", choices=["srgb", "linear"], default="srgb")
    p.add_argument("--group-by", choices=["file", "parent"], default="file", help="Use parent for multiple views/exposures of one scene")
    p.add_argument("--procedural", type=int, default=0, metavar="N")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--layout", choices=["custom", "div2k"], default="custom", help="div2k: official train 0001-0800; derived val 0801-0850 / test 0851-0900")
    args = p.parse_args()
    print(build_manifest(args.source, args.output, args.encoding, args.seed, args.group_by, args.procedural, args.layout))


if __name__ == "__main__":
    main()
