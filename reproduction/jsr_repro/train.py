"""Supervised training with exact CPU resume and explicitly inferred losses."""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SyntheticBurstDataset, verify_manifest
from .metrics import reconstruction_loss, image_metrics
from .model import JSRModel
from .utils import load_config, seed_all, device_from, sha256, save_json


def capture_rng():
    ns = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "python": random.getstate(), "numpy": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format_version") != 1 or state.get("implementation") != "inferred-jsr-v1":
        raise ValueError("unsupported reproduction checkpoint")
    model = JSRModel(**state["config"]["model"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, state


@torch.no_grad()
def validate_model(model, dataset, device, border=4, output_key="rgb"):
    was_training = model.training
    model.eval()
    values = []
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        out = model(sample["raw"].to(device), sample["shifts"].to(device))
        values.append(image_metrics(out[output_key], sample["target"].to(device), border))
    model.train(was_training)
    return {k: float(np.mean([v[k] for v in values if v[k] is not None])) if any(v[k] is not None for v in values) else None for k in values[0]}


def run_training(config, resume=None, stop_after=None, initialize=None):
    cfg = json.loads(json.dumps(config))
    seed = int(cfg.get("seed", 1234))
    seed_all(seed, int(cfg.get("threads", 2)))
    device = device_from(cfg.get("device", "cpu"))
    data, training = cfg["data"], cfg["train"]
    if data["options"].get("scale", 2) != cfg["model"].get("scale", 2):
        raise ValueError("data and model scales must match")
    if int(training["steps"]) < 1 or int(training.get("batch_size", 1)) < 1:
        raise ValueError("steps and batch_size must be positive")
    if int(training.get("validate_every", 10)) < 1:
        raise ValueError("validate_every must be positive")
    verify_manifest(data["manifest"])
    manifest_hash = sha256(data["manifest"])
    train_data = SyntheticBurstDataset(data["manifest"], "train", data["options"], seed, data.get("samples_per_scene", 1))
    val_data = SyntheticBurstDataset(data["manifest"], "val", data["options"], seed + 1)
    model = JSRModel(**cfg["model"]).to(device)
    stage = training.get("stage", "joint")
    if stage not in ("joint", "controller", "refine"):
        raise ValueError("stage must be joint, controller or refine")
    for name, param in model.named_parameters():
        if stage == "controller" and "refine" in name:
            param.requires_grad_(False)
        if stage == "refine" and "controller" in name:
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(training["lr"]), weight_decay=float(training.get("weight_decay", 0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["steps"]), eta_min=float(training.get("min_lr", 0)))
    step, epoch, batch_cursor, best = 0, 0, 0, -float("inf")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if resume and initialize:
        raise ValueError("--resume and --initialize are mutually exclusive")
    if initialize:
        initial_model, initial_state = load_checkpoint(initialize, device)
        if initial_state["config"]["model"] != cfg["model"]:
            raise ValueError("initial checkpoint model configuration differs")
        model.load_state_dict(initial_model.state_dict(), strict=True)
        save_json(output / "initialization.json", {"checkpoint_sha256": sha256(initialize), "stage": stage, "optimizer_reset": True})
    if resume:
        _, state = load_checkpoint(resume, device)
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("resume manifest changed")
        for key in ("model", "data", "train", "seed"):
            if state["config"].get(key) != cfg.get(key):
                raise ValueError(f"resume {key} configuration differs; use the original config")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, epoch, batch_cursor = state["step"], state["epoch"], state["batch_cursor"]
        best = state["best_val_psnr"]
        restore_rng(state["rng"])
    elif (output / "last.pt").exists():
        raise FileExistsError("output already contains last.pt; use --resume or a new output")
    save_json(output / "config.json", cfg)
    save_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device), "platform": platform.platform(), "deterministic_algorithms": True, "manifest_sha256": manifest_hash})
    end_step = min(int(training["steps"]), int(stop_after)) if stop_after is not None else int(training["steps"])
    if end_step <= step:
        raise ValueError("requested stopping step must exceed the checkpoint step")
    border = int(training.get("crop_border", 4))
    losses = dict(border=border, chroma_weight=float(training.get("chroma_weight", .1)), gradient_weight=float(training.get("gradient_weight", .05)))
    history = []
    started = time.perf_counter()
    model.train()
    while step < end_step:
        train_data.set_epoch(epoch)
        loader = DataLoader(train_data, batch_size=int(training.get("batch_size", 1)), shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed + epoch), drop_last=False)
        for batch_index, sample in enumerate(loader):
            if batch_index < batch_cursor:
                continue
            raw, shifts, target = (sample[k].to(device) for k in ("raw", "shifts", "target"))
            optimizer.zero_grad(set_to_none=True)
            result = model(raw, shifts)
            prediction = result["learned"] if stage == "controller" else result["rgb"]
            loss = reconstruction_loss(prediction, target, **losses)
            if stage == "joint":
                loss = loss + float(training.get("learned_weight", .1)) * reconstruction_loss(result["learned"], target, **losses)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(training.get("clip_grad", 1.0)), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            next_epoch, next_batch = (epoch + 1, 0) if batch_index + 1 == len(loader) else (epoch, batch_index + 1)
            record = {"step": step, "epoch": epoch, "loss": loss.item(), "grad_norm": norm.item(), "lr": optimizer.param_groups[0]["lr"]}
            do_validate = step % int(training.get("validate_every", 10)) == 0 or step == end_step
            improved = False
            if do_validate:
                metrics = validate_model(model, val_data, device, border, "learned" if stage == "controller" else "rgb")
                record["validation"] = metrics
                improved = metrics["psnr_linear_db"] > best
                best = max(best, metrics["psnr_linear_db"])
                print(json.dumps(record), flush=True)
            history.append(record)
            with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if do_validate:
                checkpoint = {"format_version": 1, "implementation": "inferred-jsr-v1", "config": cfg, "manifest_sha256": manifest_hash, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": capture_rng(), "step": step, "epoch": next_epoch, "batch_cursor": next_batch, "best_val_psnr": best}
                temp = output / "last.tmp.pt"
                torch.save(checkpoint, temp)
                temp.replace(output / "last.pt")
                if improved:
                    torch.save(checkpoint, output / "best.pt")
            if step >= end_step:
                break
        epoch += 1
        batch_cursor = 0
    report = {"steps_completed": step, "wall_seconds": time.perf_counter() - started, "best_val_psnr": best, "checkpoint": str(output / "last.pt"), "scope": "engineering validation only; no original JSR quality claim"}
    save_json(output / "training_summary.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p.add_argument("--initialize", help="load model only for a new stage; resets optimizer/schedule")
    p.add_argument("--stop-after", type=int, help="stop at this global step while preserving the full LR schedule")
    args = p.parse_args()
    print(json.dumps(run_training(load_config(args.config), args.resume, args.stop_after, args.initialize), indent=2))


if __name__ == "__main__":
    main()
