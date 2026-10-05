"""Explicit electron/DN camera profiles; defaults are analytic, not measured."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .utils import save_json, sha256


def analytic_profile():
    return {
        "schema": "jsr-camera-v1", "name": "analytic-full-frame-proxy",
        "provenance": {"kind": "analytic_inferred", "note": "Not the author's PTC library or a measured camera."},
        "channel_transmission": [0.70, 1.0, 0.55],
        "iso": {str(iso): {"gain_e_per_dn": 4.0 * 100 / iso, "read_noise_e": 3.0,
                           "black_dn": 512., "white_dn": 16383., "full_well_e": 60000.}
                for iso in (100, 200, 400, 800)},
    }


def validate_profile(profile):
    if profile.get("schema") != "jsr-camera-v1" or profile.get("provenance", {}).get("kind") not in ("analytic_inferred", "fitted_input"):
        raise ValueError("camera profile requires jsr-camera-v1 and explicit analytic_inferred/fitted_input provenance")
    transmission = np.asarray(profile.get("channel_transmission"), dtype=float)
    if transmission.shape != (3,) or not np.isfinite(transmission).all() or np.any(transmission <= 0) or np.any(transmission > 1):
        raise ValueError("channel_transmission must be three finite values in (0,1]")
    if not profile.get("iso"):
        raise ValueError("camera profile needs ISO entries")
    for key, entry in profile["iso"].items():
        if str(int(key)) != key or int(key) <= 0:
            raise ValueError("ISO keys must be positive integer strings")
        values = [entry.get(k, float("nan")) for k in ("gain_e_per_dn", "read_noise_e", "black_dn", "white_dn", "full_well_e")]
        if not np.isfinite(values).all() or values[0] <= 0 or values[1] < 0 or values[3] <= values[2] or values[4] <= 0:
            raise ValueError(f"invalid electron/DN units for ISO {key}")
    return profile


def load_profile(path=None):
    profile = analytic_profile() if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_profile(profile)


def profile_identity(profile):
    import hashlib
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_profile(csv_path, output, dark=None, black_dn=512., white_dn=16383., full_well_e=60000., transmission=(.7, 1., .55)):
    """CSV: iso,mean_dn,variance_dn2 (dark-subtracted mean, temporal variance).

    slope is DN/e, inverse slope is e/DN. Intercept includes quantization;
    subtract 1/12 DN² before converting read noise. Input is NOT independently
    verified measured data. Optional dark stack [T,H,W] is raw DN, one ISO only.
    When supplied, its temporal variance fixes the intercept (with a 1/12 DN²
    floor for the assumed uniform quantizer), avoiding an unreliable
    extrapolated intercept from bright, shot-noise-dominated flats. Three
    feasible weighted least-squares updates estimate a positive slope. Weights
    are inverse squared predicted variance, assuming equal sample counts per
    PTC row. The unconstrained line remains available as a quality diagnostic.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"iso", "mean_dn", "variance_dn2"}.issubset(rows[0]):
        raise ValueError("PTC CSV requires iso,mean_dn,variance_dn2 columns")
    groups = {}
    for row in rows:
        iso = int(row["iso"])
        mean, variance = float(row["mean_dn"]), float(row["variance_dn2"])
        if iso <= 0 or not np.isfinite([mean, variance]).all() or mean < 0 or variance < 0:
            raise ValueError("PTC rows require positive ISO, finite nonnegative mean and temporal variance")
        groups.setdefault(str(iso), []).append((mean, variance))
    dark_stats = None
    if dark is not None:
        if len(groups) != 1:
            raise ValueError("one dark stack can calibrate only one ISO CSV")
        stack = np.load(dark, allow_pickle=False).astype(np.float64)
        if stack.ndim != 3 or stack.shape[0] < 3 or not np.isfinite(stack).all():
            raise ValueError("dark stack must be finite raw-DN [T>=3,H,W]")
        # Temporal variance at each pixel excludes spatial DSNU variation.
        dark_stats = {"black_dn": float(stack.mean()), "temporal_variance_dn2": float(stack.var(axis=0, ddof=1).mean()),
                      "dsnu_std_dn": float(stack.mean(axis=0).std()), "sha256": sha256(dark), "frames": stack.shape[0]}
    profile = {"schema": "jsr-camera-v1", "name": "fitted-input-profile",
               "provenance": {"kind": "fitted_input", "csv_sha256": sha256(csv_path),
                              "note": "Input provenance is user supplied, not independently verified measured camera data."},
               "channel_transmission": list(transmission), "iso": {}, "fit_diagnostics": {}}
    for iso, values in groups.items():
        array = np.asarray(values, dtype=np.float64)
        if len(values) < 3 or np.ptp(array[:, 0]) <= 0:
            raise ValueError("PTC requires at least three distinct illumination samples per ISO")
        unconstrained_slope, unconstrained_intercept = np.polyfit(array[:, 0], array[:, 1], 1)
        if dark_stats is None:
            slope, intercept = unconstrained_slope, unconstrained_intercept
            if slope <= 0 or intercept < 0:
                raise ValueError("PTC fit must have positive slope and nonnegative intercept; select its linear unsaturated interval")
            method = "unconstrained_linear_fit"
        else:
            intercept = max(dark_stats["temporal_variance_dn2"], 1 / 12)
            x, y = array[:, 0], array[:, 1]
            slope = np.dot(x, y - intercept) / np.dot(x, x)
            for _ in range(3):
                if not np.isfinite(slope) or slope <= 0:
                    raise ValueError("dark-constrained PTC needs a positive slope; check matched dark/flat units and unsaturated interval")
                predicted = slope * x + intercept
                weight = 1 / predicted ** 2
                slope = np.dot(weight * x, y - intercept) / np.dot(weight * x, x)
            if not np.isfinite(slope) or slope <= 0:
                raise ValueError("dark-constrained PTC needs a positive slope; check matched dark/flat units and unsaturated interval")
            method = "dark_fixed_intercept_feasible_wls"
        gain = 1 / slope
        read_var = intercept
        profile["iso"][iso] = {"gain_e_per_dn": float(gain), "read_noise_e": float(np.sqrt(max(0, read_var - 1 / 12)) * gain),
                               "black_dn": dark_stats["black_dn"] if dark_stats else float(black_dn),
                               "white_dn": float(white_dn), "full_well_e": float(full_well_e)}
        residual = array[:, 1] - (slope * array[:, 0] + intercept)
        profile["fit_diagnostics"][iso] = {
            "samples": len(values), "method": method, "slope_dn_per_e": float(slope),
            "intercept_dn2": float(intercept), "rmse_dn2": float(np.sqrt(np.mean(residual ** 2))),
            "relative_rmse": float(np.sqrt(np.mean((residual / np.maximum(slope * array[:, 0] + intercept, 1e-12)) ** 2))),
            "unconstrained_slope_dn_per_e": float(unconstrained_slope),
            "unconstrained_intercept_dn2": float(unconstrained_intercept),
            "quantization_variance_dn2": 1 / 12,
        }
        if dark_stats is not None:
            profile["fit_diagnostics"][iso].update(
                intercept_source="measured temporal dark variance with uniform-quantizer floor",
                quantization_floor_applied=dark_stats["temporal_variance_dn2"] < 1 / 12,
                weighting_assumption="equal variance-estimate degrees of freedom across PTC levels")
    if dark_stats:
        profile["provenance"]["dark"] = dark_stats
    validate_profile(profile)
    save_json(output, profile)
    return profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv")
    p.add_argument("--output", required=True)
    p.add_argument("--dark")
    p.add_argument("--black-dn", type=float, default=512.)
    p.add_argument("--white-dn", type=float, default=16383.)
    p.add_argument("--full-well-e", type=float, default=60000.)
    args = p.parse_args()
    if args.csv:
        result = fit_profile(args.csv, args.output, args.dark, args.black_dn, args.white_dn, args.full_well_e)
    else:
        if args.dark:
            p.error("--dark requires --csv")
        result = analytic_profile()
        save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
