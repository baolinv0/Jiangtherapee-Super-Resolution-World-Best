# Google-inspired Spectral Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the new spectral protocol reconstruct the reference-frame optical image after native pixel-area response on a dense 2× grid, with explicit target-stage ablations and preserved v2 behavior.

**Architecture:** `spectral-camera-v3` resolves `post_pixel` by default and shares the reference frame's 61-band optical path with RAW. Generalize pixel integration to separate output spacing from native aperture size, while preserving the historical v2 random stream and pre-optics target. Carry target identity through training, export and validation.

**Tech Stack:** Python, PyTorch, NumPy, pytest, YAML.

**Spec:** [GOOGLE_TARGET.md](../../GOOGLE_TARGET.md)

## Global Constraints

- v3 allows `post_pixel`, `post_optics`, `pre_optics`; v2 is fixed to `pre_optics`.
- Dense 2× sampling does not shrink the native pixel footprint or integrate it twice.
- Targets use D65-normalized camera RGB at reference exposure and exclude CFA, per-frame exposure, transmission, noise, clipping and quantization.
- Same source, seed and observation options must preserve identical RAW across target stages and preserve historical v2 RNG behavior.
- Preserve old v2 strict resume; reject a changed v3 target on strict resume.
- Keep the K7 Transformer architecture. Equal-exposure control is independent engineering; HDR recipes remain task extensions.
- User has authorized implementation, validation and updating the existing `reproduce/jsr-pytorch-engineering` branch. Execute without an additional design-approval pause; do not claim Google/JSR author reproduction.

## Review Focus

- Dense center phase differs from native center phase; test formula/linear ramp rather than assuming `target[..., ::2, ::2]` equals native data.
- Native footprint must stay fixed at dense sampling; use an independent integration reference and a shrinking-footprint counterexample.
- Constant scenes cannot establish sensitivity to optics; use structured scenes and wavelength/field-dependent PSFs.
- New metadata can break legacy identity equality; test historical v2 resume and cross-stage rejection separately.
- Exposure/clipping/noise interventions can accidentally alter GT or RNG; pin RAW equivalence across stages and GT invariance to observation-only changes.

### Task 1: Core synthesis and target identity

**Files:** Modify `reproduction/jsr_repro/optics.py`, `reproduction/jsr_repro/spectral_data.py`, protocol dispatch and checkpoint/data-identity consumers under `reproduction/jsr_repro/` as needed.

**Interfaces:** Existing `sensor_integrate(scene, shifts, native_size, margin, fill_factor=.95, quadrature=4, output_scale=1)` calls remain compatible; `output_scale=2` doubles output size and halves sample spacing while retaining native aperture width. `resolve_target_stage(options)` validates protocol/stage semantics. `synthesize_spectral(source, options, seed, profile=None, assets=None)` resolves protocol and stage from `options` and returns the existing sample keys with explicit target semantics in metadata.

- [x] Add regressions for v2 historical output and v3 stage defaults, invalid stages and v2 stage conflicts; record their expected failures before implementation.
- [x] Add independent-coordinate/integration tests, fixed-footprint counterexamples and observation invariants, then implement shared reference optics plus the three target branches.
- [x] Preserve the v2 RNG namespace and legacy identity shape; add v3 target identity to export, training, resume and evaluation paths.
- [x] Run the owning unit tests and confirm legacy callers remain compatible.

### Task 2: Validation and migration regressions

**Files:** Modify `reproduction/jsr_repro/validate_spectral.py` and relevant tests under `tests/`.

**Interfaces:** `validate(output, config='configs/spectral_smoke.yaml')` reports resolved protocol/stage and checks stage-appropriate invariants. It must not require optics-invariant GT for `post_pixel` or `post_optics`.

- [x] Test the three-stage optics/fill-factor behavior, fixed-seed RAW equivalence and target invariance to observation-only changes.
- [x] Test target metadata/export identity, strict v2 resume without new fields and rejection of v3 stage changes.
- [x] Update validation labels/checks for actual target semantics, retaining historical JSON evidence unchanged.
- [x] Run `python -m pytest -q` and both documented v3 smoke validations into fresh directories; report failures rather than inventing results.

### Task 3: Integration, documentation and branch update

**Files:** Create `docs/GOOGLE_TARGET.md`, this plan and `configs/google_equal_exposure_smoke.yaml`; modify `README.md`, `docs/SPECTRAL_DATA.md`, `configs/spectral_smoke.yaml`, `configs/spectral_k7.yaml`, and comments in `configs/revision_explicit_smoke.yaml`.

**Interfaces:** New/default spectral configs explicitly select v3 / `post_pixel`. The equal-exposure control matches spectral smoke architecture with EV interval `[0,0]`, gain `[0.05,0.4]` and a distinct output directory; revision explicit smoke retains v2.

- [x] Publish the mathematical target/coordinate contract, evidence boundary, migration policy, three choices and runnable commands.
- [x] Validate YAML config semantics and label existing numerical evidence as v2 historical.
- [x] Review the combined diff and fresh test/smoke evidence; correct mismatches before staging.
- [x] Prepare the verified delivery for the authorized `reproduce/jsr-pytorch-engineering` branch; the final delivery response records the actual remote commit and read-back result.

## Execution evidence

Completed implementation and validation evidence: [110 passing tests, two 8-step CPU runs, legacy parity and real old-checkpoint resume](../../google-target-validation/VALIDATION.md).

Decisions: v3 is a new target protocol, while v2 keeps its RNG and identity; this costs one additional supported recipe but avoids silently changing old training. Dense samples retain the native footprint and existing half-pixel phase. The default K7 HDR recipe is an application extension; a separate equal-exposure control provides the Google-inspired stage baseline. These are explicit engineering choices, not claims about unpublished author intent.

Independent review findings were resolved by using a distinct v3 validation output directory and adding an independent full-spectral/four-field integral regression. No deferred review findings. GitHub publication follows these local checks and is confirmed in the delivery response.
