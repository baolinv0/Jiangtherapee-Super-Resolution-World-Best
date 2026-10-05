# Speech-inspired JSR engineering extension

status: FROZEN
baseline_local: 760f12a
baseline_remote: fa3ea8ec05d84f49695e9010678ddd2f7c950d2e
target_branch: reproduce/jsr-pytorch-engineering

## Outcome

Deliver a runnable, separately identified seven-frame Transformer RAW burst SR/HDR pipeline that implements the mechanisms described in the user-provided 2026-10-03 Zhihu speech, with camera simulation, calibration interfaces, training, evaluation, inference, configuration and numerical evidence. Preserve the existing public-v9.8-module reproduction. Explain which mechanisms match the speech and which formulas/parameters are independently inferred. This is an engineering reproduction, not recovery of undisclosed author code, training data, or a quality claim.

## Scope

1. A new seven-frame late-fusion attention model and dedicated checkpoint/CLI path.
2. A new deterministic camera-simulation dataset using the existing leak-checked scene manifests. Both equal-exposure and bracketed exposure are supported. The existing v1 dataset remains available unchanged.
3. Validated analytic camera profiles spanning ISO 100/200/400/800, optional fitted PTC/dark inputs, three-band physical diffraction PSFs and pixel integration, channel transmission and real signal saturation.
4. CPU smoke training, checkpoint resume, held-out synthetic evaluation, calibrated burst archive inference, 16-bit linear RGB export with explicit encoding scale, tests, reproducible verification script and CI extension.
5. Documentation, including source/conformance table, data/model flow, unknowns and limitations. Update the existing PR after build and independent review.

## Non-goals

- No fabricated author PTC library, lens measurements, spectral training data, original Transformer weights, preprint, benchmark scores or mobile performance.
- No exact replication of an unpublished attention layout, cyclic fusion equation, edge-preserving amplitude formula or phase expansion head.
- No dense flow/rolling shutter, moving-object deghost guarantee, optical field-dependent lens calibration, production tiling, color-managed ISP, or proof of zero color bias/SOTA.
- No full GPU training or real-camera quantitative benchmark in this CPU-only environment.
- No merging into main or modification of original source/weights/assets.

## Frozen Design

### Identity and compatibility

Use an unmistakable implementation ID such as `speech-inspired-transformer-v1`. New modules/CLIs may be separate from v1 rather than changing its checkpoint contract. The original 23 tests, public-module numerical compatibility and v1 validation must remain valid. Never load public Controller/RefineNet weights into the new Transformer. Builder owns all new implementation, tests, configs, workflow and explanatory repo docs; root owns contract/plan, review record and publication. Builder is not alone in the codebase and must not revert other edits. Do not edit CONTRACT.md or PLAN.md.

### Dataset and radiometric units

- Input source: existing float linear RGB / inverse-sRGB RGB proxy. Permit an explicit HDR scene gain so unclipped target radiance can exceed reference white=1. Source-clipped highlights are still unavailable. Do not label RGB spatial enlargement as hyperspectral or claim recovery of spectral ground truth; document the ambiguity of “全光谱”.
- Seven frames: reference frame 0 has EV=0; remaining EVs are a deterministic ordering of {-3,-2,-1,+1,+2,+3} times an interval delta sampled/configured in [0,1]. Delta=0 gives equal exposure; delta=0.3 gives the example's seven-shot bracket. Clearly distinguish interval from total six-interval span.
- Use one shared camera/ISO/optics profile per burst. Reference camera radiance is before the optional per-channel transmission; noiseless sensor signal is radiance times exposure ratio times channel transmission. Targets use the same documented reference radiance/color units. No arbitrary per-image gain fitting during metrics.
- Noise is simulated in electrons/ADU with explicit gain_e_per_dn, reference normalization span, read_noise_e, dark offset and full_well_e/ADC clip; the noise variance prior must be derived from these same parameters and returned to the model. Retain separate physical signal saturation masks (before random noise) and black/invalid masks. A low-end noise excursion must not masquerade as true high-light saturation.
- Default analytic profile covers ISO100-800, clearly marked synthetic/inferred and distinct from fitted input profiles. Validate positive units, finite values and provenance. A calibration CLI fits variance-versus-mean PTC tables per ISO from documented CSV columns; optionally accepts a dark-frame stack to estimate offset/read variance. Emit fit diagnostics and source hashes. Fitted arbitrary input is not independently verified measured camera data. Tests use known synthetic calibration data, labeled as such.
- Optical model uses wavelength-dependent diffraction for representative RGB bands, f-number in [2,8], pitch in [3,5.76] micrometers and area integration with configurable large fill factor. Airy or an explicitly justified diffraction approximation is acceptable, with nonnegative normalized kernels and reported truncation/units. Optional aberration proxy may be explicit, never described as a real prime-lens library. Dataset records all sampled parameters, source hash, scene split and random seed.
- Random handheld subpixel translations use the existing coordinate convention; alignment is either oracle translations or an explicitly documented global estimated baseline. Boundary valid masks prevent padded samples being interpreted as camera evidence. Return raw [K,1,H,W], shifts [K,2], exposure [K], channel transmission, variance and saturation/valid masks, target [3,2H,2W], plus JSON metadata.

### Model

- Fix native SR scale=2 and K=7. Pack RGGB into four half-resolution planes while preserving CFA offsets. Translate each plane into reference coordinates using a documented tested sign/center convention; account for variance interpolation and conservative saturation/valid-mask propagation.
- Form exposure/transmission-corrected radiance, noise and saturation/reliability features. Derive an edge-preserving local amplitude from dimensionless edge weights, normalize radiance and variance by amplitude and amplitude squared respectively, and restore amplitude at the output. Specify behavior for zero signal. Exposure metadata is dimensionless; noise/saturation states legitimately change processing. Conditional gain equivariance is tested only with fixed geometry/masks and covariance scaled consistently.
- Retain [B,7,C,h,w] features through at least two encoder scales and their corresponding decoder scales with per-frame skips. Include actual spatial self-attention, cross-frame attention and explicit cross-frame feature differences. Do not collapse the frame dimension in the backbone. Record shape traces for evidence.
- At decoder end only, reduce seven features using learned adjacent cyclic pairing with a clearly documented rotation/order and unpaired-frame policy, reaching 7→4→2→1. Exact pairing equation is an engineering assumption.
- Apply two successive 2x latent feature upsampling stages from packed Bayer resolution, producing 2x native RGB. Phase expansion operates on shared latent features; RGB projection occurs after expansion rather than separate per-color phase heads. This implements a shared-phase concept, not a proof that all chroma artifacts disappear.
- Expose a corrected linear merge baseline and output finite unclipped linear RGB (including >1 highlights). No final [0,1] clamp. Zero signal with zero noise returns zero. Metadata shape/value errors fail with actionable messages.

### Training and user interfaces

- Provide smoke and formal example configs. Formal settings are proposals and not described as the author's settings. Train on the new dataset, with explicitly specified linear RGB/chroma/gradient loss and a high-light-region term or reporting sufficient to exercise real saturation. No claim that a few smoke steps learn complete reconstruction.
- Checkpoint has distinct implementation ID, model/config, optimizer/schedule, RNG, dataset/profile identity and progress state. Resume on CPU matches uninterrupted training; reject changed calibration/config/source identity. Do not merely reuse a v1 checkpoint tag for the new model.
- CLI evaluation reports held-out fixed-reference linear metrics and highlight/saturation evidence; supports oracle and estimated alignment with exposure correction before estimation. Clearly label synthetic proxy, fixed range, crop, no gain fit.
- CLI inference consumes an NPZ burst with required exposure/calibration/noise/saturation metadata, rejects incompatible/missing fields, and saves float linear RGB plus a real uint16 RGB TIFF (or equivalent RGB16 format) and JSON describing a user/configurable radiance white scale and clipping fraction. A 16-bit file is storage precision, not validated 16-bit effective image information. Preserve the memory guard.
- Provide a reproducible archive preparation path from the new dataset for end-to-end verification. Real RAW import/calibration layout can be documented; no requirement to run rawpy or download a real dataset.

## Robustness Envelope

CPU float32, Python3.11; native even square crops 16 and 32/64, exactly seven frames, scale2, finite nonnegative signal; EV interval0..1; physical parameter ranges above; partially saturated scenes with at least some usable observations. All-saturated regions remain underdetermined and must be identified/documented rather than claimed recovered. Synthetic translations within a documented bound; no occlusion or object motion. Config/schema errors, NaN/Inf, invalid physical units, wrong K, missing metadata and excessive full-image size must fail clearly. For equivariance tests only: no clipping transitions, finite positive gain and consistent scaling of noise covariance.

## Required Evidence

- Full existing test suite plus new semantic tests: physical units/PTC recovery; deterministic/leak-guard dataset; EV=0 and 0.3/1 brackets; true unequal-channel saturation/HDR target; PSF normalization/range dependence/pixel integration; metadata actually affects model; frame trace remains7 until late fusion; shared-head order; finite gradients; zero and conditional gain scaling; coordinate convention; checkpoint/inference metadata contracts; resume equality; uint16 file round trip/scale.
- A one-command offline validation run with a small procedural manifest, at least16 optimizer steps, held-out oracle and estimated reports, saved calibration fit, burst NPZ, model checkpoint, output TIFF+NPY, architecture trace and JSON results. Evidence is smoke capability, not benchmark quality. Run the existing v1 smoke validation as a regression.
- One larger configuration forward/backward smoke (seven frames, native>=32 and nontrivial widths); no need to run formal100k training.
- At least one saved visual showing short/reference/long inputs, saturation/variance or amplitude and target/prediction. Record rendering/exposure scale and include visual artifact index.
- IMPLEMENTATION_REPORT.md lists changed files, commands, results, assumptions and limitations; all committed evidence is compact and excludes private absolute paths/large checkpoints.
- Independent reviewer inspects frozen requirements, code, meaningful tests, logs and visual evidence, returning PASS_TO_NEXT_STAGE or REVISE_BLOCKERS_ONLY with reproducible blockers.

## Completion Bar

All scoped components exist and their documented commands run; original tests and both offline pipeline validations pass; required new numerical/visual evidence is recorded truthfully; independent review has no remaining blocker after at most one repair cycle; updated PR contains code/docs/evidence without touching main. Root verifies remote tree/file identity and reports CI state without implying unrun full training or author-quality equivalence.

## Blocker Definition

A blocker is a reproducible violation of this contract inside its envelope: an unrunnable claimed command; incorrect radiometric/coordinate mapping; model not consuming exposure/noise/saturation or collapsing frames early; invalid checkpoint resume; data leakage; silently fabricated calibration/source provenance; claimed evidence not produced; meaningful old regressions; or missing scoped component. Accuracy gains/SOTA, unpublished author parameters, real-data access and out-of-scope production concerns are not blockers. A change to frozen scope requires a reproducible in-scope contradiction and a minimal CHANGE_REQUEST; ordinary implementation choices stay within the contract.
