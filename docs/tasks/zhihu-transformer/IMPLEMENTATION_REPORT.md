# Implementation report

Date: 2026-10-05. Local baseline760f12a; remote baselinefa3ea8ec05d84f49695e9010678ddd2f7c950d2e. Branchreproduce/jsr-pytorch-engineering. ImplementationID`speech-inspired-transformer-v1`.

## Build/review separation

The preset Builder failed before execution because its fixedgpt-5.6 model is unsupported with this ChatGPT account. Additional spawn attempts encountered the agent thread limit. Root therefore implemented the frozen design; the existing independently running`sources_audit`agent performs read-only contract review. No claim that the failed preset built code or that a preset Reviewer ran. The scope/design/evidence contract remains unchanged; review is by a different agent from the implementer.

## Delivered components

- `calibration.py`: validated analytic/fitted-input provenance, ISO100/200/400/800 defaults, explicit e/DN conversion, temporal-noise PTC fit and dark stack, fit diagnostics/hashes. Default parameters are inferred, not measured; arbitrary fitted input is not independently verified real camera data.
- `optics.py`, `bracket_data.py`: three-band Airy diffraction with f/2..8 and pitch3..5.76um, normalized truncated kernels, fill-factor area quadrature, seven-shot bracket interval0..1EV, radiance target>1, electron/DN noise, physical pre-noise saturation, separate black states, observed-noise variance, scene/hash/seed metadata and portable burst archive.
- `transformer.py`: CFA-aware inverse warp with squared-coefficient variance and conservative masks; exposure/transmission normalization; edge-weighted homogeneous amplitude; spatial/cross-frame MHA and differences; per-frame multiscale skips with K7 throughout; latecyclic7→4→2→1; two shared-latent phase expansions followed by RGB projection; unclipped radiance and unrecoverable-region marker.
- `train_transformer.py`, `evaluate_transformer.py`, `infer_transformer.py`: dedicated ID/checkpoint, linear/chroma/gradient/highlightloss, AdamW/cosine, exactCPU resume and source/profile guards; held-out oracle/estimated evaluation with highlight metrics; required calibrated NPZ schema/memory limit, float output and realRGB16 TIFF with explicit radiance scale/clipping fraction.
- `validate_transformer.py`, smoke/formal configs, verification scripts, semantic tests, CI second pipeline; Chinese data/model/conformance guide and compact evidence. Existing v1 pipeline, public core and weights remain unchanged.

## Executed evidence

From repository root, using the existingPython3.11 venv:

```text
python -m pytest -q -p no:cacheprovider --basetemp runs/pytest-zhihu-02
python -m jsr_repro.validate --output runs/zhihu-v1-regression
python -m jsr_repro.validate_transformer --output runs/zhihu-transformer
```

The ordinary pytest command first hit sandbox denial of Windows default Temp and existing cache at fixture setup (33 non-temp tests passed,10 setup errors). Redirecting temporary files to a fresh owned workspace directory and disabling cache resolved the environment issue; this is not represented as a code failure or an unexecuted test success.

The suite contains43 tests: original23 plus20 expanded tests for units/PTC/dark, bracket span, true channel saturation, electron noise, PSF/integration/CFA coordinates, deterministic/leak-checked data, retained frame shapes/shared phase head, metadata effects/gradients, zero/gain/edge preservation, malformed input, independentRGB16 decoding, exact resume and calibration guards. Saved final log: [pytest.log](../../transformer-validation/pytest.log).

Both offline validations passed: unchangedv1 32step regression matches previous metrics; newwidth8 model16steps on12 procedural scenes (10train/1val/1test). Newtest:43.97% sensor signal saturation,65.36% GT values above referencewhite1. Oraclefixed-white PSNRmerge16.6222dB vsmodel16.6350dB; estimated11.7774/11.7877dB. Only one synthetic testcrop, no quality ranking. High-lightRMSE and all-unavailable fractions remain in reports. Estimated alignment is materially worse.

Newzero maximum0; conditionalgainRMSE0 on this fixture for0.125/0.5/2 gain with consistent covariance/masks; no unconditional radiometric/color guarantee. SyntheticPTC gain maximum error4.44e-16e/DN. CPUcontinuous4 vs2+2 resume state_dict is bitwise equal. Largerwidth32/heads4/native32/K7 model307299parameters: one optimizer step, all parameter gradients finite and nonzero. This does not establish formal training convergence or a compute advantage over papers.

Float output/TIFF/NPZ/checkpoints are in ignoredruns/zhihu-transformer; compact portable evidence is [transformer-validation](../../transformer-validation/). Checkpoints/data archives are not committed. Visual index is [visual-index.json](../../transformer-validation/visual-index.json); comparison.png inspected by root: sensor mosaic, saturated reference/long exposure, GT/merge/model and amplitude with stated diagnostic encodings. GT/model differences are visible and are not concealed.

## Explicit assumptions and limitations

Speech source is user-supplied text, not independently retrieved Zhihu URL. OriginalTransformer architecture/weights/recipe/PTC/PSF/spectral construction remain unknown. Three representative wavelengths do not reconstruct hyperspectral scene data. Target is PSF-after/integration-before/transmission-before referenceRGB. Referencewhite is ISO-normalized DN brightness, not fixed absolute illumination acrossISO; f-number controls PSF here, not a jointly derived absolute photon budget. Airy discretization/truncation, simplified fullwell/readnoise, no PRNU/exposure-dependent darkcurrent/fullDSNU correction, no measured aberrations.

Conditional gain equivariance is not absence of color bias. Late/shared-phase construction is not proof of eliminating artifacts. All-saturated regions remain underdetermined. Estimated translation does not replace realcamera alignment/deghosting. Formal100k naturalimage training, realRAW end-to-end, originalbenchmark scores, externaltrained baseline comparisons, CUDA speed, productiontiling and color-managed ISP were not executed.

## Next stage

Independent read-only review against frozenCONTRACT/PLAN, code and evidence. Do not publish until any in-scope reproducible blockers have been repaired and rechecked within the specified single repair cycle. After a pass, stop code iteration and update PR#1 without touchingmain.
