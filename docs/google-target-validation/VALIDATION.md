# Google-stage target validation — 2026-10-05

Base: `0e764de8bb026b1e08da3655d8d741e299018108`. This is verification of the
independent `spectral-camera-v3` data contract, not a reproduction score for
Google or the original JSR author. [Design and migration](../GOOGLE_TARGET.md).

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Existing baseline suite | 79 passed | Executed before changes |
| Final complete suite | 110 passed | [tests.txt](tests.txt) |
| New physical and integration coverage | Dense/native center phase, fixed native aperture, three target stages, exposure/noise isolation, invalid stage rejection, checkpoint identity and export | `tests/test_google_target.py`, `tests/test_google_target_integration.py`, updated spectral tests |
| Legacy numeric compatibility | v2 and v3/pre-optics: all 13 tensors identical to pre-change baseline; other v3 stages: 12 observation tensors identical, target intentionally different | [legacy-parity.json](legacy-parity.json) |
| Actual old checkpoint | Base-revision code creates step 1; changed code strictly resumes to step 2 and evaluates as v2/pre-optics | [legacy-checkpoint-resume.json](legacy-checkpoint-resume.json) |
| Omitted-aperture negative control | Removing pixel integration makes the independent 61-band/four-field oracle test fail; restored code passes | [mutation log](omitted-aperture-mutation.txt) |
| Equal-exposure pipeline | 8 CPU optimization steps, oracle/estimated evaluation, NPZ pair export and inference to linear RGB/16-bit TIFF | [equal/validation.json](equal/validation.json) |
| Bracketed HDR extension | Same end-to-end checks, 8 CPU steps, unclipped reference GT | [hdr/validation.json](hdr/validation.json) |

The full-suite environment is recorded in [environment.json](environment.json):
Python 3.12.14, PyTorch 2.5.1 CPU, NumPy 1.26.4, OpenCV 4.10.0. All 110 tests ran;
none were skipped. The omitted-aperture mutation failed on 3068/3072 elements,
with maximum absolute error 0.00287496 against the independent integral.

The independent review found no blocking functional defect. Its two findings
were addressed: verification scripts use a separate v3 output directory, and
the nonuniform 61-band/four-node optical-field probe is now a regression test.
Acceptance is based on the numeric checks and executed tests, not reviewer
agreement. No unrelated model architecture change was made.

## Paired-target evidence

For the HDR construction probe, changing only the target stage gives exactly
zero maximum RAW difference for all three stages. Mean absolute target
differences are 0.00936027 for pre-optics versus post-optics, and 0.00331284 for
post-optics versus post-pixel. The default GT changes under f-number, pitch,
fill-factor, PSF support and field-center interventions. The same-reference
61-band test separately rejects omitted, doubled and half-width pixel apertures.

## Smoke metrics and interpretation

All scores below use held-out procedural fixtures, `post_pixel` targets,
fixed data range 1 and no per-image gain fitting. They are not natural-image
benchmark results. Different targets or exposure/gain distributions must not
be compared as if they were the same task.

| Recipe | Alignment | Corrected merge PSNR | 8-step Transformer PSNR |
|---|---|---:|---:|
| Equal exposure | Oracle | 46.37677 | 46.37616 |
| Equal exposure | Estimated | 46.57298 | 46.57338 |
| Bracketed HDR | Oracle | 21.57691 | 21.58325 |
| Bracketed HDR | Estimated | 20.09946 | 20.11144 |

The tiny differences do not establish a learned quality improvement. The
estimated-alignment score being slightly above the oracle score on the small
equal-exposure fixture set is not evidence that estimated geometry is better;
the merge is approximate and this is not a geometry benchmark. Equal-exposure
training records have zero true signal saturation. Deliberate endpoint
saturation probes in the validation report are separate from that recipe.

## Reproduce

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m jsr_repro.validate_spectral --config configs/spectral_smoke.yaml --output runs/google-target-hdr-validation
python -m jsr_repro.validate_spectral --config configs/google_equal_exposure_smoke.yaml --output runs/google-target-equal-validation
```

Use fresh output directories. CI includes the full suite and both spectral
smoke recipes. Historical v2 reports remain unchanged. Local checkpoints and
generated fixture images are not committed; the commands regenerate them.

## Remaining limits

- The 2× scene grid, bilinear motion/interpolation, finite PSF support and 4×4
  area quadrature approximate continuous image formation.
- Relative spectral curves, analytic PTC and assumed aberrations do not supply
  a calibrated real phone/lens model; RGB lifting cannot identify the true spectrum.
- Preserving native pixel response is this project's target definition;
  Google §6.3 supports retaining lens blur but does not publish this GT formula.
- Real RAW performance, convergence of a full natural-image training run and
  product-level quality improvement remain **unverified**.
