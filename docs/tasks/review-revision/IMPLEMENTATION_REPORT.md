# Implementation report — review revision

Root status update: the BUILD-authored report below is a historical handoff snapshot. The subsequent separate read-only [review](REVIEW.md) returned PASS_TO_NEXT_STAGE; [final status](FINAL_STATUS.md) records the 79-test integration result. Pending steps described below refer to the time of the BUILD handoff.

Status: FOCUSED BLOCKER REPAIR COMPLETE; independent read-only recheck pending. Baseline: 07b0a2506af9da0ce856c51c1fec344a2013eb4a on reproduce/jsr-pytorch-engineering. Executed in the independently launched BUILD workspace session; no preset builder/reviewer role or delegated sub-agent was launched here. The session has no staging, commit, branch, remote or review-verdict authority. No CONTRACT/PLAN changes were made.

## Delivered scope

- `geometry.py`: real GFTT/LK correspondence constraints, bidirectional reliability and actual inter-frame cycle checks, RANSAC global homography, normalized polynomial residual degree <=4, conditioned lower-degree fallback, finite/singular/inlier guards, independent inverse-map solution, confidence/failure reports. Native-pixel forward and inverse maps are validated, with identity frame zero. v1 reconstruction uses forward splatting; Transformer packed planes use CFA-center-specific inverse sampling with conservative support and squared variance weights. Both inference CLIs expose robust alignment while retaining estimated translation and provided/oracle baselines.
- Shared configurable v1 LCA corrects learned and legacy branches with identical channel offsets, scale and border behavior. Controller amplitude remains tied to initial legacy. RefineNet amplitude is recomputed from corrected legacy. Defaults preserve prior behavior and return keys; opt-in diagnostics expose both amplitudes and pre-transform branches.
- Explicit temporal ranks are independent of exposure and storage. The Transformer sorts features by ranks before temporal differences/late pairing only in `explicit-ranks-v1`; missing ranks reject that mode. Old configs default to `legacy-storage-v1`, preserving checkpoint semantics. Dataset/batching/archive paths carry optional ranks; malformed fractional/duplicate/permutation metadata is rejected. Existing checkpoint config identity and exact-resume guards apply to the new mode.
- 29 dark-dense DN stepped rings, 16 midpoint integration samples, half-native-pixel RGB offsets, both axial and true-chroma stripes, separate cross-axis/checkerboard/chroma metrics, fitted-line transfer measurements, K2..14 phase coverage and quality, actual Transformer stripe output, and four 61-band optical reconstruction fixtures. No improvement threshold is imposed on random models.
- Primary spherical OPD in nm (`6*rho^4-6*rho^2+1`), optional `spherical_nm` spectral generation setting. Omitting it preserves old random-number consumption. Existing 61-band optics, measured camera relative responses, analytic/fitted PTC distinction, source/leak checks, resume and output formats remain intact.
- Tier B runnable natural linear/spectral adapter and a real local flower-photograph execution, with source hash, inverse-sRGB assumptions and hypothetical spectral lift disclosed. Tier C runnable static RAW/reference adapter, measured metadata gating, independent-file guard, separate spatial registration, exposure normalization without fitted gain, conservative long-RAW validity support and file hashes. Real S5M2 execution is explicitly NOT_RUN.
- `scripts/validate_review_revision.py`, explicit-mode sample config, missing-data configs, 12 focused tests, compact JSON/CSV/PNG evidence, README/source/reproduction/Transformer/spectral/validation documentation links and `docs/REVIEW_REVISION.md` conventions.

## Validation commands and results

Shell: PowerShell, Python: `../venv/Scripts/python.exe`, installed dependencies only. Set `$env:PYTHONPATH='reproduction'` for module CLIs.

1. `../venv/Scripts/python.exe scripts/validate_review_revision.py --output runs/review-revision/final-verification --natural-spec configs/revision_natural_local.json --spectral --public-core`
   - All five commands exit zero: tests, diagnostics, missing-real protocol, spectral preservation, standalone public-core comparison.
   - This initial packaged runner invocation contained 73 tests; two final focused tests and stricter guards were then validated below.
2. Final current-code suite: `../venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp runs/review-revision/pytest-support-final > runs/review-revision/tests-support-final.txt`
   - **75 passed in 35.49 seconds**, no skipped or failed tests.
   - Includes existing numerical frontend parity, module tests, spectral generation/source checks and exact train/resume tests. Added tests exercise projective and smooth residual motion, translation sign, exact dense/oracle reconstruction counts/variance/support equivalence, inverse CFA sampling, insufficient controls/flat fallback, reference identity, LCA sign/homogeneity/both amplitude positions, storage-order invariance, malformed ranks, archive roundtrip, spherical PSF energy, metric definitions, natural-input execution, real-reference metadata gates and a clearly synthetic RAW-decoder stub.
3. Final current-code diagnostics: `../venv/Scripts/python.exe -m jsr_repro.revision_diagnostics --output runs/review-revision/final-diagnostics --natural-spec configs/revision_natural_local.json`
   - Exit zero. Tier A RUN; Tier B RUN with local natural display-linear photograph and declared hypothetical spectra; Tier C remains separate NOT_RUN record. Outputs JSON, three CSV tables and three PNGs.
4. Spectral preservation (executed by runner): `../venv/Scripts/python.exe -m jsr_repro.validate_spectral --output runs/review-revision/final-verification/spectral`
   - Exit zero; spectral-camera-v2, **61 wavelengths, 28 measured response shapes, eight CPU optimization steps**. Final validation PSNR 22.6499957196 dB on procedural fixtures; not a natural HDR or real-camera claim. All source/target interventions and unfavorable metrics retained in spectral-preservation.json.
5. Public standalone parity (executed by runner): `../venv/Scripts/python.exe scripts/compare_public_core.py --output runs/review-revision/final-verification/public-core.json`
   - Status passed. Max absolute Controller error 7.37607479095e-6; RefineNet residual 7.45058059692e-7; factorized RefineNet 7.15255737305e-7. This optional existing verifier uses installed WGPU on Intel/Vulkan; new implementation and diagnostics are CPU runnable. This verifies public standalone modules, not author's unpublished complete RAW pipeline.
6. Missing real files: `../venv/Scripts/python.exe -m jsr_repro.real_reference --spec configs/revision_real_missing.json --output runs/review-revision/real`
   - Exit zero, machine-readable NOT_RUN, missing burst and independent S5M2 long-reference input, nine_stop_claim=false.
7. Missing natural input exercise: `../venv/Scripts/python.exe -m jsr_repro.natural_diagnostic --spec configs/revision_natural_missing.json --output runs/review-revision/natural`
   - Exit zero, NOT_RUN. Separate actual local-photograph execution is preserved in natural_status.json.
8. `git diff --check`: passes. Read-only status/diff inspection confirms no protected original source/weights modifications and no staging/commit/branch/remote operations.

An early full test attempt encountered the sandbox-denied default pytest temp directory; all later invocations use ignored workspace basetemp directories. An early schema regression from unconditional diagnostic return keys was fixed by making diagnostics opt-in. Neither early failure is hidden or represented as final success.

## Evidence and retained negative results

Portable compact package: `docs/revision-validation/`, with `package.json` artifact hashes and `README.md` visual/data index. Full runs/logs/checkpoints: ignored `runs/review-revision/`. No original natural photograph or upstream weight archive is copied to the package.

- Smooth residual-motion alignment forward RMSE: **0.0139164 native pixels** on the 20-pixel-border interior. Estimator never receives ground-truth fixture motion. Flat controls explicitly fall back. Reports include actual cross-frame errors and polynomial degree.
- x-stripe false chroma RMS: initial legacy 0.0203181 -> shared-LCA legacy 0.00412628. Final random v1 RMSE 0.0422312 versus corrected legacy 0.0201600: the learned path is worse.
- True x-chroma projection: corrected legacy 0.690181; final random v1 0.410633. Reduced false color does not prove preserved true chroma. Transformer final x-stripe checkerboard projection 6.20553e-5 and cross-axis residual 2.27306e-4 are nonzero and retained.
- K2 phase coverage .125 -> K14 .6875, while corrected legacy RMSE .0289520 -> .0305565 and random final .0445038 -> .0455936. No monotonic quality or sample-count benefit is claimed for these inferred statistics/untrained models.
- Initial stepped results invalidated by REVIEW_INITIAL blocker1; superseded by the repaired fixed-geometry evidence below.
- Natural local flower crop: inverse-sRGB units and unknown source processing declared; corrected-merge RMSE .000114311, random Transformer .000118996. One small smooth crop is limited evidence and does not establish HDR recovery. Source hash and spectral construction recorded; author/capture parameters unverified.
- Optical aberration fixtures use actual 61-band diffraction plus measured relative spectral response integration. Some reconstructed panels lose nearly all stripe contrast; those outputs and per-condition metrics are deliberately retained. Independent 100nm conditions are not the unpublished 15-condition lens table.

## Limitations and remaining stage

The initial reviewer identified two blockers, repaired below subject to independent recheck. Independent read-only contract review has not run in this BUILD session and remains the next required stage; this report does not issue a review verdict or claim publication completion.

Real S5M2 burst, independent long-exposure RAW, and measured capture metadata were not found. Tier C is NOT_RUN; nine-stop claims are unsupported. The real adapter requires optional rawpy for actual files; only missing-data execution and an explicitly synthetic decoder test were run. Metadata must match decoded ADC black/white levels and use the same ISO; no arbitrary gain calibration is inferred. Moving objects, rolling shutter and dynamic-object deghosting remain unsupported.

The local natural photograph is OS-provided outside the snapshot; its optional config is machine-local and re-execution elsewhere requires a supplied source. Inverse-sRGB display-linear values and lifted spectra are computational assumptions, not measured scene radiance/spectra. The complete independent Transformer and phase-splat fusion remain INFERRED. Exact author geometry, DN code list, lens conditions, Tap semantics, topology and training recipe remain unknown. No backbone expansion, huge dataset download, paid service or original source/weight modification occurred. The three pre-existing untracked upstream NPZs remain untouched and unstaged.


## Single focused blocker-repair cycle

Both initial blockers repaired. No CONTRACT/PLAN/reviewer verdict, original public source/weights, Transformer topology, optics/assets, git or remote state changed. No mandatory plotting dependency added.

Rings execute 29 INFERRED dark-dense codes 0..16383 on identical disk r<=9 and annulus12<=r<=21, white gap and black exterior. Fixed masks disk r<=6 (112 pixels), annulus15<=r<=18 (304), border8: 416 spatial/1248 RGB samples per case. Sixteen spatial midpoint samples remain. DN normalization white16383 is separate from uint16 storage maximum65535. Initial/shared-LCA legacy/random final plateau means report affine residual separately from identity RMSE, also for 19 dark cases DN<=512. Regression checks constant masks, actual plateau values with and without shifts, fixed surroundings and quadrature count across all cases.

RAW validity is preserved before exposure normalization: finite decoder (ADC-black)/(white-black) strictly between0 and1, equivalent to decoded ADC black/white gating. Optional phase_splat validity excludes invalid contributors from both sums and counts; defaults unchanged. Metric support requires positive accepted short counts and conservatively all-valid short spatial contributors, intersected with independently registered all-valid long support. Empty joint support raises ValueError. Exposure/transmission are still independently measured, no fitted gain. Tests use a clearly synthetic decoder and actual registration, including mixed exposures and masked-count equivalence.

Exact repair commands (PowerShell, `$env:PYTHONPATH='reproduction'`):

- `../venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp runs/review-revision/repair-full > runs/review-revision/repair-tests.txt`: **79 passed in 41.36s**. Includes existing frontend parity, spectral generation/source checks and exact train/resume.
- `../venv/Scripts/python.exe -m jsr_repro.revision_diagnostics --output runs/review-revision/repair-diagnostics --natural-spec configs/revision_natural_local.json`: exit0; affected CSV/JSON/PNG regenerated, Tier A/B RUN.
- `../venv/Scripts/python.exe -m jsr_repro.real_reference --spec configs/revision_real_missing.json --output runs/review-revision/repair-real`: exit0; S5M2 NOT_RUN, nine_stop_claim=false.
- `../venv/Scripts/python.exe scripts/compare_public_core.py --output runs/review-revision/repair-public-core.json`: passed. Controller max7.37607479095e-6; RefineNet residual7.45058059692e-7; factorized7.15255737305e-7. Existing spectral-preservation.json retained from first build: repair did not change optics/assets; current suite and regenerated four actual61-band fixtures exercise preservation.

Current ring errors (DN): initial legacy affine residual0.000374036, identity0.655906, dark identity0.734538; shared-LCA legacy0.000375513/0.655915/0.734538. Random final affine residual1.015962, identity36.043788, dark identity39.471434, dark affine residual1.202957; slope0.99493955, intercept39.756882 DN. Poor random dark response remains visible; affine accuracy does not imply identity accuracy.

Reviewer seed48 counterexample: 51.3428% total ADC-invalid shorts (51.6927% interior saturated), independently estimated long registration, joint valid_fraction **0.0497685** rather than old1.0. RMSE0.00934973 retained, with no fitted gain. Doubled second exposure produces empty conservative joint support and rejects. Initial targeted run expected RUN for that mixed case and failed; assertion corrected to the required empty-support rejection without weakening masking.

Evidence: docs/revision-validation/diagnostics.json, stepped_response.csv/png, metrics.csv, fixtures.png, tests.txt, synthetic-support.json, real_status.json and refreshed package.json hashes. Full logs: runs/review-revision/repair-*. Synthetic-support.json is explicitly synthetic regression evidence, never a real Tier C execution. REVIEW_INITIAL remains immutable. Read-only recheck and root packaging/publication remain pending outside BUILD authority.
