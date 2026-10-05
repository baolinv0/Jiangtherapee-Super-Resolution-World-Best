# Compact review-revision evidence

- [diagnostics.json](diagnostics.json): complete Tier A geometry/stripe/ring/K/optical records and executed Tier B provenance.
- [metrics.csv](metrics.csv): axial, LCA, false-chroma and true-chroma metrics, including Transformer outputs.
- [k_curve.csv](k_curve.csv) and [plot](k_curve.png): K2..14 phase coverage and quality, including worsening RMSE.
- [stepped_response.csv](stepped_response.csv) and [plot](stepped_response.png): 29 fixed-geometry intensity cases over 0..16383 DN, constant disk/annulus sample counts and mean output; separate affine/identity and dark-subset errors in diagnostics.json. Plot x-axis log1p DN, y-axis linear DN.
- [fixtures.png](fixtures.png): fixed white1 linear previews of rings, both stripe axes, true chroma and 61-band optical reconstruction; PSF panels use peak normalization for display only. Dark DN steps are numerically documented in the CSV/plot; the linear preview does not brighten them.
- [natural_status.json](natural_status.json): local OS flower photograph, source hash, assumed inverse-sRGB display-linear units, hypothetical spectra and random-model metrics. Original photo not redistributed.
- [synthetic-support.json](synthetic-support.json): seed48 clipped-short decoder stub with actual registration; 51.34% total ADC-invalid, 4.98% joint metric support; mixed exposure empty support rejected. Synthetic only, no real camera evidence.
- [synthetic-support.json](synthetic-support.json): seed48 clipped-short synthetic decoder with actual registration; 51.34% total ADC invalid, 4.98% joint support; mixed exposure empty support rejected. No real camera evidence.
- [real_status.json](real_status.json): honest Tier C NOT_RUN/missing inputs, no nine-stop claim.
- [spectral-preservation.json](spectral-preservation.json): existing 61-band/28-response/eight-step CPU verification and retained intervention metrics.
- [public-core.json](public-core.json): passing standalone numerical parity; separate from complete-pipeline claims.
- [tests.txt](tests.txt): final current-code 79-test pass.
- [package.json](package.json): evidence hashes. Full runs remain ignored under runs/review-revision.

Conventions and protocols: [REVIEW_REVISION.md](../REVIEW_REVISION.md). Build commands/results/limitations: [IMPLEMENTATION_REPORT.md](../tasks/review-revision/IMPLEMENTATION_REPORT.md). Independent read-only review: [PASS_TO_NEXT_STAGE](../tasks/review-revision/REVIEW.md). Root integration: [79 tests and asset checks](ROOT_VALIDATION.json).
