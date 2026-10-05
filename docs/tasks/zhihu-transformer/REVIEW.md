# Independent contract review

Verdict: **PASS_TO_NEXT_STAGE**

Date: 2026-10-05. Reviewer: `/root/sources_audit`, a separate agent from the root implementer. Scope: frozen [CONTRACT.md](CONTRACT.md) and [PLAN.md](PLAN.md), staged delta against local760f12a, [implementation report](IMPLEMENTATION_REPORT.md), code/config/tests/workflow, numerical records and visual index. Review was read-only; only isolated ignored verification outputs were generated. The unavailable preset Builder/Reviewer models were not represented as having executed.

## Independent verification

- Complete suite: **43 passed in19.51s**, fresh `runs/independent-review-001`, cache disabled to respect the Windows sandbox.
- New offline pipeline: **16 steps passed**, `runs/independent-transformer-review-001`. Oracle/estimated metrics, saturation fractions, zero/gain probes and PTC error reproduced submitted records exactly.
- Existing pipeline: **32-step regression passed**, `runs/independent-v1-review-001`; reported metrics reproduced.
- Continuous versus resumed checkpoints: every model tensor identical; optimizer, scheduler, RNG, progress and dataset/profile identity are present.
- Larger width32/native32 check: **307299 parameters**, finite nonzero gradients and one optimizer step passed.
- Independent OpenCV decoding confirms actual **uint16 RGB TIFF**, exactly matching documented white4 quantization of saved float RGB.
- Compact records match full-run artifacts. `comparison.png` was viewed; rendering/panel descriptions agree. Both indexed images exist and the copied preview matches its generated original.

## Code findings

Electron/DN normalization, squared-coefficient variance propagation, native/HR centers and inverse-warp direction are consistent. Seven frames persist through encoder/decoder. Reduction occurs afterward as7→4→2→1, followed by two shared latent expansions and RGB projection. Inferred mechanisms, synthetic calibration, source provenance and underdetermined saturation are identified clearly.

No reproducible blocker against the frozen contract. No required code repair or scope change.

## Nonblocking follow-up

The compact visual index initially referred to an existing full-run preview not copied into its corresponding compact location. The required comparison image was already present. Root copied that existing preview into the indexed path; reviewer verified the files agree. Resolved without implementation changes.

## Limits of the verdict

This pass establishes the specified engineering behavior and evidence integrity. It does not establish author-equivalent reconstruction quality, independently measured camera calibration, real-camera validation, complete natural-image training, CUDA speed or SOTA performance. These remain explicitly outside this completion claim.
