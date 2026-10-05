# JSR review-driven pipeline revision

status: FROZEN
baseline_local: 07b0a2506af9da0ce856c51c1fec344a2013eb4a
baseline_remote: 2d4148bd8d9f237e2d4ef9e21425e862928968ce
target_branch: reproduce/jsr-pytorch-engineering

## Outcome

Address the user's supplied review with runnable alignment, shared LCA correction, explicit capture order and honest diagnostic evidence. Preserve public-core parity and the completed 61-band spectral-camera-v2 pipeline. Deliver independently built and reviewed changes on PR #1; do not merge main. The result remains an independent engineering reproduction, not recovery of undisclosed author code.

## Scope

1. Robust estimated alignment: global homography, reliable local correspondences, reference and cross-frame checks, polynomial residual fields of degree up to four, deterministic lower-degree fallback and explicit confidence/failure reporting. Integrate native-pixel geometry into v1 reconstruction and the seven-frame preprocessing path, retaining translation/oracle baselines.
2. The same configurable channel-wise LCA geometry on learned and legacy RGB before RefineNet. Controller amplitude comes from initial legacy; RefineNet amplitude is recomputed from transformed legacy. Default identity preserves previous public-core behavior.
3. Separate capture order from exposure and storage order for Transformer temporal differences and late fusion. Validate a permutation of temporal ranks; old archives/checkpoints have documented legacy defaults. Do not silently reinterpret old checkpoint semantics. Preserve reference frame 0 and exposure reference units.
4. Reproducible mathematical diagnostics: stepped rings (29 documented DN values, dark-dense sampling, fixed geometry, 16 subpixel integration points); R/G/B displacements -0.5/0/+0.5 native pixels; both axial stripes with cross-axis and checkerboard metrics separated from true chroma attenuation; coma, astigmatism, spherical aberration and defocus; v1 K=2..14 coverage/quality curve. Publish independent fixture parameters because the author's exact geometry/code list/15 optical conditions are unavailable. Exercise the Transformer stripe head too.
5. Three explicit evidence tiers: A mathematical fixtures; B natural linear/HDR or spectral inputs with source/units/provenance; C real RAW burst adapter/evaluation protocol, including independent long-exposure reference and exposure normalization without fitted arbitrary gain. Real S5M2 nine-stop claims require real files and metadata; a runnable protocol and a machine-readable NOT_RUN/missing-data record are the required deliverable when those files are absent. Never fabricate a real-camera run or relabel synthetic gain as nine-stop evidence.
6. Targeted tests, compact CPU diagnostic artifacts, runnable verification scripts/configs, README/source/conformance updates and implementation report. Retain unfavorable metrics.

## Non-goals

- No exact author Tap semantics, 36-output interpretation, Transformer topology, training recipe, camera PTC or unpublished lens library.
- No guaranteed moving-object deghost implementation, production tiling, rolling-shutter solution, full training or SOTA claim. Explicitly record dynamic-object deghost as unsupported; confidence rejection is not that capability.
- No invented 15-condition author table, exact author numerical replication or S5M2 result without data.
- No changes to original Core-Only-Source-Code files/weights; no main merge; no paid data/computation.

## Frozen Design

- Existing IDs and default behavior must remain loadable. New geometry/LCA/order settings are explicit and archived with results. Unknown details remain INFERRED; confirmable public core remains a separate path.
- Coordinate convention: frame(x,y) observes reference(x+dx,y+dy) for translations. Dense geometry documents both mapping direction and native-pixel units, output scale, valid support and inverse sampling. Do not treat homography forward coordinates as inverse sampling coordinates. Reference geometry is identity.
- Alignment must actually compute local correspondence constraints and use cross-frame consistency/reliability information; a nominal polynomial field filled with zeros is not completion. Guard conditioning, sufficient controls, finite transforms, low-texture and singular inputs; fall back with recorded reasons. Estimated geometry is independent of oracle fixture motion.
- LCA correction applies identical geometry/units/boundaries to both branches. Provide a known synthetic displacement test with the correct correction sign, and verify amplitude positions and homogeneity. Runtime defaults are identity; estimated lens calibration is not claimed.
- Capture order is metadata independent of EV. Sorting/storage permutations with corresponding metadata must not change the intended temporal processing. Keep old behavior when metadata is absent, with documented missing-order provenance. Checkpoint/config identity must guard semantics introduced for new training.
- Existing spectral assets, measured relative response provenance, analytic/fitted PTC distinction, leak guards, exact resume, float and uint16 output semantics are preserved. Add spherical OPD term using stated basis/units. No regression to RGB delta-wavelength proxy.
- Diagnostics compare the initial/legacy/final or corrected-merge paths as applicable. State checkpoint provenance (random, short-trained or public standalone). Report border exclusions, valid masks, radiance/DN scaling, residual/fitted-line definitions and true chroma attenuation separately. Do not use improvement as a pass criterion for untrained models.
- Real RAW protocol records black/white/CFA, exposure/ISO/transmission and camera identity. Long-reference metrics require independently measured exposure ratio, correct spatial registration and declared camera color units; insufficient metadata prevents quantitative claims. File hashes and explicit missing inputs are part of the report.
- Builder owns implementation/tests/configs/repo explanatory docs and IMPLEMENTATION_REPORT, but not this contract/plan, review verdict, git commits or remote publication. Root owns planning, read-only integration validation and publication. Reviewer is a different agent and must not edit implementation. All agents must preserve others' work and the current spectral phase.

## Robustness Envelope

CPU deterministic small Bayer bursts K=2..14 (Transformer K=7), native even dimensions within existing memory limits; known translations, modest homography and smooth residual motion; zero/flat/noisy/low-texture and invalid inputs; channel LCA including half a native pixel; temporal storage permutations; normalized finite PSFs with nm OPD. Real camera data quality and unknown author implementations remain outside any success guarantee.

## Required Evidence

- Full existing test suite plus meaningful tests for alignment homography/residual/fallback/sign, geometry reconstruction support, LCA branch/amplitude handling, capture order and malformed metadata, spectral spherical term, diagnostics definitions and real-reference metadata gating.
- New reproducible CPU diagnostic run and compact JSON/CSV plus visual evidence showing stepped response, axial/LCA errors and K curve/optical fixtures. Metrics may show artifacts or degraded quality; retain them.
- Existing core numerical parity tests remain valid; existing train/resume tests remain valid. Run the existing spectral verification or an equivalent preservation check where cost permits.
- Real tier status and protocol, clearly separating synthetic-only execution from missing S5M2 evidence.
- Builder implementation report listing changes, exact commands/results, provenance, limitations and evidence paths. Separate read-only reviewer verdict PASS_TO_NEXT_STAGE or REVISE_BLOCKERS_ONLY with reproducible in-scope blockers.

## Completion Bar

All scoped code/protocol/documentation is runnable; required tests and diagnostics complete with honest evidence; no unresolved reviewer blocker after one repair cycle; new changes are committed/published non-forcibly on the existing branch, PR description updated and main unchanged. Missing author files or real RAW inputs remain explicit limitations rather than invented completion.

## Blocker Definition

Reproducible violations of this frozen scope: wrong units/mapping, nominal unused alignment constraints, unequal LCA geometry or wrong amplitude location, broken temporal ordering/backward compatibility, baseline regressions, invalid metrics, missing runnable scoped diagnostics/protocol or false empirical/source claims. Improvements beyond this scope are not blockers.
