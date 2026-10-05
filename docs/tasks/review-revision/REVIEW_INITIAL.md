**REVISE_BLOCKERS_ONLY**

Reviewer identity: `/root`, this independently launched read-only review session. I did not build the implementation or use a preset reviewer role. No files or git state were modified.

Reviewed the complete frozen CONTRACT/PLAN, baseline diff and added code, implementation report, compact evidence and available validation logs. Two reproducible in-scope blockers remain.

1. **Stepped-ring diagnostic confounds spatial geometry with radiometric response.**  
   Location: `reproduction/jsr_repro/revision_diagnostics.py:15`, `:30`, `:118`, `:133`, `:157`.

   **Trigger:** `fixture('rings')` assigns different DN codes to successive radii in one image. The executable uses codes spanning **0..65535**, divides by 65535, and fits all interior pixels together. The CSV’s sample counts vary with ring position. This is not the required 29-level **0..16383 intensity sweep on fixed spatial geometry with constant plateaus and contrast surroundings**. Sixteen-point integration is present, but it does not resolve this confounding. Separate stepped-response identity-error evidence is also absent.

   **Minimal repair:** Execute 29 dark-dense intensity cases using identical documented geometry and sampling masks, with constant plateaus and contrast surroundings; retain 16-point integration and border exclusions. Report affine-fit residual and identity error separately. State normalization white independently from uint16 storage range and fixture DN range. Regenerate affected evidence and documentation. Independent radii and the precise code list remain acceptable.

2. **Real-reference metrics do not require valid, unsaturated short-reconstruction support.**  
   Location: `reproduction/jsr_repro/real_reference.py:54`, `:56`, `:67`, `:75`.

   **Trigger:** Short RAW is exposure-normalized and merged without preserving its original black/saturation masks. The final metric mask checks long-reference contributors and reference values only; it never intersects them with valid short contributors or short reconstruction counts.

   I exercised unchanged `run()` through an entirely in-memory synthetic decoder stub: seed-48 Gaussian-blurred random field `im`, two short frames `min(2*im,1)`, long RAW `.04*im`, exposure ratio 2, short transmission 1 and reference transmission .01. Actual estimated registration succeeded. Although **51.69% of interior short samples were saturated**, the protocol reported **valid_fraction=1.0** and RUN metrics.

   **Minimal repair:** Preserve short validity in decoded ADC units before exposure normalization. Propagate short spatial and unsaturated contributor support through the reconstruction, then intersect it with independently registered long-reference support. Exclude unsupported/fallback values and clipped contributors from quantitative comparison; reject an empty joint mask. Add a targeted saturated-short/valid-long regression check.

Other inspected paths did not produce blockers: geometry is applied before temporal feature sorting, and a coupled nonzero-geometry/storage-permutation probe preserved RGB within **1.19e-7**. Shared LCA geometry, correction sign, amplitude placement, actual cross-frame alignment constraints, CFA inverse sampling and spherical nm OPD are implemented. Compact artifact hashes matched; available full-suite logs record 75 passing tests.

I did not rerun file-writing tests. Public-core parity and existing spectral functionality remain accepted preserved evidence. Real S5M2 NOT_RUN is an expressly permitted limitation, not a blocker.