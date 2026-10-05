**PASS_TO_NEXT_STAGE**

Both initial blockers are resolved. No repair-induced in-scope blocker was found.

- Independently recomputed all **29 fixed-geometry cases, 0..16383 DN**: CSV values matched exactly. Identical disk/annulus masks contain **112/304 spatial samples**, with 16 midpoint integration samples per pixel. Affine residual, identity error, and the **19-case DN≤512 subset** agree with JSON. Documentation and plot distinguish normalization white **16383** from uint16 storage maximum **65535**.
- Original short ADC validity is captured before exposure normalization. Invalid contributors affect neither accepted sums nor counts. Quantitative support intersects positive accepted short counts, conservative short validity, and registered long support; empty joint support rejects.
- The original **seed48 counterexample** now yields **valid_fraction=0.0497685187**, rather than 1.0. Its doubled-exposure variant rejects empty support. A separate mixed-exposure probe exactly reproduced unsaturated-contributor sums, counts, and values. Fully saturated inputs produced zero accepted sums/counts and rejected at the metric gate with controlled registration.
- Optional mask defaults matched baseline `07b0a25` exactly in the numerical probe. Negative read excursions remain preserved by the importer; quantitative validity retains documented ADC black/white semantics. No fitted gain was introduced.
- All **14 compact-package hashes** matched.

Independent checks were read-only and in-memory; no implementation, tests, docs, or git state were modified. The packaged full-suite log records **79 passing tests**, but I did not rerun file-writing tests. Previously accepted core parity, spectral assets, and geometry/LCA/capture-order checks remain accepted. Real S5M2 **NOT_RUN** remains an allowed limitation. The initial verdict remains unchanged.