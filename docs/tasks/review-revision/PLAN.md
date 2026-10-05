# Build and independent review plan

1. Read the frozen contract and inspect current code, preserving the spectral-camera-v2 additions and original core assets.
2. Independent build agent implements geometry/LCA/order and diagnostics/protocol with targeted tests; records its implementation report. It has no commit/publication authority.
3. A separate read-only Codex CLI agent performs the contract review of the diff from 07b0a25, report and evidence. Root runs integration validation and packages compact evidence. If the reviewer reports blockers, one focused builder repair and reviewer recheck follow. Root does not implement the scoped code.
4. Root stages only reviewed changes, commits locally, publishes using the current GitHub branch tree with an exact remote-head guard and non-force update, updates PR #1 and verifies main/CI.

Agent transport: the built-in spawn tool reports `agent thread limit reached`, including reuse of sources_audit, and the preset builder previously failed because its fixed model is unsupported. A separate Codex CLI launch initially failed automatic approval review for lack of specific model-service data-transfer authorization. The user then explicitly authorized independent build/review processes to send the public repository and requirements to OpenAI. Use separate Codex CLI sessions, builder with workspace-write sandbox and reviewer with read-only sandbox, with the user's configured model. Record actual identities, do not claim preset roles ran. Do not bypass sandbox/approval rules.

Validation Python: ../venv/Scripts/python.exe from the repository root. Keep full generated runs ignored; commit compact portable reports under docs/revision-validation. No user question is needed for routine implementation choices. Real S5M2 metrics are conditional on available real files as stated in the contract.
