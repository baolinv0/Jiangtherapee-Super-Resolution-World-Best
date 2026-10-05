# Build and independent review plan

1. Builder reads CONTRACT.md and existing utilities/tests. Add independent calibration/optics/bracketed-data modules, seven-frame model and dedicated train/evaluate/infer/validate commands. Preserve v1.
2. Builder writes semantic tests and smoke/formal configs; completes >=16-step offline validation, v1 regression, larger forward/backward, saves compact numerical results and visualization, writes speech conformance/data/model documentation and IMPLEMENTATION_REPORT.md. Update workflow for the new verifier. Do not modify original source/weights or this plan/contract.
3. Root supplies the frozen contract and final diff/evidence to a distinct read-only Reviewer. Record REVIEW.md. If needed, Builder performs one blocker-only repair and Reviewer rechecks it. No unbounded redesign.
4. Root checks completion bar, updates the existing remote branch/PR through GitHub's Git Data API retaining its real ancestry, attaches the PR and verifies file hashes/CI. Write final status and user-facing artifact links.

## Paths and commands

Repo: C:/Users/Lenovo/Documents/Codex/2026-10-05/referenced-chatgpt-conversation-this-is-an/work/jsr

Python: C:/Users/Lenovo/Documents/Codex/2026-10-05/referenced-chatgpt-conversation-this-is-an/work/venv/Scripts/python.exe

Baseline local commit: 760f12a. Target branch: reproduce/jsr-pytorch-engineering. Remote baseline: fa3ea8ec05d84f49695e9010678ddd2f7c950d2e. The local repository is a partial API snapshot with unrelated local ancestry: do not push it directly or force-update any remote ref.

Validation (from repo):

```text
<python> -m pytest -q
<python> -m jsr_repro.validate --output runs/zhihu-v1-regression
<python> -m jsr_repro.validate_transformer --output runs/zhihu-transformer
```

Existing relevant evidence: docs/VALIDATION.md; docs/validation/*; runs/validation-final; ../public-core-parity.json; ../full-model-check.json. Public upstream NPZ files currently untracked locally are already remote assets and must not be staged as new additions.

Builder owns code/config/tests/workflow/explanatory docs and IMPLEMENTATION_REPORT.md. Root owns publication, REVIEW.md and FINAL_STATUS.md. Agents share the workspace; preserve edits by others. Builder need not make a git commit. If a validation command cannot run, document the concrete error and fix within scope before claiming completion.
