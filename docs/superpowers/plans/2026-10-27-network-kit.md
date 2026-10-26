# JSR Architecture and Weights Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Present the recovered network completely and deliver exact original weights in a portable PyTorch kit that can be fine-tuned and reloaded.

**Architecture:** Preserve the recovered v9.8 computation. Export unique Controller / RefineNet / affine components with verified hashes and K routes; retain a separate trained-checkpoint format and explicit supervised boundary input for fine-tuning. Generate layer/parameter catalogs and diagrams from the actual model and source weights.

**Tech Stack:** Python, PyTorch, NumPy, JSON/CSV, standalone SVG/PNG/Markdown.

**Spec:** User request in this chat; `docs/REA_RECOVERY.md` fixes the recovered computation and remaining boundaries.

## Global Constraints

- No invented layers, reinitialized original weights, or changed RAW coordinate conventions.
- K=13 aliases the K14 parameters but retains its own expected input frame count.
- Load PyTorch artifacts only with `weights_only=True`; state dictionaries and JSON-compatible metadata only.
- Original and fine-tuned provenance remain distinguishable; warm start is not optimizer resume.
- v17 selected assets remain an inventory with explicit recovery limits.

## Review Focus

- Every exported tensor must equal its source NPZ tensor exactly.
- Deduplicated aliases must restore the requested K, not the stored component's K.
- Checkpoint round trips must preserve architecture, affine and output values.
- Freezing controller/refine stages must freeze the intended parameters.
- The portable kit must load without paths into this machine's original checkout.

### Task 1: Exact pretrained component bundle

Files: `reproduction/jsr_repro/network_bundle.py`, `tests/test_network_bundle.py`.

Interfaces: `export_pretrained_bundle(weights_dir, output_dir, *, ks=None) -> dict`; `load_pretrained_bundle(bundle_dir, k) -> RecoveredJSR`.

- [x] Write and observe failing tests for exact K4 tensors/forward, K13 alias, corruption rejection, and omitted route errors.
- [x] Implement manifest-pinned component export and strict route loading.
- [x] Export all routes and verify all tensors against all 21 original networks.

### Task 2: Trainable checkpoint and supervised example

Worker-owned files: `recovered_checkpoint.py`, `finetune_recovered.py`, `test_recovered_checkpoint.py`.

- [x] Save/load complete recovered architecture and state with explicit provenance.
- [x] Add boundary-supervised controller/refine/joint fine-tuning and warm-start support.
- [x] Demonstrate an original K4 update and exact reload; test stage freezing.

### Task 3: Complete presentation and portable deliverables

Files: `docs/NETWORK_KIT.md`, architecture/weight export script, generated JSON/CSV/diagrams.

- [x] Present full pipeline, Controller layer graph, eight RefineNet blocks, FiLM and output factorization.
- [x] Enumerate every weight/bias/buffer shape, parameter count, source key, file hash and K route.
- [x] Package actual weights, portable source, original-K4 checkpoint and usage examples.
- [x] Verify clean-location load, one-step fine-tuning and full test suite.

Remote publication and CI status are recorded in the existing draft pull request.
