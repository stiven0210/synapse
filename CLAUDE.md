# SYNAPSE — Claude Code Context

## What this project is
Two-speed decision framework: a slow layer (Calibrator) that
learns/calibrates with no time limit, and a fast layer (Executor) that
decides in microseconds applying what's already been learned, without
thinking again.

**Independent** project — shares no code with any earlier project.

## Principles
- The Executor never does heavy model inference, never makes network
  calls, never retrains anything — it only evaluates an already-calibrated
  policy artifact against a compact state.
- The Calibrator never decides in real time — its only output is a
  versioned policy artifact (JSON).
- The Bridge updates the artifact atomically (temp file + atomic
  replace) — the Executor must never be able to read a half-written
  artifact.
- The Veto Layer is model-independent — its invariants hold no matter
  what happens in the Calibrator/Executor.
- An `Ejecutor`/`CicloDecision` instance is single-threaded by design: it
  expects one sequential stream in time order and has no locking. Never
  share an instance across threads; to parallelize, partition by
  account/entity (see the `src/ejecutor.py` docstring).
- Always walk-forward / temporal split — never a random split on data
  with real temporal order.
- Every formula/updater is tested against a known numeric case ahead of
  time, not just "runs without error."
- Speed is measured, never assumed — every latency benchmark reports a
  real number (microseconds), not an estimate.

## Stack
Python 3.14. `scikit-learn` for the Calibrator. Domain 1's
dataset is public, with no credentials. One exception: `src/agente_triage.py`
(triage of Veto escalations) uses `anthropic` + `ANTHROPIC_API_KEY` in the
environment — never hardcoded, and only in the real client
(`crear_cliente_claude()`); all the auditing logic is tested with an
injectable client, with no real call (see `docs/adr/0003-triage-agent.md`).

## Principles (LLM agents)
- **No LLM on the hot path.** `agente_triage.py` is never invoked from
  `CicloDecision.decidir()` — it runs afterward, over entries already
  written to `bitacora_decisiones.py`.
- **No LLM decides or blocks anything.** Its output is a hypothesis for a
  human to verify, it never changes system state.
- **Same 3-layer discipline** (deterministic schema, deterministic
  grounding, selective auditor — disagreement is never resolved by
  majority vote), implemented as independent code.

## Structure
See `README.md` for the full tree.

## Current state
v1.0.1 released. 187 tests: all pass locally with the datasets; in CI and fresh clones 185 pass and 2 skip (they need the local Sparkov dataset). 3 domains validated:
- Domain 1: Global fraud detection (ULB Credit Card, F1 0.78, 8.33 µs)
- Domain 2: Per-account fraud (Sparkov 1.17M rows, AUC-PR 0.90, 9.84 µs)
- Domain 3: Industrial IoT anomalies (SKAB 34 files, F1 0.76 unsupervised,
  3rd of 9 on official leaderboard, 13.05 µs)

Public repo (github.com/stiven0210/synapse) under the MIT license, with
English documentation and ADRs.
