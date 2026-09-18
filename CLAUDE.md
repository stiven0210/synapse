# SYNAPSE — Claude Code Context

## What this project is
Two-speed decision framework: a slow layer (Calibrator) that
learns/calibrates with no time limit, and a fast layer (Executor) that
decides in microseconds applying what's already been learned, without
thinking again. See `docs/PLAN_DE_TRABAJO.md` for phase-by-phase status.

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
- No generic interface/contract layer yet: it's built concrete for
  Domain 1 (fraud detection), and only generalized after a second real
  domain (see the Phase 6 plan).
- Always walk-forward / temporal split — never a random split on data
  with real temporal order (same statistical-honesty principle used in
  earlier projects).
- Every formula/updater is tested against a known numeric case ahead of
  time, not just "runs without error."
- Speed is measured, never assumed — every latency benchmark reports a
  real number (microseconds), not an estimate.

## Stack
Python 3.14. `scikit-learn` for the Calibrator (Phase 1). Domain 1's
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
- **Same 3-layer discipline used in an earlier project of ours**
  (deterministic schema, deterministic grounding, selective auditor —
  disagreement is never resolved by majority vote), replicated here as
  new code: SYNAPSE shares no code with other projects.

## Structure
See `README.md` for the full tree.

## Current phase
Phase 0 — Foundations (dataset downloaded and validated, ADRs in
progress). See `docs/PLAN_DE_TRABAJO.md` for detailed status.
