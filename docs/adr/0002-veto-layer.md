# ADR 0002 — Veto Layer (Hard Invariants)

**Status:** Decided for this Domain 1 iteration — with a data limitation
documented explicitly below.

**Context:** The Veto Layer exists to guard against two scenarios the
Calibrator/Executor cannot cover by design: (a) the model is miscalibrated
or the artifact arrived corrupt, and (b) there are business rules that
must always hold, independent of how good the model is.

## Decision — invariants for this iteration

1. **Missing or corrupt artifact → automatic veto to "escalate for manual
   review."** If `puente.leer_vigente()` fails, or the artifact doesn't
   validate against ADR 0001 (features and coefficients of different
   length, score outside [0,1]), the Executor must never "keep operating
   on whatever it last remembered" — that is exactly the kind of implicit,
   unauditable state that caused a kill-switch bug in a previous project.
   With no valid artifact, there is no automated decision.
2. **Extreme absolute amount → veto to review, regardless of the model's
   score.** A fixed threshold, explicitly configured (never a silent
   default) — see `config` in Phase 4.
3. **Model score outside a valid numeric range → veto.** A NaN score, or
   one outside [0,1], is never treated as "0" or "1" — it's treated as "I
   don't know," which in a fraud system means escalate, not approve.

## Known data limitation, documented explicitly (Phase 0)

This iteration's dataset (`docs/PLAN_DE_TRABAJO.md`, Credit Card Fraud
Detection) **has no card/account identifier** — every row is an anonymized,
independent transaction, with no way to group by entity. For this
iteration, that means:

- The Executor (Phase 2) maintains **global recursive statistics** (EWMA
  of amount, count in window, over the whole transaction population), not
  per card — the architecture (O(1) update per new event) is the same, but
  it loses the per-account personalization dimension a real fraud system
  needs.
- Veto invariants that would depend on per-account history (e.g. "amount
  far above *that* card's average") **are not implemented in this
  iteration** — explicitly left pending until a dataset with a real entity
  identifier is used.

This doesn't invalidate Phase 5's goal (measuring precision/recall/latency
of the two-layer pattern) — it does limit how realistic the per-account
personalization is until the data source is resolved.
