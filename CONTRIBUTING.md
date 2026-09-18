# Contributing to SYNAPSE

Thanks for considering a contribution. SYNAPSE is a small, opinionated
project — the guidelines below exist to keep it that way, not to add
process for its own sake.

## Reporting bugs

Open an issue with:

- What you expected to happen, and what actually happened.
- The minimal steps to reproduce it (a failing test is ideal, but a clear
  description works too).
- Which domain / script you were running (`scripts/validacion_end_to_end.py`,
  `scripts/deteccion_deriva.py`, etc.) and the dataset involved.

If the bug affects the Veto Layer or the Executor's decision path, say so
explicitly — those get priority, since a silent wrong decision is worse
than a crash.

## Proposing features

Open an issue describing the problem before writing code. For anything
that changes the shape of the Policy Artifact, adds an invariant to the
Veto Layer, or introduces a new architectural component, expect the
discussion to end in an ADR (see `docs/adr/`) before implementation —
that's not bureaucracy, it's how the project keeps the *why* behind a
decision as legible as the decision itself.

## Adding a new domain

SYNAPSE has been validated on 3 domains (fraud, per-account fraud,
industrial anomaly detection — see the README). Adding a 4th is one of the
most valuable contributions possible, and the bar is the same one the
existing 3 had to clear:

1. **A public, real (or realistically synthetic) dataset.** No domain
   ships without a dataset someone else can download and reproduce the
   result against.
2. **A real Calibrator/Executor split for that domain**, following the
   existing pattern — the Executor must stay pure arithmetic, no model
   inference, no network call, on the hot path.
3. **Measured latency, not estimated.** Use `time.perf_counter()` around
   the actual decision call, end-to-end, the same way the existing domains
   do — see `scripts/validacion_end_to_end.py` for the pattern.
4. **A walk-forward or temporally honest split** if the data has any time
   ordering. A random split on temporally ordered data inflates metrics
   and won't be accepted.
5. **An honest writeup** of what the numbers actually show, including
   where the domain's result is weaker than the others — the existing
   domain docs (`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`,
   `docs/CAMINO3_GENERALIZACION_IOT.md`) are the model to follow, including
   their sections on what *didn't* work.

Open an issue with the dataset and the domain's shape before writing the
implementation — it's much cheaper to correct course in a discussion than
in a PR.

## Code and test standards

- **Every formula gets a known-answer test** before it gets a "runs
  without crashing" test — a metric that isn't verified against a hand
  computed or literature-sourced expected value isn't trusted here.
- **Fail loud, never silent.** If your code can't produce a valid
  decision, it should raise or escalate — not return a default that looks
  like a real answer. See `docs/architecture.md`'s Design Principles.
- **The Executor and Veto Layer stay pure.** No model inference, no I/O,
  no network calls on the hot path. If your change adds either, it
  probably belongs in the Calibrator or a new support component instead.
- **Run the full suite before opening a PR:**
  ```bash
  pytest tests/ -q
  ```
- Match the existing code's language convention: identifiers (function,
  class, and variable names) and functional string literals (exception
  messages, enum-like values, JSON/dict field names) in `src/` and
  `tests/` are Spanish — this is a Spanish-first codebase, and some of
  those strings are matched exactly by other code (e.g.
  `bitacora_decisiones.clasificar_razon()`), so don't translate them.
  Comments and docstrings, on the other hand, are English throughout,
  same as `docs/`, the README, and the ADRs.
- Small, focused PRs over large ones. If a change touches the Policy
  Artifact contract or adds a Veto invariant, link the ADR that motivates
  it.

## Questions

Open an issue — there's no separate chat or mailing list for this project.
