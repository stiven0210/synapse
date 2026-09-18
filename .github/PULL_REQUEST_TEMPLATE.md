## What this changes

Brief description of the change and why it's needed.

## Related issue

Closes #

## Checklist

- [ ] `pytest tests/ -q` passes locally
- [ ] New logic has a known-answer test (a hand-computed or
      literature-sourced expected value), not just a "doesn't crash" test
- [ ] The Executor and Veto Layer are still free of model inference, I/O,
      and network calls (if this PR touches either)
- [ ] If this changes the Policy Artifact contract or adds a Veto
      invariant, there's an ADR in `docs/adr/` covering it
- [ ] Docstrings/identifiers follow the existing language convention
      (Spanish in `src/`/`tests/`, English in `docs/`)

## Latency impact

If this touches the hot path (`Executor`, `Veto`, `CicloDecision`), what's
the measured latency before/after? (See `scripts/validacion_end_to_end.py`
for the measurement pattern — estimates aren't accepted here.)

## Notes for reviewers

Anything that would help review this faster — tradeoffs you considered,
parts you're unsure about, etc.
