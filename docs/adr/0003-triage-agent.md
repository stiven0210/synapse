# ADR 0003 — Triage Agent for Operational Veto Escalations

**Status:** Decided, with deliberately narrow scope.

**Context:** We evaluated whether it was worth adding agentic (LLM)
components to SYNAPSE. Of 5 candidate roles discussed (narrating drift
reports, auditing artifacts before publishing, triaging Veto escalations,
explaining individually flagged transactions, proposing uncalibrated
parameters), only one had real value not already covered by deterministic
logic: **triaging the Veto escalations that represent a *system* failure,
not a judgment call about a transaction.**

Of the 4 `DecisionFinal` types (see `src/bitacora_decisiones.py`), only 2
qualify: `SCORE_INVALIDO` and `ERROR_EJECUTOR`. `MODELO` needs no triage
(it's the normal path) and `MONTO_EXCEDE_LIMITE` is self-explanatory from
the data itself (the amount already says everything) — putting an LLM
there would be cosmetic wrapping around something already deterministic.

## Decision

1. **The agent never participates in the hot path.** It is never invoked
   from `CicloDecision.decidir()` or from `Ejecutor`/`Veto`. It runs as a
   later step, over entries already written to the log — the same
   principle as `disparador_recalibracion.py` (slow layer, never fast
   layer).
2. **It never decides or blocks anything.** Its output is a hypothesis for
   a human to verify — it never approves, rejects, or changes any state.
3. **Same 3-layer discipline used in a previous project of ours**,
   replicated here as new, independent code — SYNAPSE shares no code with
   other projects (`CLAUDE.md`):
   - Layer 1 (deterministic, 100% of responses): strict schema — well-formed
     JSON, required fields present, valid enums, confidence in [0,1]. Zero
     cost, no LLM.
   - Layer 2 (deterministic, cheap, 100% of Layer 1's valid outputs): every
     claim in `evidencia_citada` must share real vocabulary with the data
     actually given to the agent (`entrada` + `contexto`) — not
     sophisticated NLP, by design; it catches the obvious case of total
     hallucination disconnected from the data.
   - Layer 3 (selective — only on Layer 2 failure or random sampling): a
     second LLM audits the hypothesis against the same data. Disagreement
     between proposer and auditor is **never resolved by majority vote** —
     the whole cycle is discarded.
4. **Rate limiting** (`src/limitador_llamadas.py`, same pattern as a rate
   limiter from that earlier project): a daily budget; exhausting it raises
   an exception instead of continuing to spend.
5. **Circuit breaker** (`CircuitoTriage`): if the discard rate (Layer 1 or
   failed Layer 3) over the last N triages exceeds a threshold, the agent
   shuts itself off (raises `CircuitoAbierto` before spending on another
   call) — the caller falls back to the flat, deterministic report (the log
   entry as-is, no narrative); the real escalation flow is never blocked.

## Test plan — 3 levels (discussed in conversation, not just unit tests)

1. **Deterministic, 100% testable with an injectable (fake) LLM client, no
   real calls** — same approach used for the equivalent agent in that
   earlier project: invalid schema is discarded at Layer 1, failed
   grounding triggers Layer 3, disagreement discards without a majority
   vote, the rate limiter exhausts its budget, the circuit breaker opens
   under a simulated discard rate.
2. **Cases with a known real cause** (see `tests/test_bitacora_decisiones.py`,
   the 2 real-cause scenarios: a NaN feature → `SCORE_INVALIDO`, a missing
   feature → `ERROR_EJECUTOR`) — verifies that, with a fake LLM client
   returning a correct hypothesis, the agent lets it through; and that a
   hypothesis inventing a cause not present in the real data gets caught by
   Layer 2/Layer 3.
3. **What can't be validated offline**: whether this actually reduces a
   human's effort investigating a real incident can only be known with real
   use — there's no honest way to simulate it against a historical dataset
   with no real incidents. This is documented as pending validation in
   production use, not pretended to be proven by the automated test suite.
