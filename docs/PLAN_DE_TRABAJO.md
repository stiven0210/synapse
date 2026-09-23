# Work plan — SYNAPSE

**Overall status:** Phases 0 through 5 (Domain 1) complete + post-implementation
audit (12 findings, 10 corrected). Domain 2 (per-account personalization)
with tuning and production integration complete -- see
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`.

## Phases (see the approved plan for full detail on each)

### ⏳ Phase 0 — Foundations — in progress
- [x] Repo structure, `CLAUDE.md`, `README.md`, `requirements.txt`.
- [x] Dataset downloaded and validated: `data/raw/creditcard.csv` — 284,807
      rows, 492 frauds (0.17%), columns `Time`, `V1`-`V28` (PCA, anonymized),
      `Amount`, `Class`. Source: OpenML mirror
      (`https://www.openml.org/data/get_csv/1673544/phpKo8OWT`), no
      authentication or payment credentials.
- [x] `docs/adr/0001-policy-artifact.md` — exact shape of the JSON that
      connects Calibrator and Executor.
- [x] `docs/adr/0002-veto-layer.md` — hard invariants for this iteration.

**⚠️ Important Phase 0 finding — data limitation:** the dataset **has no
card/account identifier** (each row is an independent, anonymized
transaction). This means the Executor (Phase 2) will maintain **global**
recursive statistics (over the whole population), not per account — the
two-speed architecture and the latency measurement remain valid, but the
per-account personalization dimension a real fraud system would have is
lost. Full detail and options in `ADR_002`.

### ✅ Phase 1 — Calibrator — complete
- [x] `src/calibrador.py`: loads and cleans the dataset (the `Class` column
      comes with literal quotes from the mirror's ARFF→CSV conversion),
      walk-forward temporal split, logistic regression with
      `class_weight="balanced"` (0.17% fraud rate).
- [x] **Design decision**: coefficients are mathematically un-scaled before
      saving, so the artifact operates directly on raw features — the
      Executor never loads `StandardScaler` or scikit-learn. Verified exact
      (diff < 1e-9 against the original scaled model,
      `test_coeficientes_desescalados_...`).
- [x] Decision threshold: the one that maximizes F1 on the validation split
      (not a fixed 0.5 — with `class_weight="balanced"` the real optimal
      threshold sits far from 0.5, in fact above 0.99 — see the correction
      below, "Real finding: threshold grid capped at 0.99").
- [x] **Real validation metrics** (56,961 transactions, 57 frauds):
      precision 0.93, recall 0.70, F1 0.80, **AUC 0.972**. Consistent with
      published logistic-regression benchmarks on this dataset. (Corrected
      after the walk-forward cross-validation finding — before the fix:
      precision 0.46, recall 0.79, F1 0.58, same AUC.)
- [x] 6 tests (`tests/test_calibrador.py`), including the exact algebraic
      verification of the un-scaling.
- [x] **Important correction**: the original model only used V1-V28+Amount
      — no recursive features. Without that, the state the Executor would
      maintain in Phase 2 would be decorative (computed, but never used in
      the decision). Added `src/features_recursivas.py` with two causal
      features (`monto_ewma_global`, `conteo_ventana_global`) in both batch
      and incremental mode — verified that both modes produce exactly the
      same numbers over the same sequence
      (`tests/test_features_recursivas.py`, 5 tests) to rule out
      train/serve skew by construction. The model recalibrated with these
      two features gives AUC 0.972 (a marginal gain over 0.971, since
      V1-V28 are already very predictive on their own), with non-zero
      coefficients for both — confirmed they do contribute to the
      decision.

### ✅ Phase 2 — Executor — complete
- [x] `src/ejecutor.py`: `validar_artefacto` (invariant 1 of `ADR_002` — no
      valid artifact, no decision), `Ejecutor.decidir()` — numerically
      stable sigmoid, applies the artifact to the state read BEFORE
      updating it (same causal semantics as the batch version).
- [x] **Bug found and fixed during testing, not in production**: the first
      attempt at the end-to-end consistency test compared the Executor
      starting from empty state right at the train/val cut, against batch
      features that already loaded the continuous history from the start
      of the dataset — they diverged by up to 24% on some scores. Fixed by
      making the Executor also process `train` before evaluating `val`,
      just like in real production (state is never reset at an arbitrary
      cut). With the fix, it matches the batch version to 1e-9
      (`test_ejecutor_coincide_exactamente_con_calculo_batch_end_to_end`).
- [x] **Real latency benchmark**: 10,000 decisions, **2.15
      microseconds/decision** measured with `time.perf_counter()`
      (`test_benchmark_latencia_real_microsegundos`) — not an estimate.
- [x] 9 tests (`tests/test_ejecutor.py`).

### ✅ Phase 3 — Bridge — complete
- [x] `src/puente.py`: `publicar()` validates against `ADR_001` before
      writing (never publishes something invalid, or partial), atomic
      write (temp file + `Path.replace`), `leer_vigente()` validates again
      on read.
- [x] 6 tests (`tests/test_puente.py`): round-trip, an invalid artifact
      writes nothing, no leftover temp file, atomic overwrite, missing
      file and corrupt file each raise the right error.

### ✅ Phase 4 — Veto — complete
- [x] `src/veto.py`: the 3 invariants from `ADR_002` — an invalid score
      (NaN/out of range/`None`) vetoes, an amount over the absolute limit
      vetoes **even if the model says it isn't suspicious** (the case that
      actually matters: the veto overrides the model, never the other way
      around).
- [x] 7 tests (`tests/test_veto.py`).

### ✅ Phase 5 — End-to-end validation — complete
`scripts/validacion_end_to_end.py` runs the full pipeline (Calibrator →
Bridge → Executor → Veto) over the **test** split (56,962 transactions, 75
frauds — untouched until now), with recursive state built in real order
from train→val→test.

**Result — SYNAPSE (two layers) vs. a naive baseline (fixed threshold on
Amount, uncalibrated, no recursive features, no veto):**

| | Precision | Recall | F1 |
|---|---|---|---|
| SYNAPSE | 0.93 | 0.67 | 0.78 |
| Baseline (fixed threshold) | 0.00 | 0.00 | 0.00 |

(Corrected after the walk-forward cross-validation finding — before the
fix: precision 0.50, recall 0.79, F1 0.61. See "Real finding: threshold
grid capped at 0.99" below.)

The baseline **doesn't catch a single one** of the 75 frauds in the test
split — verified this isn't a bug: the largest fraud in test is $1,504.93,
well below the 99.9th-percentile Amount threshold from train ($3,000).
It's a real finding, not an artifact of the comparison: in this dataset
fraud concentrates at moderate amounts, exactly the pattern a fixed amount
threshold can't capture and that the V1-V28 (PCA) features plus the
recursive ones DO pick up.

**Real end-to-end latency** (Executor + Veto, 56,962 decisions): **8.3
microseconds/decision** — higher than the isolated Phase 2 benchmark (2.15
µs) because it now also includes the Veto evaluation, but still
microseconds, not milliseconds.

### Phase 6 (future) — Generic contract extraction
Only after a second real domain is built.

## Governance

- No unjustified parameter gets a silent default value.
- Always a temporal split, never a random one, on data with a real time
  order.
- Every updater/formula is tested against a known numeric case ahead of
  time.
- Every latency benchmark reports a measured number, never an estimate.

## Post-implementation audit (with the `rigorous-build` skill)

With the 6 phases and 33 tests green, an adversarial audit was done with
fresh eyes (an independent agent, with no context on who wrote the code)
over the 5 modules in `src/` and the Phase 5 script. It found 12 real
findings; 10 fixed, 2 remain as documented minor items.

### Critical (4/4 fixed)

1. **Nothing forced a decision to go through `veto.evaluar()`** — and since
   `nan >= threshold` is `False` in Python, an invalid score (`NaN`) got
   interpreted as "not suspicious" if anything called `Ejecutor.decidir()`
   directly — exactly what `ADR_002` prohibits. Fixed with
   `src/ciclo.py::CicloDecision` — the single sanctioned entry point, the
   Veto always runs.
2. **"Escalate for manual review" on a missing/corrupt artifact (ADR_002,
   invariant 1) didn't exist as code**, only as an uncaught exception that
   crashed the process. `CicloDecision.decidir()` catches it and turns it
   into a real escalation decision.
3. **Nothing stopped building an `Ejecutor` from an artifact that never
   went through `puente.leer_vigente()`** — without the Bridge's atomic
   write or validation. `CicloDecision` can only be constructed from a
   file path.
4. **`EstadoRecursivoGlobal.tiempos_ventana` used a list with `.pop(0)`,
   O(n) not O(1)**, despite what the docstring claimed — invisible in the
   benchmarks because the data rate (~1.6 tx/sec) never filled the window
   enough to notice. Fixed with `collections.deque` (real O(1)
   `popleft`). Along the way, a related bug was found and fixed: an event
   with `Time` out of order (a late arrival, common in real streaming)
   would get stuck in the middle of the window forever and contaminate
   every count after it — now explicitly rejected (`TiempoFueraDeOrden`).

### Important (6/6 fixed)

5. The "absolute amount" veto didn't use `abs()` — a very negative amount
   (a fraudulent refund/chargeback) never triggered the veto, despite
   `ADR_002` explicitly calling it "absolute." Fixed.
6. The artifact was mutable after validation — with no defensive copy, an
   external mutation of the same dict (logging, monitoring) broke the
   "validated once, safe forever" guarantee. `Ejecutor` now keeps a copy
   (`copy.deepcopy`) of the artifact.
7. `validar_artefacto` only validated shape (lengths, ranges), not content
   — an artifact with a non-numeric coefficient or an empty feature name
   passed validation and blew up with a raw error inside the hot path. It
   now validates types explicitly.
8. `puente.py` imported validation from `ejecutor.py` — an inverted
   coupling direction (the intermediary depended on the fast consumer).
   Extracted `src/artefacto.py`, a neutral module shared by `ejecutor.py`
   and `puente.py`.
9. `sort_values("Time")` wasn't stable — with 1-second resolution and
   >1.6 transactions/second on average, ties are frequent; the default
   quicksort doesn't preserve the CSV's original row order among tied
   rows. Fixed with `kind="stable"`.
10. No controlled handling of an empty CSV or one with missing columns —
    it failed with raw pandas/numpy errors several calls later, not at the
    entry point. `cargar_dataset` now validates explicitly
    (`DatasetInvalido`).

### Minor (1 fixed, 1 documented without a code change)

11. `_mejor_umbral_por_f1` had no protection of its own if `df_val` has no
    positive cases — it relied on `roc_auc_score` blowing up as a side
    effect. `calibrar()` now explicitly validates that `df_train` and
    `df_val` each have at least one positive case.
12. Documented style contradiction, not fixed (zero impact): `ADR_001`
    describes the `features` order as "never by name at decision time,"
    but `Ejecutor.decidir()` does resolve each feature by name — correct
    in outcome, but the written design rationale contradicts the
    implementation. Pending a wording fix in the ADR, not a code change.

**Architecture update**: `scripts/validacion_end_to_end.py` was migrated to
use `CicloDecision` instead of calling `Ejecutor`+`veto.evaluar()`
separately — the same Phase 5 results (precision 0.50, recall 0.79, F1
0.61) reproduce exactly, now through the sanctioned path.

16 new tests from this audit. Full suite: 49 tests.

## How an artifact gets updated in production (recalibration)

`CicloDecision.recargar_artefacto()` is the mechanism: the Calibrator runs
again (with more recent data, or because drift monitoring — still to be
built, see the test plan discussed in conversation — detected the model
went stale), publishes a new artifact via `puente.publicar()`, and
`recargar_artefacto()` picks it up — keeping the accumulated recursive
state (no history is lost by recalibrating) and **never replacing the
current artifact if the reload fails** (a file corrupted mid-write, an
invalid artifact) — graceful degradation, never a crash.

## Drift detection (`src/deriva.py`) — the "when to recalibrate" trigger

Implements **PSI** (Population Stability Index, standard industry
thresholds: <0.1 no drift, 0.1-0.25 moderate, >0.25 recalibrate) and the
two-sample **Kolmogorov-Smirnov test** — both are used because they
measure different things (PSI is the business standard, KS is more
sensitive as an early signal). 9 tests (`tests/test_deriva.py`), including
a negative control (same distribution → PSI≈0) and a positive one (a
5-standard-deviation shift → PSI>0.25, KS p<0.05).

**Real result** (`scripts/deteccion_deriva.py`, train vs. test on the real
dataset): **8 of 31 features already show significant drift** between the
two splits, despite the dataset covering only ~48 hours. The most notable:
`conteo_ventana_global` (PSI=0.82) — makes sense, transaction volume
naturally varies by hour of day, and this feature depends directly on the
transaction rate. `Amount`, on the other hand, shows no drift (PSI=0.008)
— the average transaction amount stayed stable. Confirmed with a positive
control (artificially doubling `Amount`) that the detector does fire when
it should (PSI=4.72, KS p≈0).

### ✅ Automatic recalibration trigger — complete

`src/disparador_recalibracion.py::evaluar_y_recalibrar_si_hace_falta()` is
the job that connects `evaluar_deriva()` with `calibrador.calibrar()` and
`puente.publicar()`: it compares the reference window (the one the current
artifact was calibrated on) against a recent window, and if there's
significant drift (PSI > 0.25 on any feature) it recalibrates on that
recent window and publishes the new artifact. If the recent window doesn't
have enough positive cases to recalibrate safely (`DatasetInvalido`), **it
publishes nothing** — the current artifact stays intact and the error is
reported for human review, instead of silently degrading the model with a
bad calibration. The live reload (`CicloDecision.recargar_artefacto()`) is
still the responsibility of whoever has the cycle running — this module
only publishes, it never replaces the Executor's recursive state.

`scripts/recalibracion_automatica.py` runs the full flow on real data:
publishes the initial artifact (train/val), evaluates drift train vs.
test, and if warranted recalibrates and calls `recargar_artefacto()` on an
already-running `CicloDecision`. **Real result**: it detects the same 8/31
features with significant drift that `deteccion_deriva.py` finds,
recalibrates, publishes, and the live reload confirms success (`True`)
without losing the accumulated recursive state.

5 new tests (`tests/test_disparador_recalibracion.py`): no drift → no
recalibration and no publish; significant drift → recalibrates and
publishes; drift detected but the window has no positive cases → doesn't
publish and reports the error; and the current artifact isn't overwritten
if recalibration fails. Full suite: 61 tests.

**What's still pending, and is a production data/infrastructure problem,
not a problem with this module**: the "reference" and "current" windows
today are passed as explicit parameters (train/test from the historical
dataset) — a real production process still needs to build those two
windows from live traffic (e.g. the last N days vs. the N days before
that) and schedule periodic execution (Task Scheduler / cron / a cloud
job).

### ✅ Decision log — complete (a prerequisite for Veto triage)

Before evaluating whether a triage agent for Veto escalations was worth it
(see the analysis discussed in conversation), a real gap needed closing:
`CicloDecision.decidir()` returns a `DecisionFinal` but nothing persisted
it — there was nothing to investigate an escalation against after the
fact.

`src/bitacora_decisiones.py` — an append-only (JSON Lines) log of every
decision, with `clasificar_razon()` distinguishing 4 types from the actual
text that `veto.py`/`ciclo.py` produce:

- `MODELO` — the normal path, needs no triage.
- `MONTO_EXCEDE_LIMITE` — a business invariant, self-explanatory from the
  data itself (the amount), needs no triage.
- `SCORE_INVALIDO` and `ERROR_EJECUTOR` — **the only two real triage
  candidates**: they represent a *system* failure (the model couldn't
  produce a decision), not a judgment about a transaction.

Deliberately **not invoked from `CicloDecision.decidir()`** — the hot path
is measured in microseconds and shouldn't pay for disk I/O on every
decision; logging is the explicit responsibility of whoever has the cycle
running, the same principle as the artifact reload not being automatic.

**Explicitly documented coupling**: `clasificar_razon()` distinguishes by
the text of `razon`, not by a structured field on `DecisionFinal` (`veto.py`
wasn't touched — it had already gone through 2 audit rounds). If the text
changes, `test_clasificar_razon_cubre_los_textos_reales_de_veto_y_ciclo`
(which calls the real `veto.evaluar()`, not a hand-copied string) breaks
instead of silently misclassifying.

**2 known-real-cause scenarios** (`tests/test_bitacora_decisiones.py`),
end-to-end through the real path (Executor → Veto → CicloDecision → log,
with no monkeypatching, unlike `test_ciclo.py`):

1. A transaction with a `NaN` feature (upstream data corruption) → the
   Executor doesn't raise, but the resulting score is NaN → classified
   `SCORE_INVALIDO`.
2. A transaction missing a feature the current artifact expects (a
   source-vs-artifact schema mismatch) → `KeyError` inside the Executor →
   `CicloDecision` catches it → classified `ERROR_EJECUTOR`.

These 2 scenarios are the basis for later verifying whether a triage agent
points to the right cause — see the 3-level test plan discussed in
conversation (Level 1: deterministic guardrails with known cases; Level 2:
simulated causes like these; Level 3: real value, only verifiable with
production use, not offline).

9 new tests. Full suite: 70 tests.

### ✅ Triage agent for Veto escalations — complete

`docs/adr/0003-triage-agent.md` documents the full decision.
`src/agente_triage.py` implements the same 3-layer discipline used in a
similar agent from an earlier project of ours (new, independent code, not
imported — `CLAUDE.md`): Layer 1 (deterministic schema), Layer 2
(deterministic grounding against the real data from the entry/context),
Layer 3 (selective auditor, disagreement is never resolved by majority
vote). Never invoked from `CicloDecision.decidir()` — it runs afterward,
over `SCORE_INVALIDO`/`ERROR_EJECUTOR` entries from the log. It never
decides or blocks anything, it only proposes a hypothesis for a human to
verify.

`src/limitador_llamadas.py` — daily rate limiting (same pattern as a rate
limiter from an earlier project of ours, independent code). `CircuitoTriage`
(in `agente_triage.py`) — circuit breaker: if the discard rate over the
last N triages exceeds the threshold, the agent shuts itself off before
spending on another call (`CircuitoAbierto`), the caller falls back to the
flat, deterministic report.

**Validated per the 3-level test plan:**
- Level 1 (fake LLM client, no real calls): invalid schema is discarded at
  Layer 1, failed grounding triggers Layer 3, proposer/auditor
  disagreement discards without a majority vote, the circuit breaker opens
  under a simulated discard rate and doesn't spend another call once open.
- Level 2 (known real cause, reusing the 2 scenarios from
  `test_bitacora_decisiones.py`): a hypothesis citing the real data
  (`Amount` as NaN) passes Layer 2 without needing Layer 3; a hallucinated
  hypothesis (an invented cause, with no support in the real data) fails
  Layer 2, triggers Layer 3, and gets discarded before reaching a human as
  a trusted conclusion.
- Level 3 (real use value): explicitly not validated — requires real
  production incidents, not an offline historical dataset.

20 new tests (15 for the agent + 5 for the rate limiter). Full suite: 90
tests.

### ✅ End-to-end triage job — complete

`scripts/triage_veto.py` closes the operational wiring: reads the real
log, filters operational escalations, builds context (the most recent
drift report if one exists), invokes `AgenteTriage` through the rate
limiter, and shows the result to a human (console +
`data/reporte_triage_veto.json`).

**With no real production traffic available**, the job first feeds the
log with the same 2 known-real-cause scenarios from
`tests/test_bitacora_decisiones.py` (a NaN feature, a missing feature, and
one normal decision as a control) — so there's something real to triage
instead of starting from an empty file.

**Graceful degradation, verified with a real run**: with no
`ANTHROPIC_API_KEY` in the environment, the job doesn't invent a fake call
— it explicitly reports that it can't triage with a real LLM and falls
back to the flat report (each operational escalation as-is, no
hypothesis). `CircuitoAbierto` and `PresupuestoAgotado` are handled the
same way if they occur with a real client configured: that entry falls
back to the flat report, the whole job never stops.

**Update: tested with a real `ANTHROPIC_API_KEY` — 2 real bugs found and
fixed, neither detectable with the tests' fake client.**

1. **`crear_cliente_claude()` assumed `respuesta.content[0]` was text.**
   The model can return an extended-thinking block first (`ThinkingBlock`),
   and `.content[0].text` raised `AttributeError`. Fixed: it now explicitly
   looks for `"text"`-type blocks across all of `respuesta.content`,
   without assuming a position.
2. **The model wraps the JSON in a markdown code block**
   (` ```json ... ``` `) despite the prompt asking to "Respond with ONLY
   JSON" — common LLM behavior, not a model error. `_capa1_validar_schema`
   raised `FalloCapa1` with `"Expecting value: line 1 column 1"` (empty
   JSON after the un-stripped fence). Fixed with
   `_despojar_bloque_markdown()`.

4 new tests reproducing both cases with a simulated Anthropic client (no
real network call). Full suite: 110 tests.

**Real, end-to-end result, with the real Claude API** (the same 2
known-cause scenarios): both hypotheses correct and well grounded, passing
Layer 2 without needing Layer 3 —
- `SCORE_INVALIDO` (Amount as NaN): *"The Amount value came in as NaN...
  there's no evidence in the context that this is due to distribution
  drift"* — medium severity, `investigar_pipeline_datos`, confidence 0.55.
- `ERROR_EJECUTOR` (missing Amount): *"The executor failed because the
  'transaccion' object doesn't contain the 'Amount' field... likely caused
  a KeyError"* — high severity, `verificar_esquema_transaccion`,
  confidence 0.75.

This closes Level 2 validation with a real LLM (previously only tested
with the fake client) — the quality and grounding of the hypotheses are
correct in this specific case.

**Pending, explicitly out of this scope**: there's still no real
production traffic source feeding the log — today only the example job
feeds it, with the 2 known scenarios. Level 3 (does this actually reduce a
human's effort?) still can't be validated without real production
incidents.

### ✅ Output-score drift monitoring — complete

`evaluar_deriva()` only looked at drift per input feature — a small shift
spread across many features (each one below PSI 0.25) can move the
model's combined score without any single one showing it, exactly what
per-feature monitoring, by design, can't see.

`src/artefacto.py::calcular_scores()` — a vectorized batch version of the
same formula `Ejecutor.decidir()` applies incrementally (verified
identical to 1e-9 in `tests/test_artefacto.py`, the same train/serve
parity principle as the rest of the project). Offline diagnostics only,
never on the hot path.

`src/deriva.py::evaluar_deriva_score()` — PSI+KS on the score instead of a
feature. `disparador_recalibracion.py` now combines both signals: it
recommends recalibrating if EITHER is significant (feature or score),
using the current artifact to compute the scores.

**A hand-built case demonstrating the real value** (`tests/test_deriva.py`):
a 0.11-standard-deviation shift spread across 30 features left EACH ONE
well below PSI 0.25, but the combined score (the mean of the 30) crossed
to PSI=0.35. Also reproduced through the full trigger
(`tests/test_disparador_recalibracion.py`): 0 features cross alone, the
score does, and the trigger recalibrates anyway.

**Real result, opposite direction — just as valuable**: on the real
dataset, 8/31 features show significant drift while the combined score's
PSI comes out at only 0.07 (no significant drift) — confirms the two
signals are genuinely complementary, not redundant; they can disagree in
either direction.

A real duplication was also removed: `scripts/comparacion_umbral_por_costo.py`
had its own copy of the scoring function — it now uses
`artefacto.calcular_scores()`, a single source of truth for "how an
artifact gets applied to a full DataFrame."

8 new tests (4 in `test_artefacto.py`, 3 in `test_deriva.py`, 1 integration
test in `test_disparador_recalibracion.py`). Full suite: 118 tests.

### ✅ Drift weighted by coefficient magnitude — complete

`evaluar_deriva()` treated every feature the same: any one crossing PSI
0.25 counted equally toward `recomendacion_recalibrar`, with no
distinction between a feature the model actually weighs heavily and one it
barely uses. `deriva.ponderar_deriva_por_coeficiente()` enriches the
report with each feature's coefficient magnitude in the current artifact
and computes `contribucion_ponderada_features_con_deriva` (0 to 1: what
fraction of the model's total weight sits in the features that did
drift).

**Deliberately doesn't replace `recomendacion_recalibrar`** — changing the
automatic trigger with a new threshold on the weighted contribution would
be an empirically unjustified parameter. It's additional information for
human judgment, not a new automatic rule.

`disparador_recalibracion.py` computes it automatically using the current
artifact's coefficients (before a possible recalibration) — if no
artifact is readable yet, it's left as `None` without blocking anything.
`scripts/recalibracion_automatica.py` reports it.

**Real result** (same train/test pair from `deteccion_deriva.py`): of the
8/31 features with significant drift, only **23%** of the model's total
weight (sum of |coefficient|) sits in those features — the drift found
isn't concentrated where the model weighs most.

5 new tests (4 in `test_deriva.py`, including a hand-built numeric case
and a division-by-zero guard; 1 in `test_disparador_recalibracion.py`).
Full suite: 95 tests.

### ✅ Multi-fold walk-forward cross-validation — complete

`src/validacion_cruzada.py` answers whether the AUC/threshold reported in
Phase 1 depends on which particular cut was used, or whether the
calibration process is stable over time. Expanding window (never random
K-fold, which would leak future into past). Diagnostic only — it doesn't
replace `split_temporal()`+`calibrar()`, which remains the only production
path (only one current artifact can exist at a time).

**Real result** (`scripts/validacion_cruzada_walk_forward.py`, 5 folds on
the real dataset): AUC = 0.9775 ± 0.0066 across folds — consistent and
stable, the 0.972 from a single split **isn't a lucky cut**.

**Real finding during this validation: threshold grid capped at 0.99.**
All 5 folds gave the optimal threshold exactly at the edge of the search
grid (`np.linspace(0.01, 0.99, 99)` in `calibrador._mejor_umbral_por_f1`),
with no variation — a signal the real optimum sat outside the explored
range, not that 0.99 was genuinely the best value. Verified by hand: on
the real validation split, F1 kept climbing from 0.58 (at 0.99) to 0.79
(at 0.99999). Cause: with `class_weight="balanced"` and strong class
separation, probabilities cluster near 0 and 1.

**Fixed**: `_mejor_umbral_por_f1` now searches over the actually observed
scores via `precision_recall_curve` (O(n log n), the exact optimal
threshold, not a grid approximation) instead of a fixed grid. This
**changed the real metrics reported in Phase 1 and Phase 5** (see those
sections, already updated) — precision went from ~0.50 to ~0.93, F1 from
~0.61 to ~0.78 (recall dropped from ~0.79 to ~0.67, the net balance
improves). The AUC doesn't change (it's threshold-independent) — it served
as a control confirming the fix didn't touch anything else.

6 new tests (5 in `test_validacion_cruzada.py` + 1 hand-built numeric case
in `test_calibrador.py` that exactly reproduces the bug: scores for both
classes above 0.99, where the old grid forces F1=0.57 and the new one
finds the perfect separation). Full suite: 101 tests.

### Cost-based decision threshold — built with an explicit business
assumption, without replacing production

A real open item (can't be resolved without business data this project
doesn't have): the cost of a false positive (friction/review of a
legitimate transaction that got blocked) isn't in any public dataset. The
cost of a false negative (undetected fraud) is — it's the real `Amount` of
the missed transaction, not an invented average.

`src/costo_decision.py` (`mejor_umbral_por_costo()`) **explicitly
requires** `costo_falso_positivo` as a parameter, with no default —
passing a made-up number as if it were validated data would violate "no
unjustified parameter gets a silent default value" (`CLAUDE.md`). **It
doesn't replace the F1-based threshold `calibrar()` uses** (the only one
this project has justified with real data) — it's a comparison tool, not
a production change.

`scripts/comparacion_umbral_por_costo.py` runs the comparison on the real
test split with an **illustrative** assumption of $5 per false positive
(explicitly flagged as unvalidated in both the code and the report).
**Real result**: the cost-based threshold (0.985) gives a lower F1 than
production's (0.61 vs 0.78 — precision drops from 0.94 to 0.49) but
**reduces the total expected cost from 3,352 to 2,713** — it catches more
real fraud (recall 0.67 → 0.80) because, under that assumption, a false
positive is cheap relative to the cost of letting fraud through.
Concretely demonstrates that optimizing F1 and optimizing expected cost
aren't the same thing — the decision of which to use in production is
still pending a real, not invented, false-positive cost.

11 new tests since the previous commit (5 in `test_costo_decision.py`,
including the hand-built numeric case that demonstrates the F1-vs-cost
divergence; 5 in `test_validacion_cruzada.py`; 1 in `test_calibrador.py`
for the F1 threshold fix). Full suite: 106 tests.

## Pending from the broader test plan (not blocking)

✅ **IEEE-CIS — unblocked 2026-09-13.** It required the user's own
verified Kaggle account, joining the competition, and a real API token;
all three were eventually done and the dataset was downloaded (gitignored,
not redistributed). By then, per-entity personalization had already been
covered with Sparkov (Domain 2), so IEEE-CIS was used for two other
things: the prevalence comparison across continents (Domain 2 doc,
section 12) and the cross-account network-signal test (section 10,
a negative result).

### Finding from the concurrency investigation — documented, no code fix

`CicloDecision`/`Ejecutor` was tested under real concurrent calls from
multiple threads (16 threads, 300 calls each, with `time.sleep(0)` forcing
the GIL to yield to maximize interleaving) — no exceptions or corruption
observed in the runs. But that's not a guarantee: `EstadoRecursivoGlobal`
has no synchronization of any kind, and its correctness depends on
transactions arriving in **a single sequential stream, in `Time` order**
(`TiempoFueraDeOrden` explicitly assumes this). The fact that CPython's
GIL didn't expose visible corruption in these runs doesn't prove it's
safe under different load/hardware/Python version.

**Conclusion, no code change:** `CicloDecision` is designed for a single
thread processing a sequential stream — this restriction was never
explicitly documented before. No locking is needed: the entire point of
O(1), microsecond decisions is that a single thread already covers real
fraud volumes with a huge margin (the dataset averages ~1.6 tx/sec; even
at 10 microseconds/decision, one thread covers >90,000 tx/sec). If real
parallelism is ever needed, the correct approach is partitioning by
account/entity (once that identifier exists, see the Phase 0 limitation),
never sharing a single `Ejecutor` instance across threads. Pending: make
this restriction explicit in `CLAUDE.md`/`ejecutor.py` (no production code
was touched in this round, this was investigation and documentation of
the finding only).

### ✅ Daily runner — the "real deployment, even if small" — complete

With no real production traffic, `src/runner.py::correr_ciclo_diario()`
simulates one: it steps through the historical dataset in batches (one
per invocation), with recursive state persisted between runs (just like a
real service that restarts every day). The first run bootstraps
(calibrates on the first `frac_bootstrap` of the dataset, feeds the
recursive state with that history without logging it as a new decision);
later runs resume, evaluate drift/recalibrate
(`disparador_recalibracion.py`) against the previous batch, and triage new
operational escalations if an agent is configured.

`scripts/runner_diario.py` is the entry point for Task Scheduler/cron
(registration commands included in its docstring). **Real, verified run**:
it processed 10,000 real dataset rows across 2 manual invocations (index
90,442 → 95,442 → 100,442 out of ~284,807). Deliberately **not scheduled**
in Task Scheduler — the user decided to leave it as a manual tool for now;
scheduling it is what would give Level 3 of the triage agent (does it
really save human effort?) real data to evaluate over time, but that would
mean a process running unsupervised and spending a real LLM budget if
`ANTHROPIC_API_KEY` is configured in the system environment.

11 new tests (`tests/test_runner.py`).

## Audit of the agent components (fresh eyes, independent agent)

With the runner and the triage agent freshly built, an adversarial audit
was requested for everything agent-related — `agente_triage.py`,
`limitador_llamadas.py`, `bitacora_decisiones.py`, `runner.py`, and the 2
entry-point scripts — with the same methodology as the previous audits
(an agent with no context on who wrote the code).

**Confirmed solid**: no LLM is reachable from `CicloDecision.decidir()`
(verified by grep, not just reading); Layer 1/Layer 2/Layer 3 run exactly
when they should; proposer/auditor disagreement is never resolved by
majority vote; recursive-state serialization round-trips exactly; no
mutable state shared between instances.

**5 real findings, all 5 fixed:**

1. **🔴 Critical — a crash between logging a decision and persisting the
   runner's new index duplicated rows in the log.** Reproduced end-to-end
   before fixing: simulating the exact interruption, the next run
   reprocessed and re-logged the same row. Fixed: state is now saved row
   by row (not once per batch), and each `registrar_decision()` carries an
   explicit `indice_fila` — on resume, if the row that's next is already
   in the log (from a previously interrupted attempt), it's still run
   through `decidir()` again (causal continuity of the recursive state)
   but not logged again.
2. **🟡 Layer 2 (grounding) freely approved a hypothesis with no cited
   evidence.** `evidencia_citada: []` returned "grounded" automatically —
   this inverted the incentive (citing nothing was safer for a bad
   hypothesis than citing something verifiable). Fixed: empty evidence now
   counts as failed grounding, forcing Layer 3.
3. **🟡 `leer_bitacora()` didn't tolerate a truncated final line**, despite
   its own docstring claiming it did — an interrupted write (the same
   scenario as finding 1) made `json.loads` blow up and **no** entry could
   be read at all. Fixed: each line is parsed in its own try/except, a
   corrupt one is skipped without invalidating the rest.
4. **🟡 Only 2 exception types were caught around the triage call**
   (`CircuitoAbierto`, `PresupuestoAgotado`) — a real network/API error
   (timeout, 5xx) wasn't caught and took down the whole job, even after
   real decisions and state had already been saved. Fixed in `runner.py`
   and `scripts/triage_veto.py`: any exception from the LLM client during
   triage now counts as a failure and the job/report continues — triage is
   advisory, it should never bring down what already worked.
5. **🟡 The rate limiter's daily budget lived only in memory**, but the 2
   real entry points are single-use processes (by design, for Task
   Scheduler: "one invocation = one day") — a manual retry on the same day
   got a fresh budget. Fixed: `LimitadorLlamadasDiarias` accepts an
   optional `ruta_estado` and persists date/counter to disk across
   processes; both entry-point scripts now use it, sharing the same real
   daily budget.

9 new tests reproducing each finding before fixing it. Full suite: 132
tests.

## ✅ Domain 2 — production integration (per-account personalization) —
complete

See `docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` (section 13) for the
full detail -- summary here for continuity with the overall plan.

Built on 2026-09-14, a path parallel to Domain 1 without modifying any of
its files: `src/features_recursivas_cuenta.py` (per-account features,
`monto_ewma_cuenta`/`huella_categoria_cuenta`, the same batch/incremental
parity discipline), `src/artefacto_arboles.py` (a neutral contract with a
v3 lookup table, not raw trees), `src/calibrador_arboles.py` (trains
Gradient Boosting on Sparkov, extracts real thresholds, builds the table),
`src/ejecutor_arboles.py` (`EjecutorArboles`, just `bisect` + a flat
table), `src/puente_arboles.py`/`src/ciclo_arboles.py` (the same Domain 1
guarantees: Bridge always, Veto always -- reuses `veto.py` unmodified,
Executor exceptions escalate to manual review).

Real numbers (60/20/20 split, 1,170,945 rows): AUC-PR 0.898 (VAL) / 0.901
(TEST), lookup table exact bit-for-bit against the real `predict_proba()`
(max diff 3.47×10⁻¹⁸ over the 234,189 TEST rows), real latency 9.84
µs/decision (comparable to Domain 1's 8.33 µs). Two real bugs found and
fixed along the way: NaN wrongly rejected during artifact validation, and
a real `IndexError` when interpolating a NaN score during isotonic
calibration (`bisect` can't compare NaN). Detail on both, and on the
honest deviation from the tuning-stage numbers (real AUC-PR lower than
the 0.966±0.013 averaged over 5 folds), is in section 13 of the Domain 2
doc.

50 new tests, full repo suite: **182 tests, all green**. No commit, no
deploy -- SYNAPSE is still in its testing phase. Still not started:
Path 3 (a second domain genuinely different from fraud, e.g. IoT).
