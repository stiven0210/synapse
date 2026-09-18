# Path 3 — Does the pattern generalize to a domain other than fraud? (exploration)

**Status:** first full pass complete, with a real, positive result, but
with an honest comparison caveat (see "Verdict"). No production code
written — same as Domain 2 before it was built, this is exploratory,
single-run scripts (`data/reporte_camino3_skab.json`), nothing in `src/`.

**Why this exists:** Domain 1 and Domain 2 test the two-speed pattern
against *fraud* with different datasets — never against a genuinely
different domain. This exploration answers whether Calibrator/Executor/
Veto (the architecture, not the code) makes sense outside fraud, using
real-time industrial fault detection as the test domain.

## 1. Dataset: SKAB (Skoltech Anomaly Benchmark)

Of the 3 candidates considered (NASA Bearing, SKAB, Yahoo S5), **SKAB**
was chosen: available on GitHub (`github.com/waico/SKAB`, MIT) with no
access gate or account needed — cloned directly, without the friction that
blocked IEEE-CIS in the Sparkov session. NASA Bearing and Yahoo S5 weren't
tried (no need to — SKAB was available on the first attempt).

The `valve1` subset was used (16 CSV files): an industrial valve on a real
testbed, readings from 8 sensors at 1Hz (accelerometers, current,
pressure, temperature, thermocouple, voltage, flow rate), with `anomaly`
(0/1) and `changepoint` (0/1) labels. The 16 files are actually **a single
continuous session** split into ~20-minute chunks (verified: timestamps
line up exactly from one file to the next, only 3 gaps >2s across 18,160
rows) — they were concatenated and sorted by time as a single stream, the
same way `calibrador.py::cargar_dataset` handles Domain 1.

**18,160 rows, 34.7% anomaly rate** — much more balanced than fraud
(0.077%-3.5% in the datasets already tested), consistent with this being a
common mechanical fault on a testbed, not a rare adversarial event.

## 2. Causal features designed for this domain

Exploratory correlation with `anomaly`: `Volume Flow RateRMS` dominates
(-0.62 — the experiment is literally a valve closing progressively, so
flow drops when there's a fault), `Accelerometer2RMS` has weak positive
signal (0.10), everything else is noise (<0.07).

4 features, same causal pattern (shift, the current row never sees
itself) and the same `lambda=0.98` as Domain 1/2:

- `flow_actual`: raw reading of `Volume Flow RateRMS` (analogous to `amt`).
- `flow_ewma`: causal EWMA of flow (analogous to `monto_ewma_global`).
- `flow_delta`: `flow_actual - flow_ewma` (a sudden drop relative to the
  recent trend — a new feature with no direct fraud analog; it has
  physical meaning specific to this domain: a closing valve looks like a
  sustained drop, not an isolated spike).
- `accel2_ewma`: causal EWMA of the second accelerometer's vibration.

**No "per-entity" notion was forced** (the analog of Domain 2's
"per-account"): the testbed is a single machine, there are no multiple
independent entities in this dataset — that would have been a forced
analogy, so it was dropped, exactly as this task's directive explicitly
asked.

## 3. Real result (logistic regression, 60/20/20 temporal split)

| | VAL | TEST |
|---|---|---|
| AUC-ROC | — | 0.936 |
| AUC-PR | 0.962 | 0.937 |
| Precision (F1-optimal threshold on VAL) | — | 89.0% |
| Recall | — | 87.0% |
| F1 | — | 0.880 |

**Batch/incremental parity: maximum difference 0.00 (exact)** — the
incremental `EstadoRecursivoIoT` reproduces the batch-computed EWMA over
the full 18,160 rows bit for bit, the same kind of test the
`two-speed-decision` skill requires for any new recursive feature.

**Minimal veto designed for this domain:** a simple physical invariant
(`flow_actual < 0` is impossible on this testbed → veto regardless of
score) — it never fired once in test (real data never carries negative
flow), but it exists as a safety net the same way the absolute-amount
invariant does in Domain 1/2.

**Real measured latency** (`time.perf_counter()`, 3,632 test-split
decisions, Executor in pure arithmetic with no sklearn): **5.51
µs/decision** — the same order of magnitude as Domain 1 (8.33 µs) and
real Domain 2 (9.84 µs), confirming the speed pattern isn't specific to
the fraud domain — it's a property of the architecture.

## 4. Honest verdict — with an important comparison caveat

**The architecture generalizes well**: Calibrator (batch, no time limit)
→ artifact → Executor (pure arithmetic, microseconds) → Veto (a
model-independent physical invariant) maps cleanly onto this domain
without forcing anything, and the detection result (F1 0.88, AUC-PR
0.937) is strong.

**But this figure is NOT directly comparable to SKAB's published
leaderboard** (best real result: Conv-AE, F1=0.78) because of two
methodology differences, not architecture differences:
1. The leaderboard evaluates **unsupervised** models (trained only on
   `anomaly-free.csv`, never seeing a fraud/fault label during training) —
   this experiment used **supervised** logistic regression (trained
   seeing the real `anomaly` labels in TRAIN), which is a strictly easier
   problem.
2. The leaderboard is evaluated on the full dataset (35 files: `valve1` +
   `valve2` + `other`) — this experiment only used `valve1` (16 of 35).

**Honest conclusion:** this result cannot support the claim "we beat
SKAB's published state of the art" — that would be the same inflation
already consciously avoided in Domain 2 section 12. What does hold up:
given available labels (same as in fraud, where we do have
`Class`/`is_fraud`), the two-speed pattern detects well and decides fast
in a physically distinct domain — the architecture isn't fraud-specific.
A fair comparison against the leaderboard (same unsupervised setup, full
dataset) is left pending if this work is picked back up.

## Fair comparison against SKAB's real leaderboard (2026-09-14)

Closes the item above: an **unsupervised** comparison, over the **34
files with anomalies** (all of `data/` except `anomaly-free.csv`, which
SKAB's own official protocol doesn't use for this problem), with the
**exact official methodology** from the repo (`core/utils.py::load_preprocess_skab`
for the split, `core/metrics.py::chp_score(metric="binary")` for
F1/FAR/MAR), reusing the same code that produced the already-published
results in `results/results-*.pkl` — not a made-up metric.

**Official split reused as-is:** per file, the first 400 rows
(chronological, unshuffled) are train; the rest is test. Train never
includes the `anomaly`/`changepoint` columns — this matches SYNAPSE's own
principle exactly (the Calibrator learns offline with no label
supervision).

**Unsupervised detector built, SYNAPSE-style:**
- **Calibrator** (offline, once per file): mean and standard deviation of
  the 8 sensor features, over the 400 train rows. Never sees
  `anomaly`/`changepoint`.
- **Executor** (pure arithmetic, O(1) per row): maximum z-score across the
  8 features of the new row against the learned mean/std. Anomalous if it
  exceeds **3 standard deviations** — a standard process-control
  statistical convention, fixed in advance, **never tuned against the
  test labels** (not even looked at until the final evaluation).
- Same prediction smoothing (`rolling(3).median()`) used by SKAB's own
  Isolation Forest notebook, so this method isn't given a post-processing
  advantage the others don't have.

**Real bug found and fixed during this comparison:** the first version
matched my 34 loaded datasets against each published pickle **by list
position** — assuming the order matched. It didn't (verified: dataset
`i`'s date ranges differed between my load and the pickle — `os.walk`
enumerates files in a filesystem-dependent order). This would have
produced an invalid comparison (with numbers that even looked reasonable
— the error wasn't detectable "by eye"). Fixed by matching each dataset by
a real signature (length + first timestamp + last timestamp of its time
index), not by position.

**Real result, with the official methodology and the corrected matching**
(F1 / FAR / MAR, descending by F1; 2 of the 11 pickles — `MSCRED` and
`Vanilla_LSTM` — couldn't be aligned by signature, 0/34 matches, likely
using a different windowing preprocessing than the standard 400-row split;
excluded from the table as unreliable rather than forcing the
comparison):

| Method | F1 | FAR | MAR |
|---|---|---|---|
| Conv_AE (published) | 0.78 | 13.55% | 28.02% |
| MSET (published) | 0.78 | 39.73% | 14.13% |
| **SYNAPSE (z-score 3σ, unsupervised)** | **0.76** | **43.32%** | **14.95%** |
| T2-q (published) | 0.76 | 26.62% | 24.92% |
| LSTM_AE (published) | 0.74 | 29.96% | 25.92% |
| T2 (published) | 0.66 | 19.21% | 42.60% |
| Vanilla_AE (published) | 0.39 | 2.59% | 75.15% |
| Isolation_Forest (published) | 0.29 | 2.56% | 82.89% |
| Arima_anomaly_detection (published) | 0.00 | 0.01% | 100.00% |

**Real Executor latency** (`time.perf_counter()`, 23,801 real test
decisions): **13.05 µs/decision** — same order of magnitude as Domain 1
(8.33 µs) and Domain 2 (9.84 µs).

**Honest verdict:** with the comparison now genuinely fair (same official
split, same official metric, same evaluation code, full dataset, zero
supervision), SYNAPSE's two-speed pattern — implemented as the simplest
possible statistical detector (mean, standard deviation, z-score, 3 lines
of arithmetic) — lands **competitive, not superior**: 3rd place out of 9
evaluable methods, 2 F1 points behind the best (Conv_AE, a neural
network). It's in the same FAR/MAR range as MSET (the leaderboard's other
classical statistical method, not a neural network), which makes sense —
they share the same family of approach. The result confirms the
architecture generalizes, and that even its simplest version isn't out of
step with the state of the art, without needing to inflate the claim to
"we beat it."

## Attempt to improve the detector — Mahalanobis and EWMA smoothing (2026-09-14)

At the explicit request "can't we improve it?", 2 principled improvements
were tried on top of the z-score/3-sigma detector (F1=0.76), keeping the
same discipline of never looking at test labels to tune anything — same
official SKAB split, same metric (`chp_score`, `metric="binary"`), same
matching by real signature against the published pickles.

**Improvement 1 — Mahalanobis distance** instead of an independent
per-feature z-score: uses the covariance matrix learned on TRAIN (via
`np.linalg.pinv`, robust to features with ~0 variance) to capture joint
deviations across the 8 correlated signals, instead of looking at each
one separately. Threshold: the 99% chi-squared critical value with 8
degrees of freedom (`scipy.stats.chi2.ppf(0.99, df=8)` ≈ 20.09) — the
principled multivariate equivalent of the univariate "3-sigma," never
tuned against test.

**Improvement 2 — Causal EWMA smoothing** (`lambda=0.98`, the same
reference value used elsewhere in SYNAPSE) of the score instead of an
instantaneous reading — the EWMA state continues without resetting from
TRAIN to TEST (the same causal-continuity principle as
`EstadoRecursivoGlobal`). The threshold is derived by applying the same
smoothing to TRAIN's scores and taking mean+3·std of that smoothed
distribution — also blind to test. Tested combined with both z-score and
Mahalanobis.

**Real result, all 4 variants (F1/FAR/MAR, 34 files, official metric):**

| Variant | F1 | FAR | MAR |
|---|---|---|---|
| z-score, 3-sigma (baseline, reproduced identically) | 0.76 | 43.3% | 14.9% |
| Mahalanobis, 99% chi-squared | 0.76 | 47.6% | 12.8% |
| z-score + causal EWMA | 0.75 | 62.7% | 6.9% |
| Mahalanobis + causal EWMA | 0.75 | 70.9% | 4.2% |

**Honest verdict: neither improvement raised F1.** Mahalanobis **ties**
plain z-score (0.76) — it shifts the FAR/MAR balance (fewer misses, more
false alarms) but net F1 is the same; with only 8 features, capturing
joint correlation didn't add anything over looking at the maximum
individual deviation. EWMA smoothing **slightly hurt** (0.76→0.75) in
both variants: it does reduce MAR notably (fewer missed anomalies,
14.9%→6.9% and 12.8%→4.2%), but at the cost of a much higher FAR
(43-48%→63-71%) — the EWMA's memory keeps the score elevated for longer
after a real anomaly ends, extending false-alarm periods more than it
gains in detection. This makes sense given the nature of SKAB's anomalies
(mostly point/abrupt events, not gradual drift) — the causal smoothing
that does help in fraud (Domain 1/2, where normal behavior is noisier and
the signal is in the trend) doesn't help the same way here.

**Conclusion:** the simplest detector (z-score, 3-sigma, 3 lines of
arithmetic) remains the best of the 4 variants tried, and the F1=0.76
result (3rd of 9, tied with T2-q) stands as the current number — no real
improvement was found with this dataset, and that's documented as-is
instead of forcing a "yes, it improved" narrative.

## Pending if this is picked back up (not started)

- Build it in `src/` like Domain 2 — parallel modules
  (`calibrador_iot.py`, `ejecutor_iot.py`, etc.) if the user decides it's
  worth it — not done in this pass, left to their decision.
- Explore `changepoint` as a regime-change detection problem instead of
  point-by-point classification — different from the binary problem
  solved here.
- Investigate why `MSCRED` and `Vanilla_LSTM` didn't align (likely using a
  different sequence window than the standard 400-row split) if a
  complete table of all 11 methods is wanted.

Scripts from this exploration: session scratchpad (not persisted in the
repo). Metrics report from the earlier supervised experiment:
`data/reporte_camino3_skab.json` (gitignored).
