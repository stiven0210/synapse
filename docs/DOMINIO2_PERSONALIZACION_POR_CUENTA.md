# Domain 2 — Per-account personalization (research in progress)

**Status:** Exploration, validation, and production integration complete with
real results (see section 13). Still in testing phase -- none of this has
been deployed or connected to live traffic.

**Why this exists:** Phase 0 (Domain 1) documented a real limitation: the
Credit Card Fraud dataset has no account identifier, so the Executor can
only maintain **global** statistics, never per-account personalization. This
research resolves that limitation with a second real dataset, and along the
way answers whether it's worth going beyond the linear model.

## 1. Dataset: Sparkov (synthetic, with account identifier)

Generated locally with [`Sparkov_Data_Generation`](https://github.com/namebrandon/Sparkov_Data_Generation)
(no Kaggle, no account, no verification — a real alternative once IEEE-CIS
got blocked by Kaggle support). Parameters: 100 customers,
2013-01-01 to 2026-09-12, seed 42.

Combined and reduced to the useful columns (61 raw files → 1 CSV):
`data/raw/sparkov_2013_2026.csv` (119MB, gitignored just like `creditcard.csv`).

- **1,170,945 transactions, 99 unique accounts** (`cc_num`), 907 frauds (0.077%).
- Columns: `cc_num`, `amt`, `unix_time`/`trans_date`/`trans_time`, `category`,
  `is_fraud`, `lat`/`long` (customer), `merch_lat`/`merch_long` (merchant).
- **Schema entirely different from Domain 1** — no V1-V28 (PCA), it has
  raw/categorical features instead. Domain 1's `calibrador.py` doesn't apply
  as-is; this is a second real domain, not a file swap.
- **Customer-merchant geographic distance: dropped.** Explicitly tested
  (76.5 km average for fraud vs. 75.1 km for legitimate) — no signal in this
  generator. Not attempted again.
- **12 of 99 accounts are pure "fraud profiles"** (100% of their 7-12
  transactions are fraud — dedicated synthetic identities, not victim
  customers). They account for 119 of 907 frauds. Every validation from here
  on was run **with and without** these accounts, to avoid confusing
  "detecting fraud" with "recognizing an account already known to be bad."

## 2. Real finding: a non-stationarity trap in `category`

With a naive one-hot of `category` + a 60/20/20 temporal split, the combined
AUC fell **below 0.5** (worse than chance). The cause was found and verified,
not just suspected:

| Period | `personal_care` transactions | Frauds |
|---|---|---|
| 2013-2023 (11 years) | ~24 total | **all of them** |
| 2025-2026 | 82,456 | 4 |

From 2013 to 2023 the category has almost no simulated legitimate activity;
the few early occurrences are fraud injections (the generator can label
fraud with any category, regardless of whether that category has legitimate
volume at that point in time). The model learns "personal_care = fraud" from
those ~24 rows, and that rule fails catastrophically once the category
becomes normal in 2025-2026.

**General lesson (not specific to this category):** with only 99 accounts
over 13 years, any category whose legitimate activity isn't stably
distributed over time is a trap for one-hot + temporal split — the same
underlying principle as a "point-in-time universe" in any system that treats
a non-stationary historical distribution as if it were constant.

**Decision:** raw `category` stays out of the model. Pending if revisited:
a point-in-time encoding (fraud rate per category with a recent rolling
window, not a one-hot fixed once and never updated).

## 3. New feature: per-account behavior fingerprint

`huella_categoria_cuenta`: out of the last K=20 transactions **for that
account** (causal, count-based sliding window — same pattern as
`conteo_ventana_global` but per account and per category instead of by
time), what fraction were in the same category as the current transaction.
Accounts with <5 prior transactions fall back to the population frequency
(0.04% of rows) instead of a made-up value.

**Validated, not just designed:**
- Stable over time (annual average 0.98-0.99 across the 14 years) — doesn't
  reproduce the trap from section 2.
- Negative coefficient in logistic regression: an unusual category *for that
  specific account* → more likely fraud (the correct intuition).
- With logistic regression: AUC-ROC went from 0.976 to 0.998 when added.

## 4. AUC-ROC vs. AUC-PR — why the first number was misleading

Real published benchmarks: AUC-ROC 0.86-0.92 typical, **AUC-PR 0.51-0.66**
typical (the industry uses AUC-PR, not AUC-ROC, because with fraud this
imbalanced, AUC-ROC is easily inflated).

With logistic regression (all features, including the fingerprint):

| | VAL | TEST |
|---|---|---|
| AUC-ROC | 0.998 | 0.998 |
| **AUC-PR** | **0.161** | **0.263** |

**Below** the published benchmark (0.16-0.26 vs. 0.51-0.66), despite the
near-perfect AUC-ROC. The linear model was falling short.

## 5. Gradient Boosting — the ceiling was the model, not the data

Diagnosis: same features, same split, `HistGradientBoostingClassifier`
(sklearn, no new dependency) instead of logistic regression.

| | AUC-PR (with fraud profiles) | AUC-PR (without fraud profiles) |
|---|---|---|
| Logistic regression | 0.161 / 0.263 (val/test) | 0.144 / 0.198 |
| **Gradient Boosting** | 0.953 / 0.932 | 0.945 / 0.924 |

Practically no change when removing the 12 trivial accounts — confirms this
is real signal about anomalous behavior in normal accounts, not memorization
of known account identities.

**In business terms (best threshold by F1, without fraud profiles):**

| | VAL | TEST |
|---|---|---|
| Real frauds | 154 | 200 |
| Caught | 139 (90.3%) | 180 (90.0%) |
| False alarms | 24 | 29 |
| Precision | 85.3% | 86.1% |

Well above the published AUC-PR benchmark (0.51-0.66) — with the honest
caveat that it's still a **synthetic** dataset: it's already known that
Sparkov's simulated fraud has a cleaner separation than real fraud (average
amount 8x higher in fraud). This validates the feature engineering and that
the linear model was the bottleneck, not a promise of identical performance
against real adversarial fraud.

## 6. Export to pure arithmetic — validated exact, not approximate

Open question: Gradient Boosting requires an ML library at decision time,
which conflicts with the Executor's core principle (pure arithmetic, no
heavy library on the hot path).

**Resolved:** the internal tree structure of `HistGradientBoostingClassifier`
was extracted (`_predictors[i][0].nodes`, a structured array with
`feature_idx`, `num_threshold`, `left`, `right`, `is_leaf`, `value`,
`missing_go_to_left`) and a pure-Python evaluator was written (comparisons +
sums + a sigmoid, no sklearn). Compared against the original model's
`predict_proba()`: **maximum difference 2.8×10⁻¹⁷** — identical, not a lossy
approximation.

**Implication:** Gradient Boosting is indeed compatible with the Executor,
at an engineering cost (not an architectural one). Still pending to build
for production:

1. New artifact format (a list of trees serialized as JSON, instead of
   `coeficientes`/`intercepto`).
2. A new Executor that walks trees instead of a dot product — same
   `decidir(transaccion) -> dict` contract.
3. Probably a parallel `CicloDecision` (not touching the one already audited
   twice, for a need specific to this domain).
4. Tests verifying byte-for-byte accuracy against sklearn, same as this
   section's manual validation.

### Evaluator latency (2026-09-13) — below Domain 1, far above industry needs

Measured with the same methodology as `scripts/validacion_end_to_end.py`
(decision by decision, not batched), over the 175 trees of the final model
(`max_depth=3, learning_rate=0.05, max_iter=200`):

| Evaluator | Latency/decision |
|---|---|
| Domain 1 Executor (dot product, reference) | 8.33 µs |
| Trees — nodes as Python dictionaries | 157.3 µs |
| Trees — **generated code** (thresholds baked in as literals, compiled once with `compile()`/`exec()`, zero dictionaries on the hot path) | **39.6 µs** |

The dictionary version is 19x slower than Domain 1; the generated-code
version brings it to ~4.75x — the improvement (4x) comes from eliminating
dictionary-access overhead at decision time, not from changing the
algorithm. Accuracy verified equally on both: maximum difference 1.39×10⁻¹⁷
vs. sklearn's `predict_proba()`.

**Honest conclusion:** below the internal standard (Domain 1), but
39.6 µs = 0.04 ms is still ~2,500x faster than a typical card-network
authorization window (tens of milliseconds) — not a real blocker for
production, just a difference in elegance/internal parity. Reports in
`data/reporte_latencia_arboles_dominio2.json` (dictionary version) and
`data/reporte_latencia_arboles_codegen.json` (generated-code version). The
code generator is still exploratory (a one-off script, not yet a `src/`
module) — if production integration is resumed, item 2 of the list above
should use this generated-code approach, not the dictionary one.

### Second optimization round (2026-09-13) — no NaN check, and a lookup table (failed)

**Removing the NaN check from every node:** the 5 features are never NaN in
this pipeline (verified with an `assert` before trusting the result, not
just assumed) — the check was dead code. Result: **39.6 µs → 30.5 µs**
(1.3x faster), a cumulative 5.16x faster than the dictionary version, same
accuracy (1.39×10⁻¹⁷). Report in
`data/reporte_latencia_arboles_codegen_sin_nan.json`.

**Precomputed lookup table (tried, discarded as-is):** idea — discretize the
5 features into quantile bins, evaluate the model once per combination
(393,216 combos, hour indexed exactly 0-23), and at decision time only do a
`bisect` + an array access. Real speed: **1.15 µs/decision — faster than the
Domain 1 Executor** (26.5x faster than the generated code). **But it wrecks
detection**: at the same threshold, precision 91.4%→18.8%, recall
80.0%→27.5%. Cause found: quantile binning collapsed
`huella_categoria_cuenta` (the most important feature, 0.945) to **only 1
real bin** out of the 16 requested, and `conteo_ventana_global` to 2 out of
8 — both features have very concentrated/discrete distributions where the
quantile cuts all land on the same value. The speed ceiling (1.15 µs) is
real and worth pursuing, but naive quantile binning doesn't work for these
specific features — the binning needs to be redesigned (treat
`huella_categoria_cuenta` by its real discrete values, not continuous
quantiles) before this route is usable. Report in
`data/reporte_tabla_lookup_dominio2.json`.

**Latency exploration status (updated, see third round below):** 30.5 µs
(generated code, no NaN) was the best result without a table. The lookup
table was fully resolved in the third round — see below.

### Third round (2026-09-13) — a lookup table with the model's real thresholds: exact and faster than Domain 1

The v1 table (naive quantiles, 16 bins per feature) **wrecked detection**
(precision 91.4%→18.8% at the same threshold) because quantile binning
collapsed `huella_categoria_cuenta` (the most important feature) to 1 real
bin — that feature has only 40 discrete values, and 97.7% of rows land
exactly on 1.0, so the quantile cuts all fell on the same point. The v2
(exact indices for `hora`/`conteo_ventana_global`/`huella_categoria_cuenta`,
quantiles only for `amt`/`monto_ewma_cuenta`) improved a lot (precision
85.2% vs. 91.4%, 1.90 µs) but was still an approximation.

**Definitive solution (v3):** instead of quantiles or ad-hoc indices, the
table's bins are defined by **the real thresholds the model itself
learned** — extracted from the nodes of the 175 trees (`amt`: 44 unique
thresholds, `hora`: 14, `conteo_ventana_global`: **only 1** — which explains
why its individual importance was so low, `monto_ewma_cuenta`: 26,
`huella_categoria_cuenta`: 18). Since a tree never distinguishes two values
that fall within the same interval between two consecutive thresholds, the
table ends up **exact by construction**, not approximate — and at the
smallest possible size: 692,550 combinations (vs. millions if using
arbitrary resolution).

**Real bug found and fixed along the way:** the first version of v3 gave
`diff_max=0.43` on 33 of 234,189 rows — cause: `bisect.bisect_right` routes
a value *exactly equal* to a threshold to the right-hand bin, but the tree
compares with `<=` (goes left). Fixed by switching to `bisect.bisect_left`,
re-verified over the full 234,189 rows (not a sample): **MAE=0, maximum
difference=0, correlation=1.000000** — a bit-perfect reproduction.

| Evaluator | Latency/decision | Accuracy |
|---|---|---|
| Dictionaries | 157.3 µs | Exact |
| Generated code, no NaN | 30.5 µs | Exact |
| Table v1 (naive quantiles) | 1.15 µs | Broken (91.4%→18.8% precision) |
| Table v2 (partial indices) | 1.90 µs | Approximate (~6 pts of precision) |
| **Table v3 (model's real thresholds)** | **1.53 µs** | **Exact (bit-perfect)** |
| Domain 1 Executor (reference) | 8.33 µs | — |

**Final result: 5.43x faster than the Domain 1 Executor, without losing a
single bit of accuracy.** Reports in `data/reporte_tabla_lookup_dominio2.json`
(v1), `data/reporte_tabla_lookup_v2.json` (v2), `data/reporte_tabla_lookup_v3_exacta.json`
(v3, final). The table-building script is still in the session scratchpad —
pending a move into `src/` if production integration is resumed (section 6,
item 2): the v3 table should be the real mechanism of the new tree Executor,
not the generated code or the dictionaries.

**Note for production, if resumed:** the table's size (692,550 combinations,
~5.5MB in float64) depends on how many thresholds the model learns — if
retrained with more trees or a different `max_depth`, the table needs to be
rebuilt (it's not static); the same threshold generator + table should be
part of the artifact publishing pipeline, not a manual step.

## 7. Ongoing tuning — feature importance (finding, not concluded)

At the user's explicit request, production integration (section 6) is
**paused until the model is fully tuned** — "before starting any domain we
need to finish tuning the model completely."

`permutation_importance` (scoring=`average_precision`, on val) of the
Gradient Boosting model from section 5:

| Feature | Importance |
|---|---|
| **huella_categoria_cuenta** | **0.945** |
| **amt** | **0.552** |
| hora | 0.089 |
| monto_ewma_cuenta | 0.076 |
| conteo_ventana_global | 0.028 |
| monto_ewma_global | 0.005 |
| conteo_ventana_cuenta | 0.002 |

**Honest finding, still unresolved:** two features (fingerprint + amount)
concentrate almost all the importance. The 4 recursive features inherited
from the Domain 1 approach (EWMA/count, global and per account) contribute
very little once the non-linear model already has fingerprint+amount — this
could be genuine redundancy (Gradient Boosting infers from
fingerprint+amount the same thing those features would contribute) or a
signal that those features need better design for this specific domain. No
conclusion yet.

## Tuning plan — complete, all 5 steps closed

### Step 1 — Multi-fold cross-validation

5 walk-forward folds (expanding window, same criterion as Domain 1's
`validacion_cruzada.py`), the 7-feature set, `max_depth=6`:

AUC-PR per fold: 0.913, 0.948, 0.932, 0.962, 0.966 → **mean 0.944 ± 0.020**.
Stable across the 5 temporal cuts — not an accident of a single split.

### Step 2 — Overfitting check

Train-val gap per fold: +0.008, -0.002, -0.018, +0.017, +0.002 —
negligible, and in 2 of 5 folds val beat train. **No overfitting.**

### Step 3 — Low-contribution features

Multi-fold ablation (3 feature sets):

| Set | Mean AUC-PR |
|---|---|
| A: full (7 features) | 0.944 ± 0.020 |
| B: without `monto_ewma_global` + `conteo_ventana_cuenta` (5 features) | 0.939 ± 0.016 |
| C: only `amt` + `huella_categoria_cuenta` (2 features) | 0.821 ± 0.019 |

B ≈ A (difference within noise) → those 2 features can be dropped with no
real loss. C drops for real → `hora`, `monto_ewma_cuenta`, and
`conteo_ventana_global` do add value together despite their modest
individual importance (section 7).

**Final production set: 5 features** — `amt`, `hora`,
`conteo_ventana_global`, `monto_ewma_cuenta`, `huella_categoria_cuenta`.

### Step 4 — Hyperparameter search

A 12-combination grid (`max_depth` ∈ {3,5,8}, `learning_rate` ∈
{0.05,0.1}, `max_iter` ∈ {200,300}) × 3 folds, on the 5-feature set.

Real finding: **`max_depth=8` gets worse and becomes unstable** (AUC-PR
0.84-0.90, std up to 0.084) — clear overfitting with deep trees given only
99 accounts. `max_depth=3` is better and more stable than the 6 originally
used. `max_iter` doesn't matter (200 = 300, converges earlier).

**Winning configuration:** `max_depth=3, learning_rate=0.05, max_iter=200`.
Final 5-fold validation with this configuration: 0.943, 0.975, 0.975,
0.960, 0.978 → **AUC-PR 0.966 ± 0.013** — better and more stable than the
original configuration (0.944 ± 0.020).

### Step 5 — Probability calibration

**Real finding:** raw scores are severely miscalibrated in the mid-high
range — an expected consequence of artificially balancing classes with
`sample_weight` during training (it affects probabilities, not the
ranking, which is why AUC-PR didn't expose it).

| Score percentile (test) | Average raw score | Real fraud rate |
|---|---|---|
| 99.5-99.9% | 45.5% | 0.85% |
| 99.9-99.99% | 98.9% | 79.9% |
| 99.99-100% | 99.95% | 100% |

Fixed with `IsotonicRegression` (fit on VAL, applied to TEST, never the
other way around): **Brier score 0.00124 → 0.00013 (9.5x better)**, minimal
cost in AUC-PR (0.973 raw vs. 0.961 calibrated on that specific split — the
robust 5-fold validation, 0.966±0.013, is the number that counts).
Calibrated percentiles land close to the real rate (0.59% / 79.2% / 100%
vs. real 0.85% / 79.9% / 100%).

**Necessary if `costo_decision.py` (Domain 1) is ever used** with this
model — that tool assumes the score is a real probability, not just a good
relative ranking.

## Status: tuning complete, production integration complete (see section 13)

All 5 steps closed with a positive result on each. Final model: Gradient
Boosting, 5 features, `max_depth=3, learning_rate=0.05, max_iter=200`, with
isotonic calibration as a post-processing step. Production integration
(artifact + Executor + Cycle, section 6) was built on 2026-09-14 — see
section 13 for the detail and the real reproduced figures.

## 8. Validation against the real LatAm market — an important prevalence correction

A search was made for a public, downloadable transactional fraud dataset at
the Latin America level: **none exists** (confirmed against Kaggle,
academic literature, and Colombia/Mexico/Brazil government open-data
portals). Two real, useful sources were found instead:

**a) Real academic benchmark** (Rugeles Diaz et al. 2025, *Financial
Innovation*, real data from a payment gateway across 7 LatAm countries,
221,292 transactions from 2022, real fraud rate 1%): an honest evaluation on
imbalanced data (their Table 14, not Tables 11-13, which use a rebalanced
sample and so inflate the metrics) gives **AUC 0.85-0.90, F1 0.60-0.65** per
merchant segment. The raw dataset is proprietary, not downloadable.

**b) Official real fraud-rate figure** (Central Bank of Brazil, public Pix
API, `EstatisticasFraudesPix`): confirmed fraud ≈ **4.5-5.4 cases per
100,000 transactions (≈0.005%)**, far below the 0.077% fraud rate in our
synthetic Sparkov dataset (907/1,170,945).

**Honest correction this forces:** the precision reported in section 7
(83-97% depending on the recall chosen) assumes Sparkov's synthetic
prevalence. Recalculating with the real market prevalence (~0.005%, via
Bayes, keeping the model's same TPR/FPR):

| Recall | Precision (synthetic prevalence 0.07%) | Precision (real prevalence ~0.005%) |
|---|---|---|
| 90% | 83% | 26% |
| 80% | 91% | 41% |
| 76% | 97% | 69% |

**Conclusion:** the model is still useful and comparable to or better than
the real LatAm academic benchmark (F1 0.60-0.65), but the earlier claim of
"well above the market" was inflated by the synthetic dataset's artificial
imbalance, not by real superior model capability. Before any real
deployment, the decision threshold (and possibly `costo_decision.py`) needs
to be recalibrated against the real prevalence expected in the target
market, not the training dataset's.

## 9. Full curve recalculated against real prevalence (2026-09-13)

Training was redone (same final model: Gradient Boosting,
`max_depth=3, learning_rate=0.05, max_iter=200`, 5 features, 60/20/20
temporal split) to get TPR/FPR across the full threshold curve on the TEST
set (234,189 rows, 200 real frauds), not just the 3 points from section 8.
Precision recalculated via Bayes with the same formula
(`TPR·π' / (TPR·π' + FPR·(1-π'))`), `π'` = 4.95×10⁻⁵ (midpoint of the BCB
Pix range, 4.5-5.4/100k):

| Recall | Precision (synthetic) | Precision (real market ~0.005%) |
|---|---|---|
| 5-40% | 100% | 100% |
| 45% | 97.8% | 72.3% |
| 50% | 98.0% | 74.3% |
| 60% | 96.0% | 58.4% |
| 70% | 94.6% | 50.5% |
| 80% | 91.4% | 38.2% |
| 90% | 83.0% | 22.1% |
| 95% | 67.0% | 10.5% |

Full curve (19 points, every 5% of recall) in
`data/reporte_precision_recall_prevalencia_real.json`.

**New finding, not previously reported:** between 5% and 40% recall the
model has **zero false positives** (100% precision at both prevalences) —
almost certainly the trivial detection of the 12 "pure fraud profile"
accounts (section 1), not signal that generalizes to fraud in normal
accounts. The sharp precision drop starts right after 40-45% recall, which
is where those trivial accounts run out and the model shifts to detecting
real fraud in normal accounts — that's the range that matters for judging
the model, not the "apparent" 90% precision in the low part of the curve.

Consistent with the 3 earlier points from section 8 (recall 90/80% → 83%/91%
synthetic, 22%/38% real — section 8 gave 26%/41%, an expected difference
from split/seed variation, not a contradiction).

## 10. Cross-account network signals (collusion) — a negative attempt (2026-09-13)

An old item from the "Explicitly pending" list below: test whether there is
collusion structure (accounts connected by shared merchant/device) using the
fractal dimension of networks (box-covering, Song/Havlin/Makse 2005,
*Nature* — a real, established method for measuring self-similarity in
graphs, not a metaphor).

**Attempt 1, dropped without running:** use Sparkov's `merch_lat`/`merch_long`
as a merchant-identity proxy. Verified that **every transaction has a
unique** lat/long combination (1,170,945 of 1,170,945) — Sparkov adds GPS
noise to every transaction, so location doesn't identify the merchant.
Recovering the real `merchant` field would require regenerating the entire
dataset with Sparkov_Data_Generation — **tried and dropped for real
slowness**: a test with just 2 customers × 10 days took >13 minutes without
finishing on this machine (the README's benchmark assumes 64 cores/128
threads). Also, inspecting `datagen_transaction.py`, the merchant is
assigned via an independent `random.sample()` per transaction — it's not
reconstructible from the already-generated data without re-simulating the
entire random sequence.

**Attempt 2, run against IEEE-CIS (already downloaded, no wait involved):**
an account(`card1`)–account graph via shared `DeviceInfo` (identity table).
Generic OS/browser names (`Windows`, `iOS Device`, `MacOS`, Firefox
versions) were filtered out as not real devices — left with 26,529 rows
with a specific device model, 1,203 devices shared by >1 account. Resulting
graph: 2,254 nodes in the largest connected component, 96,773 edges.

**Result: negative.** The real graph's fractal dimension (3.91, R²=0.93) is
**practically identical** to that of a random control graph with the same
degree sequence (4.05, R²=0.89) — the method's central diagnostic test
(real vs. null model with the same degrees) found no difference. No
evidence of collusion structure with this proxy. Likely cause: `DeviceInfo`
is a **phone model** (e.g. "Samsung Galaxy S7"), not a unique physical
device identifier — hundreds of unrelated accounts "share" the same model
simply for being a popular phone, the same underlying problem that ruled
out the lat/long route (a popularity proxy, not a real identity one).

**To do this properly** (not attempted, real non-trivial engineering):
combine `card1+card2+card5+addr1+D1` as a pseudo account identity (a known
technique from that Kaggle competition's winning solutions) and several
`id_3x` fields together as a device fingerprint, instead of a single field.
Full report in `data/reporte_grafo_fractal_ieee.json`.

## Explicitly pending, not started

- Test the model/approach against North American, Central American, and
  European fraud datasets — **CLOSED 2026-09-13**: IEEE-CIS (North America,
  3.50% prevalence) and the classic European ULB/`creditcard.csv` dataset
  (0.17% prevalence) were tested. The inflated-prevalence-in-synthetic/curated
  pattern **isn't LatAm-specific** — it repeats across all three continents,
  with the most extreme gap in IEEE-CIS (700x the real rate vs. 14x in
  Sparkov and 34x in the European one). See the comparison table in the
  2026-09-13 session.
- Build the production integration from section 6 (tree artifact +
  compatible Executor + isotonic calibration as part of the artifact) —
  **CLOSED 2026-09-14**, see section 13.
- Cheap exploratory validation of cross-account network signals (collusion)
  — requires recovering the `merchant` field, dropped when the files were
  combined. **Attempted 2026-09-13, negative result, see detail above.**
- Point-in-time encoding of `category` (section 2), if that feature is
  revisited — **CLOSED 2026-09-14, positive result**, see section 15.
- Behavior fingerprint with EWMA-style decay instead of a fixed-count
  window — **CLOSED 2026-09-14, negative/inconclusive result**, see
  section 15.

## 11. Saerens-Latinne-Decaestecker (SLD) — a more rigorous prevalence correction (2026-09-13)

An industry-standard method (iterative EM) for estimating/correcting real
prevalence when it differs from the training prevalence — more rigorous
than the single-point Bayes estimate from sections 8/9.

**First attempt, failed (100% error):** feeding SLD the model's raw
(uncalibrated) scores made the EM diverge to an estimated prior of ~0
instead of the test's real prevalence (0.0854%). Cause: SLD assumes
well-calibrated input probabilities, and it was already known (tuning
Step 5) that the raw scores are NOT calibrated.

**Fixed:** calibrate with `IsotonicRegression` (fit on VAL, same as Step 5)
before running SLD, using VAL's real prevalence as the training prior
(not an assumed 50/50). With that, SLD estimated the test prevalence
**without seeing the labels**: 0.0896% vs. the real 0.0854% — **4.87%
relative error**. A real, positive validation: the method works once its
calibration assumption is respected.

**Own interpretation error, found and corrected on the spot:** an attempt
was made to recompute "precision by recall level" using SLD's rescaled
probabilities against the assumed real prevalence — the result matched the
*synthetic* column from section 9 (91.4% at 80% recall), not the real-market
column (38.2%). Reason: precision TP/(TP+FP) computed over the test's real
labels always reflects that test's actual composition (synthetic
prevalence) no matter what recalibrated probability gets assigned to each
row — rescaling the score doesn't change the labels or the counts. **The
precision/recall-against-real-prevalence curve from section 9 (Bayes over
TPR/FPR) is still the correct one and doesn't change because of this.**

**What SLD does genuinely add, validated and new:**
1. Estimating real prevalence without needing external data (like the BCB
   Pix figure) — useful if in the future there's no known a priori market
   rate, only unlabeled production data.
2. A rescaled probability per individual transaction (not just aggregate
   points on a curve) — useful for an absolute business threshold ("flag
   for manual review if adjusted probability > X%"), complementary to
   section 9's curve, not a replacement.

Full report (with both attempts) in `data/reporte_sld_prior_shift.json`.

## 12. Benchmark against real datasets (IEEE-CIS, ULB) and concept drift (2026-09-13)

Motivated by honestly evaluating whether the approach "competes with the
market" on detection, not just on latency (which is already confirmed,
section 6).

**Benchmark on ULB `creditcard.csv` (real, Europe):** the same methodological
discipline (temporal split, Gradient Boosting, hyperparameter grid) applied
to this dataset's real data (not Sparkov). Result: AUC-ROC 0.982, AUC-PR
0.780 (temporal split) / 0.817 (random split, matching the typical
methodology of public notebooks). **Below** the best known published result
(tuned XGBoost, AUC-PR 0.9133) — neither tuning hyperparameters (grid of 12)
nor matching the split closed that gap. Reports in
`data/reporte_benchmark_ulb*.json`.

**Honest conclusion:** on detection, **we don't beat the published state of
the art** on real data — we're competitive, not superior. The real,
confirmed advantage remains latency (section 6), not detection.

**Concept drift (Sparkov, within the test range itself, 2023-2026):**
AUC-PR by calendar year: 2024→0.971, 2025→0.922, 2026(partial)→0.892 —
**a consistent, monotonic drop**, while AUC-ROC stays ~1.0 across all
three (confirming again that AUC-ROC hides the deterioration, same as in
section 4). With ~4 points of AUC-PR drop per year, **retraining at least
annually would be reasonable** in a real deployment — although with only
35-106 frauds per year the exact slope has real statistical uncertainty,
the direction (always dropping) is consistent across the 3 periods, not an
accident of one cut. Report in `data/reporte_concept_drift_sparkov.json`.

## 13. Production integration — built and validated (2026-09-14)

Closes the pending item from section 6: the tree artifact + compatible
Executor + a parallel `CicloDecision`, now in `src/`, not scratchpad
scripts. **A path parallel to Domain 1, without touching any existing file**
(`artefacto.py`, `ejecutor.py`, `ciclo.py`, `calibrador.py`,
`features_recursivas.py`, `veto.py`, `puente.py` remain untouched) —
`veto.py::evaluar` is reused unmodified (model-independent by design), as
is `features_recursivas.py::EstadoRecursivoGlobal`/`calcular_features_recursivas_batch`
for `conteo_ventana_global` (the same global metric from Domain 1).

**New modules:**
- `src/features_recursivas_cuenta.py` — `monto_ewma_cuenta` and
  `huella_categoria_cuenta` per account (`EstadoRecursivoPorCuenta` for the
  Executor, `calcular_features_recursivas_cuenta_batch` for the Calibrator),
  with the same batch/incremental parity discipline as Domain 1 (verified
  with an explicit test, not just by sharing code).
- `src/artefacto_arboles.py` — a neutral contract: instead of serializing
  the 175 trees to walk them on the hot path (slower, see section 6), the
  artifact directly stores the **v3 lookup table** (the model's real
  thresholds + a flat table) and the isotonic calibration as `x`/`y`
  breakpoints (linear interpolation in pure Python, verified identical to
  `IsotonicRegression.predict(out_of_bounds="clip")`).
- `src/calibrador_arboles.py` — trains a `HistGradientBoostingClassifier`
  on `data/raw/sparkov_2013_2026.csv` (`class_weight="balanced"` is
  natively supported by the constructor in sklearn 1.9.1, no manual
  `sample_weight` needed), extracts the nodes' real thresholds, and builds
  the table by evaluating `predict_proba` once per bin combination
  (vectorized with `np.meshgrid`, not pure Python — that runs during
  calibration, never on the hot path).
- `src/ejecutor_arboles.py` — `EjecutorArboles`, the fast layer: only
  `bisect_left` + flat-table indexing (no tree walking, no node
  dictionaries), reuses `EstadoRecursivoGlobal` and `EstadoRecursivoPorCuenta`.
- `src/puente_arboles.py` / `src/ciclo_arboles.py` — atomic writes and the
  single sanctioned entry point, same guarantees as Domain 1 (the artifact
  always goes through the Bridge, the result always goes through
  `veto.py`, any Executor exception escalates to manual review). `veto.py`
  expects the `Amount` key (Domain 1's contract) -- `CicloDecisionArboles`
  adds an `amt` alias when calling `evaluar_veto`, without modifying
  `veto.py`.

**Real reproduced figures** (60/20/20 temporal split over the 1,170,945
rows, `random_state=42`):

| | VAL | TEST |
|---|---|---|
| n | 234,189 | 234,189 (200 real frauds) |
| Precision | 88.7% | 87.8% |
| Recall | 81.8% | 83.0% |
| F1 | 0.851 | 0.853 |
| AUC-ROC | 0.9999 | 0.9999 |
| AUC-PR | 0.898 | 0.901 |

**Real thresholds extracted from the 175 trees** (defining the table's
bins): `amt` 42, `hora` 13, `conteo_ventana_global` 1, `monto_ewma_cuenta`
27, `huella_categoria_cuenta` 15 → a table of `43×14×2×28×16 = 539,392`
combinations (vs. section 6's 692,550 -- that number came from the model
prior to Step 4's hyperparameter search, with `max_depth=6`; the final
`max_depth=3` model generates simpler trees with fewer unique cuts per
feature, a smaller table, same principle).

**Bit-for-bit accuracy, verified over the full 234,189 rows of TEST**
(replayed from the very start of the ENTIRE history -- train+val+test in
order, not just the test slice, so the per-account/global recursive state
starts the same way it would in real production, not empty at an arbitrary
cut point): MAE = 1.94×10⁻²³, maximum difference = 3.47×10⁻¹⁸,
correlation = 1.000000 — bit-perfect against real `predict_proba()` +
isotonic calibration, replicating section 6's result.

**Real measured latency** (`time.perf_counter()`, the same methodology as
`scripts/validacion_end_to_end.py`, 1,170,945 decisions): **9.84
µs/decision** for the full `EjecutorArboles.decidir()` (table + both
recursive states, global and per account) — comparable to Domain 1's
Executor (8.33 µs). Isolating just the lookup table (no state maintenance,
same methodology as section 6): **3.51 µs/decision**. Both numbers are
**higher than the 1.53 µs reported in section 6** — that figure measured
only the table's arithmetic with features already computed, without the
real cost of maintaining `EstadoRecursivoPorCuenta` (a `deque` of 20
categories + a per-account dictionary) or of `hora_utc()`
(`datetime.fromtimestamp` per decision). The honest number comparable
against Domain 1 is the full-path one: **9.84 µs**, not 1.53 µs — still
~5,000x faster than a typical card-network authorization window, not a
real blocker.

**Honest deviation from the tuning figures:** the real AUC-PR
(0.898/0.901 val/test) comes in below the 0.966±0.013 reported in tuning
Step 4 (average of 5 *walk-forward* folds with an expanding window) and the
0.973/0.961 points from Step 5. Most likely explanation, not exhaustively
verified: here a single fixed 60/20/20 split is used (higher variance than
a 5-fold average), plus minor reimplementation differences (the exact
population-fallback rule under `min_historial=5`, `random_state=42`,
sklearn version). It wasn't investigated further (e.g. running the same
5-fold walk-forward on this implementation) because that was outside this
task's scope — no deeper investigation was done because it wasn't part of
this task's scope.

**Domain 1's 132 tests remain green, with no existing file modified.**
50 new tests (`tests/test_features_recursivas_cuenta.py`,
`tests/test_artefacto_arboles.py`, `tests/test_puente_arboles.py`,
`tests/test_ciclo_arboles.py`, `tests/test_calibrador_arboles.py`,
`tests/test_ejecutor_arboles.py`) — full suite: **182 tests, all green**.
This includes the bit-exact test above (automatically skipped if
`data/raw/sparkov_2013_2026.csv` isn't present) and an equivalent fast
version over a small synthetic dataset so runs don't depend on the real
119MB CSV every time.

**Two real audit findings along the way, fixed before closing:**
1. `artefacto_arboles._es_numero` rejected `NaN` during shape validation
   -- inconsistent with Domain 1's `artefacto.py` (which does accept NaN
   there, leaving the real protection to `veto.py`). Fixed so the contract
   is the same across both domains.
2. `_interpolar_isotonica` (the calibration's pure linear interpolation)
   crashed with an `IndexError` on a `NaN` input score -- `bisect` can't
   compare NaN, and neither `x <= xs[0]` nor `x >= xs[-1]` is true for NaN,
   so it fell through to the normal interpolation path and ran off the
   array. Fixed with an explicit check up front (`math.isnan`) that returns
   a clean NaN, same as Domain 1's `Ejecutor._sigmoide`
   (`math.exp(nan)` doesn't raise). Found by a test that deliberately built
   an artifact with `NaN` in the table.

**No existing ADR was touched** (they belong to Domain 1, still valid as
written). **No commit or deploy was made at the time this section was
written** -- see section 14 for the close-out of the AUC-PR gap
investigation, and `docs/CAMINO3_GENERALIZACION_IOT.md` for Path 3's
close-out (the IoT domain), completed after this section.

## 14. AUC-PR gap investigation — root cause found: per-account history dependency (2026-09-14)

Section 13 left uninvestigated why production's real AUC-PR (0.898/0.901)
comes in below tuning Step 4's 0.966±0.013. This was revisited, and 4
hypotheses were ruled out with direct evidence until a real explanation
was found (not just "probably variance").

**First, the gap was confirmed to be real, not single-split variance:**
running the same 5-fold walk-forward (`frac_train_inicial=0.5`, same as
`validacion_cruzada.py`) over the production code
(`src/calibrador_arboles.py`): AUC-PR per fold 0.877, 0.879, 0.931, 0.935,
0.947 → **mean 0.914 ± 0.030**. The single split (0.898/0.901) is in line
with this average, not a rare case.

**4 hypotheses tested and ruled out, with direct evidence:**

1. **Isotonic calibration** (production code always applies it before
   measuring AUC-PR; the doc's Step 4, before Step 5's calibration,
   probably measured raw scores) — measured directly: the raw vs.
   calibrated difference is only ±0.003-0.005 per fold. Doesn't explain
   the ~0.05 gap.
2. **`class_weight="balanced"` vs. manual `sample_weight`** (the doc talks
   about "balancing with sample_weight," production code uses the
   constructor parameter) — tested on 2 folds: they give **identical**
   results (0.8814 and 0.9505 in both cases). Without balancing, it drops
   to 0.82/0.79 — confirms balancing is necessary and already correctly
   applied, but isn't the cause of the gap.
3. **Data leakage in `frecuencia_poblacional_categoria`** (was the
   exploratory script computing it over the full dataset instead of just
   TRAIN?) — deliberately tested with leakage: the leaked result is
   **equal to or slightly worse**, not better. Ruled out.
4. **Bug in `huella_categoria_cuenta`'s calculation** — checked against the
   exact figure the doc itself had already reported in section 6 (third
   round): 40 unique values, 97.7% of rows at exactly 1.0. Production code
   reproduces those numbers **exactly**, along with total rows (1,170,945),
   total frauds (907), and accounts (99), all identical to the rest of the
   document. The model's most important feature is correctly computed.

**Root cause found — per-account cold start:** varying `frac_train_inicial`
(how much "warm-up" history each account has before the first validation
fold starts):

| `frac_train_inicial` | AUC-PR per fold | Mean | Std dev |
|---|---|---|---|
| 0.5 (cold start, new accounts) | 0.877 – 0.947 | **0.914** | ±0.030 |
| 0.6 | 0.869 – 0.964 | **0.926** | ±0.033 |
| 0.7 (broad history, established accounts) | 0.919 – 0.991 | **0.940** | ±0.026 |

A **monotonic and consistent** trend: the more accumulated per-account
history before measuring, the better the model performs — consistent with
`huella_categoria_cuenta` (the highest-importance feature, 0.945) needing
real account history to be informative; with little history, more rows
fall back to the population default (generic, less discriminative). This
doesn't close 100% of the gap against the doc's 0.966 (probably the
exploratory script used even more warm-up, or some other minor
methodological difference not identified), but it explains most of it in
a verifiable way, with monotonic evidence, not speculation.

**This isn't a bug — it's a real operational prerequisite of Domain 2,
which needs to be communicated if the approach is actually used:**

- **New account / little history:** AUC-PR ≈ 87.7%-94.7% (mean 91.4%),
  precision 85-95%, recall 82-89% (range measured in the
  `frac_train_inicial=0.5` split's detail).
- **Established account / broad history:** AUC-PR ≈ 91.9%-99.1% (mean
  94.0%).

Per-account personalization improves over time as an account accumulates
transactions — the same "cold start" problem any personalization system
has (recommenders, credit scoring). A new account in real production would
start in the weaker scenario.

**Current number to report going forward:** 0.898-0.914 AUC-PR (depending
on the history scenario), not 0.966 — that Step 4 number is now marked as
not reproduced with production code, and its exact origin unconfirmed (the
exploratory script is no longer available).

## 15. Two minor refinements explored (2026-09-14)

Closes the last 2 items of "Explicitly pending." Pure exploration — none
of this was brought into `src/`. Same comparison methodology as section 14
(5-fold walk-forward, `frac_train_inicial=0.5`,
`HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
max_iter=200, class_weight="balanced", random_state=42)`, isotonic fit on
VAL, AUC-PR on calibrated scores), against the same 5-feature baseline:
**0.914 ± 0.030** (folds: 0.877, 0.879, 0.931, 0.935, 0.947).

### 15.1 Point-in-time encoding of `category` — positive result

A 6th feature was added, `frecuencia_categoria_expandida`: for each row,
`(conteo_categoria_hasta_ahora + 1) / (conteo_total_hasta_ahora +
n_categorias)` — causal (only counts strictly earlier rows) and expanding
(recomputed row by row over the full history, with Laplace smoothing
`alpha=1` to avoid extreme values on a category's earliest occurrences).

**Explicitly verified that it does NOT reproduce section 2's non-stationarity
trap** (a fixed one-hot + temporal split breaking the model via
`personal_care`): this feature's real trajectory for that category, by
year —

| Year | Transactions | Feature's mean value |
|---|---|---|
| 2013-2023 (11 years) | 1-4 per year | 0.000023 – 0.000090 |
| 2025 | 63,876 | 0.029253 |
| 2026 | 18,580 | 0.064972 |

— shows the value **adapts smoothly** to the real volume, without staying
stuck on the ~24 early rows: unlike a static one-hot fit once, here the
weight of that old evidence dilutes itself (by construction) as the total
denominator grows. There's no fixed coefficient learning a spurious rule
from a handful of rows.

**Result (6 features vs. 5-feature baseline):**

| | AUC-PR per fold | Mean | Std dev |
|---|---|---|---|
| Baseline (5 features) | 0.877, 0.879, 0.931, 0.935, 0.947 | 0.914 | ±0.030 |
| **+ `frecuencia_categoria_expandida` (6 features)** | 0.907, 0.923, 0.976, 0.958, 0.970 | **0.947** | ±0.027 |

**A real improvement of +0.033 AUC-PR, consistent across the 5 folds**
(improved in 4 of 5, fold 0 is practically unchanged). Not noise — the
magnitude and the consistency across folds exceed the standard deviation.

**Recommendation:** worth bringing this feature to production
(`src/calibrador_arboles.py` + `src/ejecutor_arboles.py`). **Implemented
2026-09-14, see section 16** — real production AUC-PR confirmed the
improvement (0.947±0.027 in 5-fold, reproducing this finding almost
exactly).

### 15.2 Fingerprint with EWMA decay instead of a fixed K=20 window — negative/inconclusive result

`huella_categoria_cuenta` (a causal window over the account's last 20
transactions) was replaced with a per-account category histogram with
exponential decay: on every transaction, all existing weights decay by
`lambda`, `(1-lambda)` is added to the current category, and the
fingerprint is the current category's normalized weight (same population
fallback as today for accounts with <5 real transactions, unweighted).

**Result (replacing the fixed-window fingerprint, not adding to it):**

| Variant | AUC-PR per fold | Mean | Std dev |
|---|---|---|---|
| Baseline (fixed K=20 window) | 0.877, 0.879, 0.931, 0.935, 0.947 | 0.914 | ±0.030 |
| EWMA λ=0.98 (same λ as the rest of the project) | 0.857, 0.852, 0.931, 0.904, 0.924 | **0.894** | ±0.033 |
| EWMA λ=0.9 (faster decay) | 0.866, 0.901, 0.950, 0.941, 0.958 | **0.923** | ±0.035 |

**λ=0.98 clearly gets worse** (-0.020, worse in 4 of 5 folds). **λ=0.9
improves slightly** (+0.009), but the standard deviation (±0.033-0.035)
exceeds the difference — no solid evidence that it's a real improvement
and not sampling noise.

**Honest verdict: no clear case for replacing the fixed 20-transaction
window.** The fixed window (which "forgets" entirely after 20 transactions)
seems to capture "typical recent behavior" better than a smooth decay that
never fully forgets — possibly because per-account behavior in Sparkov
doesn't have a gradual drift that decay can exploit, similar in spirit to
Path 3's finding (EWMA smoothing didn't help there either, for the same
reason: not every signal benefits from long memory).

**Recommendation:** don't bring this change to production — the current
version (fixed K=20 window) stays as the best validated option.

## 16. `frecuencia_categoria_expandida` brought to production (2026-09-14)

Closes section 15.1's recommendation: the 6th feature (causal point-in-time
encoding of `category`, Laplace `alpha=1`) is now in `src/`, not just in
exploration.

**Modified files:**
- `src/features_recursivas_cuenta.py` — a new batch function
  `calcular_frecuencia_categoria_expandida_batch(df, n_categorias)`
  (vectorized: `groupby("category").cumcount()` for the causal per-category
  count + `np.arange` for the total, no loop) and a new incremental-state
  class `EstadoFrecuenciaCategoriaGlobal` (global, not per account — it
  lives in this module for cohesion with
  `calcular_frecuencia_poblacional_categoria`, not because it's
  "per account"). `n_categorias` is derived from
  `len(frecuencia_poblacional_categoria)` (TRAIN's vocabulary) instead of
  adding a new field to the artifact.
- `src/calibrador_arboles.py` — `FEATURES_ARBOLES` goes from 5 to 6
  entries; `construir_features()` adds the new feature.
- `src/ejecutor_arboles.py` — `EjecutorArboles` maintains
  `EstadoFrecuenciaCategoriaGlobal` alongside the other two recursive
  states, feeding the table's 6th dimension.
- `src/artefacto_arboles.py` — **no schema change**: `features` was
  already a generic list, and `n_categorias` is reused from
  `frecuencia_poblacional_categoria` instead of a new field.
- New tests: 3 in `tests/test_features_recursivas_cuenta.py` (a hand-worked
  case, isolated incremental state, batch/incremental parity over a random
  sequence) + 1 in `tests/test_ejecutor_arboles.py` (confirms the new
  feature genuinely participates in the lookup table, with a synthetic
  6-feature artifact).

**Real reproduced figures** (production code, not the exploratory script):

| Methodology | AUC-PR | Compare against (5 features) |
|---|---|---|
| 5-fold walk-forward (`frac_train_inicial=0.5`) | **0.947 ± 0.027** | 0.914 ± 0.030 |
| Single 60/20/20 split, VAL | 0.932 | 0.898 |
| Single 60/20/20 split, TEST | 0.925 | 0.901 |

The 5-fold almost exactly reproduces what section 15.1's exploration had
measured (0.947±0.027) — confirms the finding wasn't an artifact of the
exploratory script. The single split also improves consistently with the
6th feature (not just the 5-fold), though it still reflects the
history/cold-start dependency already documented in section 14
(0.925-0.932 vs. the 5-fold's 0.947, same pattern as before).

**Lookup table:** goes from `[43, 14, 2, 28, 16]` (539,392 combinations,
5 features) to `[42, 11, 2, 26, 16, 29]` (**11,147,136 combinations**, 6
features) — grows ~20x from the new dimension (29 unique thresholds the
trees learned for `frecuencia_categoria_expandida`). Still exact by
construction (same `bisect_left` mechanism), verified with the real
bit-exact test: maximum difference < 1×10⁻⁹ over the full 234,189 rows of
TEST.

**Real measured latency:** **10.90 µs/decision** (1,170,945 decisions,
`time.perf_counter()`) — up from the 5-feature 9.84 µs (one more state
read + one more table dimension), still the same order of magnitude as
Domain 1 and far below industry budget.

**Tests: 186 total, all green** (182 prior + 4 new; none of the 182
existing tests were modified to make this work — the generality of
`FEATURES_ARBOLES` as a list, and of `features`/`umbrales_por_feature` in
the artifact, absorbed the change from 5 to 6 dimensions without breaking
anything).

**Domain 1 was not touched.** No commit or deploy was made — left to the
user's decision. SYNAPSE is still in testing phase.

**Update (2026-09-14, after section 21):** `LEARNING_RATE` changed from
`0.05` to `0.1` in `src/calibrador_arboles.py`, implementing section 21's
recommendation. Retrained and re-verified bit for bit (19 Domain 2 tests +
186 total, all green). Updated real figures from the single 60/20/20
split:

| | VAL (before / after) | TEST (before / after) |
|---|---|---|
| Precision | 88.7% / **94.3%** | 87.8% / **90.3%** |
| Recall | 81.8% / 85.7% | 83.0% / 84.0% |
| F1 | 0.851 / 0.898 | 0.853 / 0.871 |
| AUC-PR | 0.898 / **0.951** | 0.901 / **0.926** |

Consistent improvement across both metrics and both splits, not just in
section 21's 5-fold. Lookup table: 11,147,136 → **16,189,440**
combinations (`[39, 11, 2, 31, 16, 30]` thresholds per feature, larger due
to the new thresholds the trees learn with `learning_rate=0.1`) — still
exact by construction, re-verified bit for bit. This update's first run
was done directly (not via a subagent) due to a weekly API quota limit
reached during the session.

## 17. Ablation: is there redundancy between the 2 category features? (2026-09-14)

A direct question after bringing `frecuencia_categoria_expandida` (global)
to production alongside `huella_categoria_cuenta` (per account, already
existing): do they overlap with each other, or do they contribute
genuinely distinct signal? A 4-set ablation, same comparison methodology
as sections 14-16 (5-fold walk-forward, `frac_train_inicial=0.5`, the
Step 4 winning configuration, isotonic fit on VAL, AUC-PR on calibrated
scores):

| Set | AUC-PR per fold | Mean | Std dev |
|---|---|---|---|
| **6 features (production baseline)** | 0.907, 0.923, 0.976, 0.958, 0.970 | **0.947** | ±0.027 |
| Without `huella_categoria_cuenta` (5, only the global one) | 0.438, 0.332, 0.395, 0.454, 0.606 | **0.445** | ±0.091 |
| Without `frecuencia_categoria_expandida` (5, the earlier version) | 0.877, 0.879, 0.931, 0.935, 0.947 | **0.914** | ±0.030 |
| Only `amt` + the 2 category features (3) | 0.833, 0.857, 0.838, 0.827, 0.895 | **0.850** | ±0.025 |

**Honest verdict: NO redundancy — on the contrary, the two category
features are asymmetrically important, and both contribute real signal.**

- **Removing `huella_categoria_cuenta` is catastrophic**: the model
  collapses from 0.947 to **0.445** (more than half) and the spread
  across folds triples (±0.091) — by far the feature carrying the most
  weight in the model, consistent with the permutation importance already
  measured in section 7 (0.945, the highest of all).
- **Removing `frecuencia_categoria_expandida` costs much less**: drops
  from 0.947 to 0.914 (-0.033) — real, but not comparable in magnitude to
  removing the fingerprint. It's a genuine complement, not decorative, but
  it isn't the model's backbone.
- **The other 3 features (`hora`, `conteo_ventana_global`,
  `monto_ewma_cuenta`) also contribute for real**: with only `amt` + the 2
  category features (3 features), the model drops to 0.850 — 0.097 below
  the full set. Not dispensable despite modest individual importance
  (same pattern already seen in section 7's ablation: "Set C" with only 2
  features dropped similarly).

**Conclusion:** the 6 production features stay as they are — each
contributes real, verified signal, none is redundant enough with another
to justify dropping it. `huella_categoria_cuenta` is, by a wide margin,
Domain 2's single most critical feature.

## 18. Demographic columns never tested — unavailable, not a real negative result

Pending item #2 from the additional-tests list: test customer
age/gender/occupation/city as new features, typical candidates from the
`Sparkov_Data_Generation` generator that were never used in the model.

**Real finding before anything could be tested:** `data/raw/sparkov_2013_2026.csv`
**doesn't have those columns**. Its 11 real columns are: `cc_num`, `lat`,
`long`, `trans_date`, `trans_time`, `unix_time`, `category`, `amt`,
`is_fraud`, `merch_lat`, `merch_long` — verified by reading the CSV
directly, not assumed. Section 1 of this doc already explained it: the
dataset was "combined and reduced to the useful columns (61 raw files → 1
CSV)" — gender, date of birth, occupation, and city/state were dropped in
that combination step, never made it into the CSV the project uses.

**A search was made for any original generation file that still had that
data** (`/tmp/sparkov_gen`, the generator install used at the time):
`datagen_customer.py`, `profiles/` (the generator's generic demographic
profiles, not the 99 real accounts'), and `demographics.csv` (a U.S.
population reference table, also not specific to the accounts) exist, but
**there's no `customers.csv` from the real run that generated the 99
accounts** in `sparkov_2013_2026.csv` — only the already-combined, already-
reduced transactions CSV was kept. Recovering that data would require
**regenerating the entire dataset from scratch** with
Sparkov_Data_Generation, the same obstacle already documented in section 10
(a test with just 2 customers × 10 days took >13 minutes without finishing
on this machine — regenerating 100 customers × 13 years isn't feasible
here).

**Honest verdict: this test is closed for lack of data, not because of a
negative model result.** This isn't "tried and didn't help" (like section
15.2) — it's "couldn't be tried because the data no longer exists in the
dataset we have." If Sparkov is ever regenerated from scratch while also
saving the customer file, this test is genuinely still pending for later.
With the current data, the transactions dataset is exhausted in terms of
explorable columns — the 6 production features are everything
`sparkov_2013_2026.csv` has to offer as it exists today.

## 19. Drift detector (`src/deriva.py`) vs. `frecuencia_categoria_expandida` — fires too often, a real recommendation

`src/deriva.py` (PSI + Kolmogorov-Smirnov, Domain 1) is generic —
`evaluar_deriva()` runs on any numeric column of a DataFrame, with no real
dependency on Domain 1. It was applied as-is to Domain 2's 6 production
features. `ponderar_deriva_por_coeficiente()` **is** specific to Domain 1
(it assumes an artifact with `coeficientes`/a linear model) — it doesn't
apply to Domain 2's tree artifact without adapting it; **a separate
finding, outside this test's scope**: there is no automatic recalibration
trigger yet for Domain 2 (`disparador_recalibracion.py` is exclusive to
Domain 1).

**Positive control — the detector DOES see the real, expected change:**
comparing 2013-2018 (reference) against 2024-2026 (current),
`frecuencia_categoria_expandida` gives **PSI=4.84**
(`deriva_significativa_recalibrar`, far above the 0.25 threshold) —
consistent with the real change already documented in section 15.1
(`personal_care` going from ~0.00002 to ~0.06). The detector works: it
isn't blind to real change.

**The real problem — it fires almost always, not just when it matters.**
Comparing each year against the next (13 valid transitions,
2013→2014 … 2025→2026):

| Feature | Triggers recalibration (PSI>0.25) |
|---|---|
| `amt` | 0/13 (0%) |
| `hora` | 3/13 (23%) |
| `conteo_ventana_global` | 0/13 (0%) |
| `monto_ewma_cuenta` | 1/13 (8%) |
| `huella_categoria_cuenta` | 0/13 (0%) |
| **`frecuencia_categoria_expandida`** | **11/13 (85%)** |

**Aggregate recalibration recommendation** (with the full `evaluar_deriva()`,
"recalibrate if ANY feature fires"), same 13 year-over-year transitions:

| Feature set | Recommends recalibrating |
|---|---|
| 5 features (without the new one) | 3/13 (23%) |
| **6 features (with the new one)** | **12/13 (92%)** |

Adding `frecuencia_categoria_expandida` to drift monitoring makes the
system recommend recalibration in **practically every** year transition —
not because the model is deteriorating, but because this feature, by
construction (expanding frequency, Laplace), **always** moves over time —
that's its normal, expected behavior, documented since it was introduced
(section 15.1), not a real anomaly every time it happens.

**Honest verdict: `deriva.py`'s current behavior with this feature is
technically correct (it measures the real movement well) but
operationally noisy (it would recalibrate almost always, without
distinguishing "change expected by design" from "real deterioration that
warrants attention").** A real recommendation, not implemented in this
pass: **exclude `frecuencia_categoria_expandida` from the set of columns
passed to `evaluar_deriva()`/a future Domain 2 recalibration trigger**
(monitor the other 5 normally) — or, a more elaborate alternative, monitor
its *rate of change* relative to an expected trend instead of its raw
value, but that's additional, non-trivial engineering, not a one-line
tweak. Since there's no recalibration trigger yet for Domain 2 (see
above), this is noted for whenever one gets built, not as an urgent bug to
fix today.

## 20. Adversarial test — simple evasion of `huella_categoria_cuenta` (2026-09-14)

All of Sparkov's simulated fraud so far has been "naive": the generator
injects fraud without bothering to imitate the account's normal behavior.
The real question: how easy would it be for an attacker who DID know the
`huella_categoria_cuenta` mechanism (the highest-importance feature,
section 17) to evade detection?

**Simulated attack (the simplest and cheapest possible):** for each of the
200 real fraudulent transactions in TEST, replace only its `category` with
that account's most frequent category within its causal window of the last
20 transactions (the exact same criterion `huella_categoria_cuenta` uses) —
without touching amount, timing, or account. Under this substitution,
`huella_categoria_cuenta` recomputes to **exactly** the maximum value
reachable for that account at that moment, and
`frecuencia_categoria_expandida` also recomputes exactly (counting real
prior occurrences of the chosen category across the entire global history
before that row). `amt`, `hora`, `conteo_ventana_global`, and
`monto_ewma_cuenta` stay the same — the attacker only controls the
category, not the amount or timing (a more aggressive attacker could also
disguise the amount; not tested in this pass). **Documented limitation:**
the change isn't propagated forward (later transactions on the same
account that would see this disguised category within their own window) —
a second-order effect, ignored as out of scope.

**Real result, on the production model (6 features) and the same TEST
split (234,189 rows, 200 real frauds):**

| | Without disguise | With disguise |
|---|---|---|
| Frauds detected | 164/200 | **59/200** |
| Recall | 82.0% | **29.5%** |
| Average score | 0.816 | 0.322 |
| Average `huella_categoria_cuenta` | 0.091 | 0.738 |
| Average `frecuencia_categoria_expandida` | 0.0844 | 0.0579 |

**Recall drop: 52.5 percentage points — 105 of 200 frauds go undetected**
with the simplest, cheapest possible attack (one single decision: which
category to use). `huella_categoria_cuenta` goes from an average of 0.09
(clearly anomalous) to 0.74 (looks like normal behavior) — the evasion
mechanism works exactly as expected. Notably,
`frecuencia_categoria_expandida` (the global feature) **barely moves**
(0.084→0.058) — confirming what section 17's ablation already showed: it's
much harder to manipulate by picking a single category, because it
depends on the full population, not on one account's history.

**Honest conclusion, not a bug to fix but a design limitation to
communicate:** any system that personalizes detection by learning "what's
normal for this account" is, by nature, evadable by an adversary who knows
that normal pattern and deliberately imitates it — not a defect unique to
SYNAPSE, but a tension inherent to behavior-based personalization (the
same problem recommendation systems and profile-based bot detection have).
`huella_categoria_cuenta`'s real advantage is against "naive" fraud (most
real fraud, which isn't specifically optimized against this mechanism) —
not a promise of robustness against a sophisticated, informed attacker. If
used in a context where the adversary could plausibly learn the mechanism
(e.g. an insider, or after a leak of how the model works), this limitation
needs to be communicated explicitly, not assumed away. A possible future
mitigation, not implemented: a feature measuring "how perfect/typical the
category choice is" itself (a choice that's *too* optimal could,
paradoxically, be a secondary signal of manipulation) — an idea, not a
proven solution.

## 21. Hyperparameter search redone with the 6 features (2026-09-14)

Closes the last of this round's 5 tests. The tuning's original "Step 4"
(a 12-combination grid, `max_depth` ∈ {3,5,8}, `learning_rate` ∈
{0.05,0.1}, `max_iter` ∈ {200,300}, over 3 folds) had been run with the 5
features that existed back then, before adding
`frecuencia_categoria_expandida` (section 16). It was repeated with the
current 6 production features.

**Method note:** the first attempt crashed from real memory pressure — the
machine has only 5.9GB of total RAM, and `calibrar()` also builds the full
lookup table (unnecessary just to compare AUC-PR; with a high `max_depth`
it can grow to tens of millions of combinations). It was redone by training
and measuring AUC-PR directly, without building the table, with explicit
memory cleanup between combinations.

**12-combination grid (3-fold, same methodology as the original Step 4):**

| Configuration | AUC-PR (3-fold) |
|---|---|
| **`max_depth=3, lr=0.1, max_iter=200`** | **0.9479 ± 0.0223** |
| `max_depth=3, lr=0.05, max_iter=300` | 0.9459 ± 0.0263 |
| `max_depth=3, lr=0.1, max_iter=300` | 0.9464 ± 0.0207 |
| `max_depth=3, lr=0.05, max_iter=200` (current production config) | 0.9430 ± 0.0234 |
| `max_depth=5` (all 4 combinations) | 0.9169 – 0.9204 |
| `max_depth=8` (all 4 combinations) | 0.8491 – 0.8768, much more unstable |

`max_depth=8` confirms the same pattern already seen with 5 features (the
original Step 4): it overfits and is unstable with only 99 accounts
(std dev up to ±0.085). `max_depth=3` remains, by a wide margin, the best
family.

**Validating the winning candidate (`max_depth=3, lr=0.1, max_iter=200`)
with the full 5-fold** (`frac_train_inicial=0.5`, same standard as
sections 14-20):

| | AUC-PR per fold | Mean | Std dev |
|---|---|---|---|
| Current config (`lr=0.05`) | 0.907, 0.923, 0.976, 0.958, 0.970 | 0.947 | ±0.027 |
| **Candidate (`lr=0.1`)** | 0.914, 0.957, 0.976, 0.955, 0.983 | **0.957** | ±0.024 |

**A real improvement of +0.0098**, better or practically tied in 4 of 5
folds (never clearly worse), with a slightly lower std dev. Not a dramatic
improvement like `frecuencia_categoria_expandida`'s (+0.033, section 16),
but consistent and doesn't look like noise.

**Recommendation:** worth changing `LEARNING_RATE` from `0.05` to `0.1` in
`src/calibrador_arboles.py` — this means retraining, rebuilding the lookup
table (with the new model, the trees' real thresholds change), and
re-validating bit-for-bit accuracy, the same discipline as any production
change in this project.

**Implemented the same day**, see the update at the end of section 16
(real figures with `learning_rate=0.1`, table rebuilt and re-verified bit
for bit, 186 tests green).

## Close-out of the additional 5-test round (2026-09-14)

Summary of the 5 tests the user asked to run one at a time, after closing
`frecuencia_categoria_expandida`'s integration (section 16):

1. **Redundancy ablation** (section 17): no redundancy, the current 6
   features stay as they are.
2. **Sparkov demographic features** (section 18): closed for lack of data
   (never saved when the 61 raw files were combined), not a negative
   model result.
3. **Drift detector with the new feature** (section 19): technically
   works fine, but is noisy with `frecuencia_categoria_expandida`
   (triggers recalibration 85% of the time vs. 23% without it) —
   recommendation to exclude it from monitoring, not implemented. Along
   the way, found that Domain 2 doesn't have its own recalibration
   trigger yet (only Domain 1's exists).
4. **Adversarial test** (section 20): a cheap attack (imitating the
   account's typical category) drops recall from 82.0% to 29.5% —
   a design limitation inherent to behavior-based personalization,
   documented, not "fixed."
5. **Hyperparameter search with 6 features** (this section): a real
   improvement (+0.0098 AUC-PR in 5-fold, also confirmed in the single
   split: AUC-PR VAL 0.898→0.951, TEST 0.901→0.926) by changing
   `learning_rate` from 0.05 to 0.1 — **implemented the same day**, see
   the update at the end of section 16.

**Domain 2's general status after this round:** the production model
changed once relative to what section 16 closed with — `learning_rate=0.1`
instead of `0.05` (finding #5, implemented and re-verified bit for bit).
The other 4 tests were pure diagnostics, without touching `src/`. The two
recommendations that remained open are now closed (see below): #3
(exclude `frecuencia_categoria_expandida` from drift monitoring)
implemented via `FEATURES_MONITOREO_DERIVA`; #4 (adversarial mitigation)
tried and dropped, see section 22.

## 22. Adversarial mitigation tried and dropped (2026-09-14)

Closes section 20's speculative recommendation ("a feature measuring how
perfect/typical the category choice is"). Concrete design tested:
`brecha_categoria = huella_categoria_cuenta -
frecuencia_categoria_expandida` as a 7th feature — under section 20's
attack, `huella` jumps to ~0.74 but `frecuencia_expandida` barely moves
(~0.06), a gap of ~0.68 vs. ~0.007 in normal, non-disguised fraud.

**Real result (walk-forward 5-fold for overall AUC-PR, single split for
the attack, same methodology as section 20):**

| | 6 features (production) | 7 features (+ `brecha_categoria`) |
|---|---|---|
| Overall AUC-PR (5-fold) | 0.9568 ± 0.0240 | **0.9479 ± 0.0322** |
| Recall without disguise | 82.0% (164/200) | 83.0% (166/200) |
| Recall with disguise | 29.5% (59/200) | **33.5% (67/200)** |

**Honest verdict: not worth it.** It only recovers 4 points of recall
under attack (29.5%→33.5%, far from the normal 82-83%), at the cost of
worsening overall AUC-PR (-0.0089) and increasing variance across folds
(±0.024→±0.032). The cost in overall performance outweighs the marginal
benefit against this specific attack — the same pattern as other
improvement attempts dropped in this project (section 15.2's EWMA, Path
3's smoothing): a reasonable idea in theory that doesn't hold up with real
data. **Not implemented in `src/`.**

With this, all 5 recommendations from the test round are resolved (3
implemented, 2 tried and honestly dropped). No finding remains pending a
decision in Domain 2 as of this session's close.
