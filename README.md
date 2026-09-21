# SYNAPSE

[![Tests](https://github.com/stiven0210/synapse/actions/workflows/test.yml/badge.svg)](https://github.com/stiven0210/synapse/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Two-Speed Decision Framework for Real-Time Systems

### The Problem

Production systems that need to act on every event — approving a
transaction, flagging an anomaly, triggering an alert — face a hard
tradeoff: the model that *should* make the decision is too slow for the
latency budget, and the fast path that *can* meet the budget has no
calibrated judgment behind it.

Most real-time decision systems also lack a hard backstop. When the
model's confidence is wrong — a bad calibration, a distribution shift, a
silent bug — nothing overrides it, and nothing outside the model's own
scoring says "stop" with a guarantee that doesn't depend on the model
being right.

SYNAPSE solves both problems by separating **learning** from **deciding**
and adding a **model-independent veto layer** that always overrides.

### The Approach: Two-Speed Architecture

The slow layer learns. The fast layer decides. They never share a time
budget.

```
  historical data
        │
        ▼
 ┌────────────────────┐
 │     CALIBRATOR      │   slow layer — learns offline,
 │  (offline, no time  │   no time constraint
 │      limit)          │
 └──────────┬──────────┘
            │ produces
            ▼
 ┌────────────────────┐
 │   POLICY ARTIFACT    │   versioned JSON contract
 │   (versioned JSON)    │   (weights, thresholds, version N)
 └──────────┬──────────┘
            │ atomic swap (tmp file + replace —
            │ Executor never reads a half-written artifact)
            ▼
 ┌────────────────────┐
 │       BRIDGE          │
 └──────────┬──────────┘
            │
 live event │
      ──────┼───────────►┌────────────────────┐
            │             │      EXECUTOR         │  fast layer —
            │             │  (pure arithmetic,     │  microseconds,
            │             │   µs-level latency)    │  no model inference,
            │             └──────────┬───────────┘  no network, no retrain
            │                        ▼
            │             ┌────────────────────┐
            │             │      VETO LAYER        │  hard invariants,
            │             │  (model-independent,   │  always overrides —
            │             │   always overrides)    │  correct even if the
            │             └──────────┬───────────┘  model is miscalibrated
            │                        ▼
            │                  final decision
            │                        │
            │             ┌──────────▼───────────┐
            │             │     DECISION LOG        │  full audit trail
            │             └────────────────────────┘
            │
            └── observed by ──► DRIFT DETECTION (PSI + KS)
                                 triggers recalibration back
                                 into the Calibrator
```

- **Calibrator** — learns from history, offline, no latency constraint.
  Can use any algorithm (logistic regression, gradient boosting, neural
  nets). Its only output is a Policy Artifact.
- **Policy Artifact** — the only channel between the two speeds: a
  versioned JSON contract with features in contractual order,
  coefficients, decision threshold, and validation metrics. Never code.
- **Bridge** — swaps the artifact atomically so the Executor never reads a
  partial write.
- **Executor** — pure arithmetic against the current artifact + compact
  recursive state (EWMA, windowed counts). No model inference, no network
  call, no retraining on the hot path.
- **Veto Layer** — hard, model-independent invariants that override the
  Executor's decision when they fire. Corrupt artifact → veto. Extreme
  value → veto. Invalid score (NaN, out of [0,1]) → veto. These hold
  regardless of model quality.
- **Drift Detection** — PSI (Population Stability Index) +
  Kolmogorov-Smirnov on live features. Decides *when* the Calibrator
  needs to run again — not on a schedule, but when the data says so.
- **Decision Log** — every decision recorded with the artifact version
  that produced it, the veto status, and the full input — kept for audit
  and reproducibility.

### Validated Domains

| Domain | Dataset | Key Metric | Latency |
|---|---|---|---|
| Global fraud detection | ULB Credit Card Fraud (284,807 tx, public) | F1 0.78, AUC-ROC 0.978 | 8.33 µs/decision |
| Per-account fraud detection | Sparkov synthetic (1.17M tx, 99 accounts) | F1 0.85, AUC-PR 0.90 (test) | 9.84 µs/decision |
| Industrial anomaly detection | SKAB (34 anomaly files, official split, 8 sensors) | F1 0.76 — 3rd of 9 on official leaderboard | 13.05 µs/decision |

**Methodology notes** (because the numbers are only useful if the
comparison is honest):

- The SKAB result uses an **unsupervised** detector (z-score, 3σ
  threshold, no anomaly labels seen in training) evaluated with the
  **official SKAB methodology** (`core/metrics.py::chp_score`,
  `metric="binary"`) on all 34 anomaly files — directly comparable to the
  published leaderboard. A separate supervised variant scores higher
  (F1 0.88) but isn't leaderboard-comparable.
- Latency is measured end-to-end (`time.perf_counter()`, full
  `Executor.decidir()` call including per-event state maintenance) — the
  honest, reproducible number, not isolated arithmetic.
- Domain 1 is reproducible end-to-end with the Quick Start commands
  below. Domain 2's code, tests, and methodology are in the repo — the
  dataset is generated locally with Sparkov (see its research notes for
  exact parameters). Domain 3 is an exploratory validation — see its
  research notes for methodology and results.

### Quick Start

```bash
git clone https://github.com/stiven0210/synapse.git
cd synapse
pip install -r requirements.txt

python -m scripts.demo                      # synthetic data, no dataset needed, ~5 seconds

pytest tests/ -q                            # 187 tests

# Full validation with a real dataset (ULB Credit Card Fraud — public, Kaggle):
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# place it at data/raw/creditcard.csv

python -m scripts.validacion_end_to_end     # real metrics + latency, Domain 1
python -m scripts.deteccion_deriva          # real drift report
```

### Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full C4-style
breakdown (system context, containers, component responsibilities and
non-responsibilities) or explore the [interactive
diagram](docs/architecture_diagram.html).

### Design Principles

- **Fail loud, never silent.** A decision the system can't make is
  escalated or rejected — never defaulted to something that looks like a
  real answer.
- **Separation of learning from deciding.** The Calibrator and Executor
  never share a time budget. Each is honest about its own constraints.
- **Hard invariants over model confidence.** The Veto Layer doesn't trust
  the model. It checks independently, every time.
- **Versioned policy contracts.** The artifact is the only interface
  between slow and fast. It's JSON, human-readable, and auditable.
- **Full auditability.** Every decision is logged with the artifact
  version, veto status, and inputs that produced it.
- **Speed is measured, never assumed.** Latency numbers come from
  `time.perf_counter()`, not estimates.

### Applications

**Validated:**
- Financial fraud detection (global patterns and per-account behavior)
- Industrial anomaly monitoring (IoT sensor data)

**Applicable (not yet validated):**
- Telecommunications fraud (SIM swap, roaming fraud)
- Cybersecurity intrusion detection
- Real-time credit scoring
- Any domain requiring sub-millisecond auditable decisions with hard
  guardrails

### Documentation

- [Architecture (C4)](docs/architecture.md)
- [Architecture Decision Records](docs/adr/README.md)
- Domain research notes (detailed, in Spanish):
  [Domain 2 — per-account personalization](docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md),
  [Domain 3 — industrial IoT generalization](docs/CAMINO3_GENERALIZACION_IOT.md)

### License

[MIT](LICENSE)

### Author

Genes Stivens Benavides Angel — Solutions Architect.
Personal research project.