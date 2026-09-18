# SYNAPSE

## Two-Speed Decision Framework for Real-Time AI Governance

### The Problem

LLM-based agents reason well but decide slowly — seconds per call, sometimes
more. Production systems that gate real transactions (fraud, industrial
control, multi-agent tool calls) need a decision on every event, often at
sub-millisecond latency. Calling a model in that hot path either blows the
latency budget or forces you to skip the check.

Most AI-adjacent systems also lack a hard backstop. When the model's
confidence is wrong — a bad calibration, a distribution shift, a silent
bug — nothing overrides it, and nothing outside the model's own scoring says
"stop" with a guarantee that doesn't depend on the model being right.

Multi-agent systems compound both problems: decisions cascade through
several agents, and when the outcome is wrong there is no accountability
chain — no versioned record of which policy produced which decision, and
why.

### The Approach: Two-Speed Architecture

SYNAPSE separates **learning** (slow, offline, no time pressure) from
**deciding** (fast, online, microseconds). The two never share a time
budget:

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
- **Policy Artifact** — the only channel between the two speeds: a
  versioned JSON contract, never code.
- **Bridge** — swaps the artifact atomically so the Executor never reads a
  partial write.
- **Executor** — pure arithmetic against the current artifact. No model
  inference, no network call, no retraining, on the hot path.
- **Veto Layer** — hard, model-independent invariants that override the
  Executor's decision when they fire, regardless of model confidence.
- **Drift Detection** — PSI + Kolmogorov-Smirnov on live features, decides
  *when* the Calibrator needs to run again.
- **Decision Log** — every decision, with the artifact version that
  produced it, kept for audit.

### Validated Domains

| Domain | Dataset | Metric | Latency |
|---|---|---|---|
| Global fraud detection | ULB Credit Card Fraud (284,807 transactions, 492 frauds, public) | F1 0.78, AUC-ROC 0.978 (5-fold walk-forward avg, ±0.007) | 8.33 µs/decision |
| Per-account fraud detection | Sparkov synthetic (1,170,945 transactions, 99 accounts) | F1 0.85, AUC-PR 0.90 (test) | 9.84 µs/decision |
| Industrial anomaly detection (unsupervised) | SKAB (`valve1` subset, 16/35 files) | F1 0.76 — 3rd of 9 published leaderboard methods, 2 F1 points off the best (a neural net) | 13.05 µs/decision |

Methodology notes, because the numbers are only useful if the comparison is
honest:
- The SKAB result uses the **unsupervised** detector (z-score/3-sigma,
  no fraud labels seen in training) to stay comparable to SKAB's published
  leaderboard, which only evaluates unsupervised methods on the full
  35-file dataset. A separate supervised variant scores higher (F1 0.88)
  but isn't leaderboard-comparable — trained on labels the published
  benchmark doesn't allow.
- Latency is measured end-to-end (`time.perf_counter()`, full
  `Executor.decidir()` call, including per-event state maintenance), not
  isolated arithmetic — the honest, reproducible number, not the best-case
  one.
- Each row is reproducible from this repo — see Quick Start.

### Quick Start

```bash
git clone https://github.com/stiven0210/synapse.git
cd synapse
pip install -r requirements.txt

# Example dataset (ULB Credit Card Fraud — public, Kaggle):
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# place it at data/raw/creditcard.csv

pytest tests/ -q                            # 187 tests
python scripts/validacion_end_to_end.py     # real metrics + latency, Domain 1
python scripts/deteccion_deriva.py          # real drift report
```

### Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full C4-style
breakdown — system context, containers, and each component's
responsibilities and non-responsibilities — or explore the [interactive
diagram](docs/architecture_diagram.html) directly.

### Applications

- Financial fraud detection (global and per-account)
- Industrial anomaly monitoring
- Multi-agent AI governance — a hard, auditable veto layer in front of
  agent-driven actions
- Any domain that needs a sub-millisecond, auditable decision in front of
  a slower model

### Why Two Speeds?

Coupling learning and deciding forces a bad tradeoff: either the model runs
on the hot path (and the latency budget breaks), or the hot path runs
without the model's judgment (and there's no calibration at all). Splitting
them lets each side be honest about its own constraints — the Calibrator
can take as long as it needs and be as complex as it needs, because it
never touches the request path; the Executor can be trivially fast and
auditable, because it never does more than evaluate an already-calibrated
artifact against hard invariants.

The Veto Layer exists because a fast, well-calibrated Executor is still a
model — it can be wrong. The invariants it enforces don't depend on the
model being right; they're checked independently, every time.

### Documentation

- [Architecture (C4)](docs/architecture.md)
- [Architecture Decision Records](docs/adr/README.md)
- Domain research notes (detailed, Spanish): [Domain 2 — per-account
  personalization](docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md),
  [Domain 3 — industrial IoT
  generalization](docs/CAMINO3_GENERALIZACION_IOT.md)

### License

[MIT](LICENSE)

### Author

Genes Stivens Benavides Angel — Solutions Architect.
Personal research project. Not affiliated with any employer.
