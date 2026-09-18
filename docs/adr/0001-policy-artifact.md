# ADR 0001 — Shape of the Policy Artifact

**Status:** Decided.

**Context:** The Calibrator (slow layer) needs to hand the Executor (fast
layer) the parameters it learned, without the Executor ever having to load
the training model or any heavy ML library. The artifact is the only point
of contact between the two layers — its shape has to be stable, versioned,
and self-describing (the Executor must never have to guess the order the
coefficients come in).

## Decision

The Policy Artifact is a JSON document with this exact shape. Field names
are Spanish, matching the rest of the codebase (`src/`, tests) — this is
the literal contract the code reads and writes, not a translated
illustration:

```json
{
  "version": 1,
  "fecha_calibracion": "2026-09-12T00:00:00",
  "modelo": "regresion_logistica",
  "features": ["monto_ewma_ratio", "conteo_ventana_5min", "..."],
  "coeficientes": [0.42, -0.13, "..."],
  "intercepto": -1.85,
  "umbral_decision": 0.5,
  "metricas_validacion": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "auc": 0.0,
    "n_transacciones_validacion": 0
  }
}
```

- `features` fixes the **exact order** the Executor must use to build the
  input vector — never looked up by name at decision time (that would mean
  a costly lookup on the hot path); the order is contractual.
- `coeficientes` and `features` have the same length and the same order —
  validated on load, failing loud if they don't match.
- `metricas_validacion` travels with the artifact so any monitoring layer
  can know how good the current policy is without re-querying the
  calibration process.
- `version` is a simple incrementing integer (not semver — there's no
  backward compatibility to manage yet; it's an internal contract between
  two modules of the same project).

## Why JSON, not pickle/joblib

Same principle used in earlier projects: JSON is human-readable, auditable
at a glance, and doesn't depend on the exact scikit-learn version that
calibrated the model matching the one that reads it in the Executor — the
Executor doesn't even need scikit-learn installed, it only multiplies
numbers.
