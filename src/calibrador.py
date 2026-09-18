"""Calibrator (slow layer) — Phase 1. Trains a classifier on historical
transactions and produces the policy artifact from
`docs/adr/0001-policy-artifact.md` (validated in `src/artefacto.py`,
shared with `ejecutor.py` and `puente.py`). Never decides in real time —
that's the Executor's exclusive responsibility (`ejecutor.py`).

**Design decision**: coefficients are mathematically "un-scaled" before
saving (see `_desescalar`), so the artifact operates directly on raw
features — the Executor never needs to load a `StandardScaler` or
scikit-learn, it just multiplies and adds numbers.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features_recursivas import calcular_features_recursivas_batch

VERSION_ARTEFACTO = 1
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "monto_ewma_global", "conteo_ventana_global"]
COLUMNAS_CRUDAS_REQUERIDAS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}


class DatasetInvalido(Exception):
    """The input CSV doesn't have the expected shape — fails explicitly at
    the data entry point, not with a raw pandas/numpy error several calls
    later."""


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """The OpenML mirror leaves the Class column with literal quotes
    ("'0'", "'1'") from the ARFF->CSV conversion — cleaned here, once, at
    the data entry point. Sorted by Time (**stable** order —
    `kind="stable"`, see below) and the recursive features
    (`features_recursivas.py`) are added over the full history, BEFORE
    splitting into train/val/test — the recursive state evolves
    continuously, it isn't reset at an arbitrary split cut point."""
    df = pd.read_csv(ruta)

    if df.empty:
        raise DatasetInvalido(f"{ruta} no tiene filas")
    faltantes = COLUMNAS_CRUDAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise DatasetInvalido(f"{ruta} no tiene las columnas requeridas: {faltantes}")

    df["Class"] = df["Class"].astype(str).str.strip("'").astype(int)
    # kind="stable" (mergesort): with 1-second resolution on Time and >1.6 transactions/sec
    # on average, ties are frequent -- a non-stable sort (the default quicksort)
    # doesn't preserve the CSV's original order among tied rows, which can subtly alter
    # the causal order used by EWMA/count and make the pipeline non-reproducible across runs.
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return calcular_features_recursivas_batch(df)


def split_temporal(df: pd.DataFrame, frac_train: float = 0.6, frac_val: float = 0.2) -> tuple:
    """Walk-forward split by Time — never random, so future transactions
    never leak into training (statistical honesty). The test split (the
    rest, ~20%) is reserved for Phase 5 — this module never touches it."""
    n = len(df)
    fin_train = int(n * frac_train)
    fin_val = int(n * (frac_train + frac_val))
    return df.iloc[:fin_train], df.iloc[fin_train:fin_val], df.iloc[fin_val:]


def _desescalar(coef_escalados: np.ndarray, intercepto: float, escalador: StandardScaler) -> tuple:
    """score = w·((x-mu)/sigma) + b = (w/sigma)·x + (b - w·mu/sigma) —
    standard algebra so a model trained on scaled data can operate
    directly on raw data."""
    coef_crudos = coef_escalados / escalador.scale_
    intercepto_crudo = intercepto - float(np.sum(coef_escalados * escalador.mean_ / escalador.scale_))
    return coef_crudos, intercepto_crudo


def _mejor_umbral_por_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Searches over the actually observed scores (via `precision_recall_curve`,
    which evaluates every relevant threshold in O(n log n)), not a fixed
    grid.

    **Fix for a real finding** (`docs/PLAN_DE_TRABAJO.md`, walk-forward
    cross-validation): the earlier version used `np.linspace(0.01, 0.99, 99)`,
    capped at 0.99. With `class_weight="balanced"` and strong separation
    between classes, probabilities concentrate near 0 and 1 — the real
    optimal threshold on the production dataset sat in (0.99, 1.0), outside
    the range the earlier grid explored. Verified by hand: F1 kept rising
    from 0.58 (at 0.99, the old cap) to 0.79 (at 0.99999) over the real
    validation split. The only thresholds that matter are the observed
    scores -- between two consecutive scores the predicted partition
    doesn't change, so this finds the exact optimum, not a grid
    approximation."""
    precisiones, recalls, umbrales = precision_recall_curve(y_true, scores)
    if len(umbrales) == 0:
        return 0.5  # no positive case in y_true -- calibrar() already validates this before getting here
    # precision_recall_curve returns an extra point at the end (recall=0, precision=1)
    # with no associated threshold -- discarded to align lengths.
    precisiones, recalls = precisiones[:-1], recalls[:-1]
    denominador = precisiones + recalls
    f1s = np.divide(2 * precisiones * recalls, denominador, out=np.zeros_like(denominador), where=denominador > 0)
    return float(umbrales[np.argmax(f1s)])


def calibrar(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """Trains the logistic regression (class_weight='balanced' — 0.17%
    fraud rate, without this the trivial "never fraud" model would win on
    accuracy) and returns the `ADR_001` artifact, with the threshold that
    maximizes F1 on the validation split."""
    if df_train["Class"].sum() == 0:
        raise DatasetInvalido("df_train no tiene ningún caso positivo — no hay nada que aprender")
    if df_val["Class"].sum() == 0:
        # This used to degrade silently: every F1 comes out 0 (zero_division=0), the chosen
        # threshold stayed fixed at the first value tried (0.01, almost any transaction
        # would be "suspicious"), and it was only noticed because roc_auc_score blows up with a
        # single class -- an sklearn side effect, not an explicit safeguard in this module.
        raise DatasetInvalido("df_val no tiene ningún caso positivo — el umbral por F1 no se puede calibrar razonablemente")

    escalador = StandardScaler()
    X_train = escalador.fit_transform(df_train[FEATURES])
    y_train = df_train["Class"].to_numpy()

    modelo = LogisticRegression(class_weight="balanced", max_iter=1000)
    modelo.fit(X_train, y_train)

    X_val = escalador.transform(df_val[FEATURES])
    y_val = df_val["Class"].to_numpy()
    scores_val = modelo.predict_proba(X_val)[:, 1]

    umbral = _mejor_umbral_por_f1(y_val, scores_val)
    y_pred_val = (scores_val >= umbral).astype(int)

    metricas = {
        "precision": float(precision_score(y_val, y_pred_val, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred_val, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred_val, zero_division=0)),
        "auc": float(roc_auc_score(y_val, scores_val)),
        "n_transacciones_validacion": int(len(y_val)),
    }

    coef_crudos, intercepto_crudo = _desescalar(modelo.coef_[0], float(modelo.intercept_[0]), escalador)

    return {
        "version": VERSION_ARTEFACTO,
        "fecha_calibracion": datetime.now(timezone.utc).isoformat(),
        "modelo": "regresion_logistica",
        "features": FEATURES,
        "coeficientes": coef_crudos.tolist(),
        "intercepto": intercepto_crudo,
        "umbral_decision": umbral,
        "metricas_validacion": metricas,
    }


def guardar_artefacto(artefacto: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(artefacto, indent=2), encoding="utf-8")
