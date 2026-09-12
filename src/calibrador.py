"""Calibrador (capa lenta) — Fase 1. Entrena un clasificador sobre
transacciones históricas y produce el artefacto de política de
`docs/ADR_001_artefacto_de_politica.md` (validado en `src/artefacto.py`,
compartido con `ejecutor.py` y `puente.py`). Nunca decide en tiempo real —
eso es responsabilidad exclusiva del Ejecutor (`ejecutor.py`).

**Decisión de diseño**: los coeficientes se "des-escalan" matemáticamente
antes de guardarlos (ver `_desescalar`), de modo que el artefacto opere
directo sobre las features crudas — el Ejecutor nunca necesita cargar un
`StandardScaler` ni scikit-learn, solo multiplica y suma números.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features_recursivas import calcular_features_recursivas_batch

VERSION_ARTEFACTO = 1
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "monto_ewma_global", "conteo_ventana_global"]
COLUMNAS_CRUDAS_REQUERIDAS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}


class DatasetInvalido(Exception):
    """El CSV de entrada no tiene la forma esperada — falla explícito en el
    punto de entrada de datos, no con un error crudo de pandas/numpy varias
    llamadas después."""


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """El mirror de OpenML deja la columna Class con comillas literales
    ("'0'", "'1'") por la conversión ARFF->CSV — se limpia aquí, una sola
    vez, en el punto de entrada de datos. Se ordena por Time (orden
    **estable** — `kind="stable"`, ver más abajo) y se agregan las features
    recursivas (`features_recursivas.py`) sobre el historial completo,
    ANTES de partir en train/val/test — el estado recursivo evoluciona de
    forma continua, no se reinicia en un punto de corte arbitrario del split."""
    df = pd.read_csv(ruta)

    if df.empty:
        raise DatasetInvalido(f"{ruta} no tiene filas")
    faltantes = COLUMNAS_CRUDAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise DatasetInvalido(f"{ruta} no tiene las columnas requeridas: {faltantes}")

    df["Class"] = df["Class"].astype(str).str.strip("'").astype(int)
    # kind="stable" (mergesort): con resolución de 1 segundo en Time y >1.6 transacciones/seg
    # en promedio, hay empates frecuentes -- un sort no estable (el quicksort por defecto)
    # no preserva el orden original del CSV entre filas empatadas, lo que puede alterar sutilmente
    # el orden causal usado por EWMA/conteo y hacer el pipeline no reproducible entre corridas.
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return calcular_features_recursivas_batch(df)


def split_temporal(df: pd.DataFrame, frac_train: float = 0.6, frac_val: float = 0.2) -> tuple:
    """Split walk-forward por Time — nunca aleatorio, para no filtrar
    transacciones futuras hacia el entrenamiento (honestidad estadística).
    El tramo de prueba (el resto, ~20%) queda reservado para Fase 5 — este
    módulo no lo toca."""
    n = len(df)
    fin_train = int(n * frac_train)
    fin_val = int(n * (frac_train + frac_val))
    return df.iloc[:fin_train], df.iloc[fin_train:fin_val], df.iloc[fin_val:]


def _desescalar(coef_escalados: np.ndarray, intercepto: float, escalador: StandardScaler) -> tuple:
    """score = w·((x-mu)/sigma) + b = (w/sigma)·x + (b - w·mu/sigma) —
    álgebra estándar para que un modelo entrenado sobre datos escalados
    opere directo sobre datos crudos."""
    coef_crudos = coef_escalados / escalador.scale_
    intercepto_crudo = intercepto - float(np.sum(coef_escalados * escalador.mean_ / escalador.scale_))
    return coef_crudos, intercepto_crudo


def _mejor_umbral_por_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    mejor_umbral, mejor_f1 = 0.5, -1.0
    for umbral in np.linspace(0.01, 0.99, 99):
        y_pred = (scores >= umbral).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > mejor_f1:
            mejor_umbral, mejor_f1 = float(umbral), f1
    return mejor_umbral


def calibrar(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """Entrena la regresión logística (class_weight='balanced' — 0.17% de
    fraude, sin esto el modelo trivial "nunca es fraude" ganaría en
    accuracy) y devuelve el artefacto de `ADR_001`, con el umbral que
    maximiza F1 en el tramo de validación."""
    if df_train["Class"].sum() == 0:
        raise DatasetInvalido("df_train no tiene ningún caso positivo — no hay nada que aprender")
    if df_val["Class"].sum() == 0:
        # Antes esto degradaba en silencio: todos los F1 dan 0 (zero_division=0), el umbral
        # elegido quedaba fijo en el primer valor probado (0.01, casi cualquier transacción
        # sería "sospechosa"), y solo se notaba porque roc_auc_score revienta con una sola
        # clase -- un efecto colateral de sklearn, no una protección explícita de este módulo.
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
