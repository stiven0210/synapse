import numpy as np
import pandas as pd
import pytest

from src.artefacto import calcular_scores
from src.ejecutor import Ejecutor

ARTEFACTO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "regresion_logistica",
    "features": ["Amount"],
    "coeficientes": [0.1],
    "intercepto": -1.0,
    "umbral_decision": 0.5,
    "metricas_validacion": {},
}


def test_calcular_scores_caso_numerico_a_mano():
    df = pd.DataFrame({"Amount": [0.0, 10.0, 100.0]})

    scores = calcular_scores(ARTEFACTO, df)

    # z = -1.0 + 0.1*Amount
    esperados = 1.0 / (1.0 + np.exp(-(-1.0 + 0.1 * np.array([0.0, 10.0, 100.0]))))
    assert scores == pytest.approx(esperados)


def test_calcular_scores_coincide_exactamente_con_ejecutor_decidir_fila_por_fila():
    # Train/serve parity: the batch version (offline diagnostic) must produce
    # exactly the same score as the real incremental version
    # (Ejecutor.decidir(), the hot path) for the same rows.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Amount": rng.normal(50, 20, 200).clip(min=0), "Time": np.arange(200, dtype=float)})

    scores_batch = calcular_scores(ARTEFACTO, df)

    ejecutor = Ejecutor(artefacto=ARTEFACTO)
    scores_incremental = np.array([
        ejecutor.decidir({"Amount": fila["Amount"], "Time": fila["Time"]})["score"]
        for _, fila in df.iterrows()
    ])

    assert scores_batch == pytest.approx(scores_incremental, abs=1e-9)


def test_calcular_scores_numericamente_estable_para_z_muy_negativo():
    df = pd.DataFrame({"Amount": [-100_000.0]})
    artefacto = {**ARTEFACTO, "coeficientes": [1.0]}  # very negative z

    scores = calcular_scores(artefacto, df)

    assert np.isfinite(scores).all()
    assert scores[0] == pytest.approx(0.0, abs=1e-9)


def test_calcular_scores_numericamente_estable_para_z_muy_positivo():
    df = pd.DataFrame({"Amount": [100_000.0]})
    artefacto = {**ARTEFACTO, "coeficientes": [1.0]}  # very positive z

    scores = calcular_scores(artefacto, df)

    assert np.isfinite(scores).all()
    assert scores[0] == pytest.approx(1.0, abs=1e-9)
