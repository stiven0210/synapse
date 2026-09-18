import time

import numpy as np
import pandas as pd
import pytest

from src.calibrador import calibrar, split_temporal
from src.ejecutor import ArtefactoInvalido, Ejecutor, validar_artefacto
from src.features_recursivas import calcular_features_recursivas_batch

ARTEFACTO_VALIDO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "regresion_logistica",
    "features": ["Amount"],
    "coeficientes": [0.1],
    "intercepto": -1.0,
    "umbral_decision": 0.5,
    "metricas_validacion": {},
}


def test_validar_artefacto_valido_no_lanza_error():
    validar_artefacto(ARTEFACTO_VALIDO)  # must not raise


def test_validar_artefacto_campo_faltante():
    invalido = {k: v for k, v in ARTEFACTO_VALIDO.items() if k != "umbral_decision"}
    with pytest.raises(ArtefactoInvalido, match="faltan campos"):
        validar_artefacto(invalido)


def test_validar_artefacto_longitudes_distintas():
    invalido = {**ARTEFACTO_VALIDO, "coeficientes": [0.1, 0.2]}
    with pytest.raises(ArtefactoInvalido, match="distinta longitud"):
        validar_artefacto(invalido)


def test_validar_artefacto_umbral_fuera_de_rango():
    invalido = {**ARTEFACTO_VALIDO, "umbral_decision": 1.5}
    with pytest.raises(ArtefactoInvalido, match=r"\[0,1\]"):
        validar_artefacto(invalido)


def test_ejecutor_rechaza_artefacto_invalido_al_construirse():
    with pytest.raises(ArtefactoInvalido):
        Ejecutor(artefacto={**ARTEFACTO_VALIDO, "umbral_decision": 2.0})


def test_decidir_score_caso_conocido():
    # z = -1.0 + 0.1*20 = 1.0 -> sigmoid(1.0) = 0.7310585786300049
    ejecutor = Ejecutor(artefacto=ARTEFACTO_VALIDO)
    resultado = ejecutor.decidir({"Amount": 20.0, "Time": 0.0})
    assert resultado["score"] == pytest.approx(0.7310585786300049)
    assert resultado["es_sospechosa"] is True  # 0.731 >= threshold 0.5


def test_decidir_bajo_el_umbral_no_es_sospechosa():
    # z = -1.0 + 0.1*5 = -0.5 -> sigmoid(-0.5) ~ 0.377 < 0.5
    ejecutor = Ejecutor(artefacto=ARTEFACTO_VALIDO)
    resultado = ejecutor.decidir({"Amount": 5.0, "Time": 0.0})
    assert resultado["es_sospechosa"] is False


def test_ejecutor_coincide_exactamente_con_calculo_batch_end_to_end():
    # The definitive test: calibrate in batch mode, then process the SAME sequence
    # of transactions one by one through the Executor -- the scores must match
    # exactly the ones the calibration process itself produces over the
    # validation split (same pipeline, no leakage, no skew).
    rng = np.random.default_rng(7)
    n = 1000
    es_fraude = rng.random(n) < 0.05
    data = {"Time": np.arange(n, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n) + (0.5 if i == 1 else 0) * es_fraude
    data["Amount"] = np.where(es_fraude, rng.normal(500, 100, n), rng.normal(50, 20, n)).clip(min=0)
    data["Class"] = es_fraude.astype(int)
    df = calcular_features_recursivas_batch(pd.DataFrame(data))

    train, val, _ = split_temporal(df, frac_train=0.7, frac_val=0.3)
    artefacto = calibrar(train, val)

    from src.calibrador import FEATURES

    z_batch = artefacto["intercepto"] + (val[FEATURES].to_numpy() @ np.array(artefacto["coeficientes"]))
    scores_batch = 1 / (1 + np.exp(-z_batch))

    # The Executor must ALSO process `train` first -- `val`'s recursive features in
    # batch mode load with the continuous history from the start of the
    # dataset (train+val), not from an empty state that starts right at the cut.
    ejecutor = Ejecutor(artefacto=artefacto)
    columnas = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
    for _, fila in train.iterrows():
        ejecutor.decidir(fila[columnas].to_dict())

    scores_ejecutor = []
    for _, fila in val.iterrows():
        resultado = ejecutor.decidir(fila[columnas].to_dict())
        scores_ejecutor.append(resultado["score"])

    assert np.array(scores_ejecutor) == pytest.approx(scores_batch, abs=1e-9)


def test_benchmark_latencia_real_microsegundos():
    ejecutor = Ejecutor(artefacto=ARTEFACTO_VALIDO)
    n = 10_000
    transacciones = [{"Amount": float(i % 100), "Time": float(i)} for i in range(n)]

    inicio = time.perf_counter()
    for t in transacciones:
        ejecutor.decidir(t)
    duracion_total = time.perf_counter() - inicio

    microsegundos_por_decision = (duracion_total / n) * 1_000_000
    print(f"\nLatencia real: {microsegundos_por_decision:.2f} microsegundos/decisión ({n} decisiones)")

    # Generous bound (not a vendor's microsecond promise) -- the real number
    # printed above is the measurement that matters; this only guards against a gross regression.
    assert microsegundos_por_decision < 500.0
