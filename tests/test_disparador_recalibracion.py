import numpy as np
import pandas as pd
import pytest

from src.calibrador import FEATURES
from src.disparador_recalibracion import evaluar_y_recalibrar_si_hace_falta
from src.features_recursivas import calcular_features_recursivas_batch
from src.puente import leer_vigente, publicar

ARTEFACTO_VIEJO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "regresion_logistica",
    "features": ["Amount"],
    "coeficientes": [0.1],
    "intercepto": -1.0,
    "umbral_decision": 0.5,
    "metricas_validacion": {},
}


def _dataset_sintetico(n: int = 2000, seed: int = 0, con_fraude: bool = True) -> pd.DataFrame:
    """Same generator as `tests/test_calibrador.py` -- real, separable
    signal in V1/Amount, so `calibrar()` has something to learn."""
    rng = np.random.default_rng(seed)
    es_fraude = (rng.random(n) < 0.05) if con_fraude else np.zeros(n, dtype=bool)

    data = {"Time": np.arange(n, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n) + (0.5 if i == 1 else 0) * es_fraude
    data["Amount"] = np.where(es_fraude, rng.normal(500, 100, n), rng.normal(50, 20, n)).clip(min=0)
    data["Class"] = es_fraude.astype(int)
    return calcular_features_recursivas_batch(pd.DataFrame(data))


def test_sin_deriva_no_recalibra_ni_publica(tmp_path):
    referencia = _dataset_sintetico(n=2000, seed=0)
    actual = _dataset_sintetico(n=2000, seed=1)  # same generating distribution, different seed
    ruta = tmp_path / "artefacto.json"

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert resultado.recomendacion_recalibrar is False
    assert resultado.se_recalibro is False
    assert resultado.version_artefacto_nueva is None
    assert resultado.error is None
    assert resultado.contribucion_ponderada_features_con_deriva is None  # no current artifact yet
    assert not ruta.exists()


def test_con_deriva_significativa_recalibra_y_publica(tmp_path):
    referencia = _dataset_sintetico(n=2000, seed=0)
    actual = _dataset_sintetico(n=2000, seed=2)
    actual["Amount"] = actual["Amount"] * 5.0 + 1000.0  # strong drift injected, as in scripts/deteccion_deriva.py
    ruta = tmp_path / "artefacto.json"

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert resultado.recomendacion_recalibrar is True
    assert resultado.se_recalibro is True
    assert resultado.version_artefacto_nueva is not None
    assert resultado.error is None
    assert resultado.contribucion_ponderada_features_con_deriva is None  # no current artifact yet
    assert ruta.exists()
    assert leer_vigente(ruta)["version"] == resultado.version_artefacto_nueva


def test_deriva_detectada_pero_ventana_actual_sin_fraude_no_publica_y_reporta_error(tmp_path):
    referencia = _dataset_sintetico(n=2000, seed=0)
    actual = _dataset_sintetico(n=2000, seed=3, con_fraude=False)
    actual["Amount"] = actual["Amount"] * 5.0 + 1000.0  # forces the recalibration branch
    ruta = tmp_path / "artefacto.json"

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert resultado.recomendacion_recalibrar is True
    assert resultado.se_recalibro is False
    assert resultado.version_artefacto_nueva is None
    assert resultado.error is not None
    assert not ruta.exists()


def test_no_sobreescribe_artefacto_vigente_si_recalibracion_falla(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO_VIEJO, ruta)

    referencia = _dataset_sintetico(n=2000, seed=0)
    actual = _dataset_sintetico(n=2000, seed=3, con_fraude=False)
    actual["Amount"] = actual["Amount"] * 5.0 + 1000.0

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert leer_vigente(ruta) == ARTEFACTO_VIEJO
    # ARTEFACTO_VIEJO only knows "Amount" (coefficient 0.1) and "Amount" drifted -> 100% of known weight drifted
    assert resultado.contribucion_ponderada_features_con_deriva == pytest.approx(1.0)


def test_deriva_del_score_dispara_recalibracion_aunque_ninguna_feature_cruce_sola(tmp_path):
    # The case that justifies monitoring the score in addition to each feature: a
    # small shift (0.12) spread across the 28 V features leaves EACH ONE
    # well below PSI 0.25 -- but the current artifact's combined score
    # (which weighs all 28 equally) does cross the threshold. Parameters
    # verified by hand before writing the test.
    ruta = tmp_path / "artefacto.json"
    artefacto_vigente = {
        "version": 1, "fecha_calibracion": "2026-01-01T00:00:00", "modelo": "regresion_logistica",
        "features": FEATURES,
        "coeficientes": [1 / 28] * 28 + [0.0, 0.0, 0.0],  # only V1..V28 carry weight, Amount/recursive ones at 0
        "intercepto": 0.0, "umbral_decision": 0.5, "metricas_validacion": {},
    }
    publicar(artefacto_vigente, ruta)

    referencia = _dataset_sintetico(n=5000, seed=0)
    actual = _dataset_sintetico(n=5000, seed=2)
    for v in [f"V{i}" for i in range(1, 29)]:
        actual[v] = actual[v] + 0.12

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert resultado.n_features_con_deriva == 0  # no individual feature crossed on its own
    assert resultado.deriva_score_interpretacion == "deriva_significativa_recalibrar"
    assert resultado.recomendacion_recalibrar is True  # the score alone is enough to recommend
    assert resultado.se_recalibro is True


def test_contribucion_ponderada_usa_los_coeficientes_del_artefacto_vigente(tmp_path):
    ruta = tmp_path / "artefacto.json"
    coeficientes = [0.0] * len(FEATURES)
    coeficientes[FEATURES.index("Amount")] = 4.0  # the rest at 0 -- only "Amount" carries weight in the current score
    artefacto_vigente = {
        "version": 1, "fecha_calibracion": "2026-01-01T00:00:00", "modelo": "regresion_logistica",
        "features": FEATURES, "coeficientes": coeficientes, "intercepto": -1.0,
        "umbral_decision": 0.5, "metricas_validacion": {},
    }
    publicar(artefacto_vigente, ruta)

    referencia = _dataset_sintetico(n=2000, seed=0)
    actual = _dataset_sintetico(n=2000, seed=2)
    actual["Amount"] = actual["Amount"] * 5.0 + 1000.0  # only Amount drifts strongly

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    # "Amount" is the only feature with weight (4.0) in the current artifact and the only one that drifted -> 1.0
    assert resultado.contribucion_ponderada_features_con_deriva == pytest.approx(1.0)
