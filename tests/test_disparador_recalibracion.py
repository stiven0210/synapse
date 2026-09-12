import numpy as np
import pandas as pd

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
    """Mismo generador que `tests/test_calibrador.py` -- señal real y
    separable en V1/Amount, para que `calibrar()` tenga algo que aprender."""
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
    actual = _dataset_sintetico(n=2000, seed=1)  # misma distribución generadora, otra semilla
    ruta = tmp_path / "artefacto.json"

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert resultado.recomendacion_recalibrar is False
    assert resultado.se_recalibro is False
    assert resultado.version_artefacto_nueva is None
    assert resultado.error is None
    assert not ruta.exists()


def test_con_deriva_significativa_recalibra_y_publica(tmp_path):
    referencia = _dataset_sintetico(n=2000, seed=0)
    actual = _dataset_sintetico(n=2000, seed=2)
    actual["Amount"] = actual["Amount"] * 5.0 + 1000.0  # deriva fuerte inyectada, como en scripts/deteccion_deriva.py
    ruta = tmp_path / "artefacto.json"

    resultado = evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert resultado.recomendacion_recalibrar is True
    assert resultado.se_recalibro is True
    assert resultado.version_artefacto_nueva is not None
    assert resultado.error is None
    assert ruta.exists()
    assert leer_vigente(ruta)["version"] == resultado.version_artefacto_nueva


def test_deriva_detectada_pero_ventana_actual_sin_fraude_no_publica_y_reporta_error(tmp_path):
    referencia = _dataset_sintetico(n=2000, seed=0)
    actual = _dataset_sintetico(n=2000, seed=3, con_fraude=False)
    actual["Amount"] = actual["Amount"] * 5.0 + 1000.0  # fuerza la rama de recalibración
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

    evaluar_y_recalibrar_si_hace_falta(referencia, actual, columnas_deriva=FEATURES, ruta_artefacto=ruta)

    assert leer_vigente(ruta) == ARTEFACTO_VIEJO
