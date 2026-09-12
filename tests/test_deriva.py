import numpy as np
import pandas as pd
import pytest

from src.deriva import calcular_psi, evaluar_deriva, evaluar_ks, interpretar_psi


def test_psi_misma_distribucion_es_cercano_a_cero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    igual = rng.normal(0, 1, 5000)
    psi = calcular_psi(ref, igual)
    assert psi < 0.05
    assert interpretar_psi(psi) == "sin_deriva_significativa"


def test_psi_deriva_fuerte_supera_el_umbral_significativo():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    drift_fuerte = rng.normal(5, 1, 5000)  # desplazamiento de 5 desviaciones estándar
    psi = calcular_psi(ref, drift_fuerte)
    assert psi > 0.25
    assert interpretar_psi(psi) == "deriva_significativa_recalibrar"


def test_psi_referencia_sin_variacion_devuelve_cero_sin_error():
    ref = np.full(100, 5.0)  # todos los valores iguales -- percentiles degenerados
    actual = np.array([1.0, 2.0, 3.0])
    assert calcular_psi(ref, actual) == 0.0


def test_psi_es_simetrico_en_deteccion_pero_no_en_valor():
    # PSI(ref, actual) no tiene que ser igual a PSI(actual, ref) -- los bins se fijan
    # sobre la referencia -- pero ambos deben detectar deriva significativa en la misma
    # situación de desplazamiento fuerte.
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 3000)
    b = rng.normal(4, 1, 3000)
    assert interpretar_psi(calcular_psi(a, b)) == "deriva_significativa_recalibrar"
    assert interpretar_psi(calcular_psi(b, a)) == "deriva_significativa_recalibrar"


def test_ks_misma_distribucion_no_marca_deriva():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    igual = rng.normal(0, 1, 5000)
    resultado = evaluar_ks(ref, igual)
    assert resultado["hay_deriva"] is False
    assert resultado["pvalue"] > 0.05


def test_ks_deriva_fuerte_marca_deriva():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    drift_fuerte = rng.normal(5, 1, 5000)
    resultado = evaluar_ks(ref, drift_fuerte)
    assert resultado["hay_deriva"] is True
    assert resultado["pvalue"] < 0.05


def test_evaluar_deriva_reporta_por_columna_y_recomienda_recalibrar():
    rng = np.random.default_rng(2)
    df_ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    df_actual = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(6, 1, 2000)})  # solo "b" derivó

    reporte = evaluar_deriva(df_ref, df_actual, columnas=["a", "b"])

    assert reporte["por_feature"]["a"]["psi_interpretacion"] == "sin_deriva_significativa"
    assert reporte["por_feature"]["b"]["psi_interpretacion"] == "deriva_significativa_recalibrar"
    assert reporte["n_features_con_deriva_psi"] == 1
    assert reporte["recomendacion_recalibrar"] is True


def test_evaluar_deriva_sin_ninguna_columna_con_deriva_no_recomienda_recalibrar():
    rng = np.random.default_rng(3)
    df_ref = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    df_actual = pd.DataFrame({"a": rng.normal(0, 1, 2000)})

    reporte = evaluar_deriva(df_ref, df_actual, columnas=["a"])

    assert reporte["recomendacion_recalibrar"] is False
