import numpy as np
import pandas as pd
import pytest

from src.deriva import calcular_psi, evaluar_deriva, evaluar_deriva_score, evaluar_ks, interpretar_psi, ponderar_deriva_por_coeficiente


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
    drift_fuerte = rng.normal(5, 1, 5000)  # shift of 5 standard deviations
    psi = calcular_psi(ref, drift_fuerte)
    assert psi > 0.25
    assert interpretar_psi(psi) == "deriva_significativa_recalibrar"


def test_psi_referencia_sin_variacion_devuelve_cero_sin_error():
    ref = np.full(100, 5.0)  # all values equal -- degenerate percentiles
    actual = np.array([1.0, 2.0, 3.0])
    assert calcular_psi(ref, actual) == 0.0


def test_psi_es_simetrico_en_deteccion_pero_no_en_valor():
    # PSI(ref, actual) doesn't have to equal PSI(actual, ref) -- the bins are fixed
    # over the reference -- but both must detect significant drift in the same
    # strong-shift situation.
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
    df_actual = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(6, 1, 2000)})  # only "b" drifted

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


def test_evaluar_deriva_score_misma_distribucion_no_recomienda_recalibrar():
    rng = np.random.default_rng(0)
    scores_ref = rng.uniform(0, 1, 3000)
    scores_actual = rng.uniform(0, 1, 3000)

    reporte = evaluar_deriva_score(scores_ref, scores_actual)

    assert reporte["psi_interpretacion"] == "sin_deriva_significativa"


def test_evaluar_deriva_score_con_deriva_fuerte_la_detecta():
    rng = np.random.default_rng(0)
    scores_ref = rng.uniform(0, 0.3, 3000)
    scores_actual = rng.uniform(0.6, 1.0, 3000)  # the model started flagging everything as suspicious

    reporte = evaluar_deriva_score(scores_ref, scores_actual)

    assert reporte["psi_interpretacion"] == "deriva_significativa_recalibrar"
    assert reporte["ks"]["hay_deriva"] is True


def test_evaluar_deriva_score_detecta_deriva_que_ninguna_feature_individual_muestra_sola():
    # The case that justifies monitoring the score in addition to each feature: a
    # small shift (0.11 standard deviations) spread across 30 features
    # leaves EACH feature far below the PSI threshold -- but the combined
    # score (which aggregates the net effect of all 30) does cross the threshold.
    # Parameters verified by hand before writing the test, not tuned
    # afterward to "make it pass".
    rng = np.random.default_rng(11)
    n, k, delta = 5000, 30, 0.11
    ref = rng.normal(0, 1, (n, k))
    actual = rng.normal(delta, 1, (n, k))

    for i in range(k):
        psi_feature = calcular_psi(ref[:, i], actual[:, i])
        assert interpretar_psi(psi_feature) != "deriva_significativa_recalibrar"

    score_ref = ref.mean(axis=1)
    score_actual = actual.mean(axis=1)
    reporte_score = evaluar_deriva_score(score_ref, score_actual)

    assert reporte_score["psi_interpretacion"] == "deriva_significativa_recalibrar"


def test_ponderar_deriva_por_coeficiente_calcula_contribucion_ponderada():
    rng = np.random.default_rng(2)
    df_ref = pd.DataFrame({
        "a": rng.normal(0, 1, 2000),
        "b": rng.normal(0, 1, 2000),
        "c": rng.normal(0, 1, 2000),
    })
    df_actual = pd.DataFrame({
        "a": rng.normal(6, 1, 2000),  # strong drift
        "b": rng.normal(0, 1, 2000),  # no drift
        "c": rng.normal(6, 1, 2000),  # strong drift
    })
    reporte = evaluar_deriva(df_ref, df_actual, columnas=["a", "b", "c"])
    assert reporte["por_feature"]["a"]["psi_interpretacion"] == "deriva_significativa_recalibrar"
    assert reporte["por_feature"]["b"]["psi_interpretacion"] == "sin_deriva_significativa"
    assert reporte["por_feature"]["c"]["psi_interpretacion"] == "deriva_significativa_recalibrar"

    # peso_total = |3| + |1| + |0| = 4 -- "a" and "c" drifted, weighing 3+0=3 -> 3/4 = 0.75
    ponderado = ponderar_deriva_por_coeficiente(reporte, features=["a", "b", "c"], coeficientes=[3.0, 1.0, 0.0])

    assert ponderado["contribucion_ponderada_features_con_deriva"] == pytest.approx(0.75)
    assert ponderado["por_feature"]["a"]["peso_relativo_en_score"] == pytest.approx(0.75)
    assert ponderado["por_feature"]["b"]["peso_relativo_en_score"] == pytest.approx(0.25)
    assert ponderado["por_feature"]["c"]["peso_relativo_en_score"] == pytest.approx(0.0)
    assert ponderado["recomendacion_recalibrar"] == reporte["recomendacion_recalibrar"]
    assert ponderado["n_features_con_deriva_psi"] == reporte["n_features_con_deriva_psi"]


def test_ponderar_deriva_sin_features_con_deriva_da_contribucion_cero():
    rng = np.random.default_rng(3)
    df_ref = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    df_actual = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    reporte = evaluar_deriva(df_ref, df_actual, columnas=["a"])

    ponderado = ponderar_deriva_por_coeficiente(reporte, features=["a"], coeficientes=[5.0])
    assert ponderado["contribucion_ponderada_features_con_deriva"] == 0.0


def test_ponderar_deriva_con_todos_los_coeficientes_en_cero_no_divide_por_cero():
    rng = np.random.default_rng(4)
    df_ref = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    df_actual = pd.DataFrame({"a": rng.normal(6, 1, 2000)})
    reporte = evaluar_deriva(df_ref, df_actual, columnas=["a"])

    ponderado = ponderar_deriva_por_coeficiente(reporte, features=["a"], coeficientes=[0.0])
    assert ponderado["contribucion_ponderada_features_con_deriva"] == 0.0
    assert ponderado["por_feature"]["a"]["peso_relativo_en_score"] is None


def test_ponderar_deriva_feature_sin_coeficiente_conocido_no_cuenta_en_contribucion():
    rng = np.random.default_rng(5)
    df_ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "d": rng.normal(0, 1, 2000)})
    df_actual = pd.DataFrame({"a": rng.normal(0, 1, 2000), "d": rng.normal(6, 1, 2000)})
    reporte = evaluar_deriva(df_ref, df_actual, columnas=["a", "d"])  # "d" drifted

    # the current artifact only knows "a" -- "d" isn't in features/coeficientes
    ponderado = ponderar_deriva_por_coeficiente(reporte, features=["a"], coeficientes=[2.0])

    assert ponderado["por_feature"]["d"]["coeficiente"] is None
    assert ponderado["por_feature"]["d"]["peso_relativo_en_score"] is None
    assert ponderado["contribucion_ponderada_features_con_deriva"] == 0.0  # "d" carries no weight in the current score
