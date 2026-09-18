import numpy as np
import pandas as pd
import pytest

from src.artefacto_arboles import validar_artefacto_arboles
from src.calibrador import DatasetInvalido
from src.calibrador_arboles import (
    FEATURES_ARBOLES,
    FEATURES_MONITOREO_DERIVA,
    _calibrar_detalle,
    _mejor_umbral_por_f1,
    cargar_dataset,
    calibrar,
    construir_features,
)
from src.ejecutor_arboles import EjecutorArboles
from src.features_recursivas_cuenta import calcular_frecuencia_poblacional_categoria


def _dataset_sintetico(n=3000, tasa_fraude=0.08, semilla=0) -> pd.DataFrame:
    # Deliberately separable (high amt -> fraud) -- not meant to be realistic, just to
    # exercise the full pipeline (threshold extraction, lookup table, calibration) fast in tests.
    rng = np.random.default_rng(semilla)
    cuentas = rng.choice(["A", "B", "C", "D", "E"], size=n)
    categorias = rng.choice(["grocery", "gas", "entertainment"], size=n)
    amt_base = rng.exponential(50, n)
    is_fraud = (rng.uniform(0, 1, n) < tasa_fraude).astype(int)
    amt = np.where(is_fraud == 1, amt_base + 400, amt_base)
    unix_time = np.arange(n) * 60 + 1_700_000_000
    return pd.DataFrame({"cc_num": cuentas, "amt": amt, "unix_time": unix_time, "category": categorias, "is_fraud": is_fraud})


@pytest.fixture(scope="module")
def pipeline_sintetico():
    from src.calibrador import split_temporal

    df = _dataset_sintetico()
    df_train_crudo, _, _ = split_temporal(df)
    frecuencia = calcular_frecuencia_poblacional_categoria(df_train_crudo)
    df_feat = construir_features(df, frecuencia)
    df_train, df_val, df_test = split_temporal(df_feat)
    artefacto, modelo = _calibrar_detalle(df_train, df_val, frecuencia)
    return {"artefacto": artefacto, "modelo": modelo, "df_train": df_train, "df_val": df_val, "df_test": df_test}


def test_calibrar_produce_artefacto_valido(pipeline_sintetico):
    validar_artefacto_arboles(pipeline_sintetico["artefacto"])  # must not raise


def test_calibrar_aprende_la_separacion_sintetica(pipeline_sintetico):
    m = pipeline_sintetico["artefacto"]["metricas_validacion"]
    # The synthetic rule (high amt -> fraud) is trivially separable -- the model should
    # learn it almost perfectly; this is NOT a claim about performance on real fraud.
    assert m["auc_pr_val"] > 0.95
    assert m["recall_val"] > 0.8


def test_calibrar_rechaza_train_sin_positivos():
    df = _dataset_sintetico(tasa_fraude=0.08)
    df_sin_fraude = df.copy()
    df_sin_fraude["is_fraud"] = 0
    frecuencia = calcular_frecuencia_poblacional_categoria(df_sin_fraude)
    df_feat = construir_features(df_sin_fraude, frecuencia)
    with pytest.raises(DatasetInvalido):
        calibrar(df_feat.iloc[:1800], df_feat.iloc[1800:2400], frecuencia)


def test_cargar_dataset_valida_columnas_requeridas(tmp_path):
    ruta = tmp_path / "malo.csv"
    pd.DataFrame({"amt": [1.0, 2.0]}).to_csv(ruta, index=False)
    with pytest.raises(DatasetInvalido):
        cargar_dataset(ruta)


def test_cargar_dataset_ordena_por_unix_time(tmp_path):
    ruta = tmp_path / "bueno.csv"
    pd.DataFrame({
        "cc_num": ["A", "A"], "amt": [1.0, 2.0], "unix_time": [200, 100],
        "category": ["x", "x"], "is_fraud": [0, 0],
    }).to_csv(ruta, index=False)
    df = cargar_dataset(ruta)
    assert df["unix_time"].tolist() == [100, 200]


def test_mejor_umbral_por_f1_caso_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    umbral = _mejor_umbral_por_f1(y, scores)
    assert 0.3 < umbral <= 0.7  # any cutoff in the gap separates perfectly


def test_tabla_busqueda_bit_exacta_contra_predict_proba_dataset_sintetico(pipeline_sintetico):
    # Fast version (synthetic dataset, ~3000 rows) of the real bit-exact test in
    # tests/test_ejecutor_arboles.py -- always runs, doesn't depend on the real Sparkov
    # CSV being present. Same discipline: replay from the start of the ENTIRE history, not just test.
    artefacto = pipeline_sintetico["artefacto"]
    modelo = pipeline_sintetico["modelo"]
    df_test = pipeline_sintetico["df_test"]
    df_full = pd.concat([pipeline_sintetico["df_train"], pipeline_sintetico["df_val"], df_test], ignore_index=True)

    ejecutor = EjecutorArboles(artefacto=artefacto)
    n_total, n_test = len(df_full), len(df_test)
    scores = np.empty(n_total)
    for i, fila in enumerate(df_full.itertuples(index=False)):
        transaccion = {"cc_num": fila.cc_num, "amt": fila.amt, "unix_time": int(fila.unix_time), "category": fila.category}
        scores[i] = ejecutor.decidir(transaccion)["score"]

    scores_test_ejecutor = scores[n_total - n_test:]
    X_test = df_test[FEATURES_ARBOLES].to_numpy(dtype=float)
    scores_crudos_reales = modelo.predict_proba(X_test)[:, 1]
    xs = np.array(artefacto["calibracion_isotonica"]["x"])
    ys = np.array(artefacto["calibracion_isotonica"]["y"])
    scores_test_reales = np.interp(scores_crudos_reales, xs, ys)

    assert np.abs(scores_test_ejecutor - scores_test_reales).max() < 1e-9


def test_features_monitoreo_deriva_excluye_frecuencia_categoria_expandida():
    # Doc section 19: frecuencia_categoria_expandida is non-stationary by design and
    # triggers recalibration on 85% of year-over-year transitions (vs. 23% for the rest) --
    # a constant false alarm, not a real signal. It's excluded from the set passed to
    # deriva.py::evaluar_deriva(), without touching that module (it's already generic over any column).
    assert "frecuencia_categoria_expandida" not in FEATURES_MONITOREO_DERIVA
    assert set(FEATURES_MONITOREO_DERIVA) == set(FEATURES_ARBOLES) - {"frecuencia_categoria_expandida"}
    assert len(FEATURES_MONITOREO_DERIVA) == len(FEATURES_ARBOLES) - 1
