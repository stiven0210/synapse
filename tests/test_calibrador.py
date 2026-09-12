import json

import numpy as np
import pandas as pd
import pytest

from src.calibrador import DatasetInvalido, FEATURES, cargar_dataset, calibrar, guardar_artefacto, split_temporal
from src.features_recursivas import calcular_features_recursivas_batch


def _dataset_sintetico(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Dataset sintético con señal real y separable: el fraude tiene Amount
    y V1 sistemáticamente distintos, suficiente para que la regresión
    logística encuentre una frontera de decisión no trivial. Incluye las
    features recursivas -- mismo pipeline que `cargar_dataset` en producción."""
    rng = np.random.default_rng(seed)
    es_fraude = rng.random(n) < 0.05

    data = {"Time": np.arange(n, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n) + (0.5 if i == 1 else 0) * es_fraude
    data["Amount"] = np.where(es_fraude, rng.normal(500, 100, n), rng.normal(50, 20, n)).clip(min=0)
    data["Class"] = es_fraude.astype(int)
    return calcular_features_recursivas_batch(pd.DataFrame(data))


def test_cargar_dataset_limpia_comillas_de_class(tmp_path):
    ruta = tmp_path / "mini.csv"
    columnas = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    filas = [
        [2.0] + [0.1] * 28 + [10.0, "'0'"],
        [1.0] + [0.2] * 28 + [20.0, "'1'"],
    ]
    pd.DataFrame(filas, columns=columnas).to_csv(ruta, index=False)

    df = cargar_dataset(ruta)

    assert df["Class"].tolist() == [1, 0]  # se reordenó por Time (1.0 antes que 2.0)
    assert df["Class"].dtype == int


def test_split_temporal_tamanos_y_sin_traslape():
    df = _dataset_sintetico(n=1000)
    train, val, test = split_temporal(df, frac_train=0.6, frac_val=0.2)

    assert len(train) == 600
    assert len(val) == 200
    assert len(test) == 200
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_calibrar_produce_artefacto_con_schema_de_adr_001():
    df = _dataset_sintetico(n=2000)
    train, val, _ = split_temporal(df)

    artefacto = calibrar(train, val)

    for clave in ("version", "fecha_calibracion", "modelo", "features", "coeficientes", "intercepto", "umbral_decision", "metricas_validacion"):
        assert clave in artefacto
    assert artefacto["features"] == FEATURES
    assert len(artefacto["coeficientes"]) == len(FEATURES)
    assert 0.0 <= artefacto["umbral_decision"] <= 1.0
    for metrica in ("precision", "recall", "f1", "auc", "n_transacciones_validacion"):
        assert metrica in artefacto["metricas_validacion"]


def test_calibrar_encuentra_señal_real_auc_alto():
    # Con separación real entre clases, el AUC en validación debe ser claramente > 0.5 (azar).
    df = _dataset_sintetico(n=3000)
    train, val, _ = split_temporal(df)
    artefacto = calibrar(train, val)
    assert artefacto["metricas_validacion"]["auc"] > 0.8


def test_coeficientes_desescalados_coinciden_exactamente_con_modelo_escalado():
    # Verifica la identidad algebraica score = w·((x-mu)/sigma)+b == (w/sigma)·x + (b - w·mu/sigma)
    # -- no una aproximación, debe coincidir hasta precisión de punto flotante.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = _dataset_sintetico(n=2000)
    train, val, _ = split_temporal(df)

    artefacto = calibrar(train, val)

    escalador = StandardScaler()
    X_train_escalado = escalador.fit_transform(train[FEATURES])
    modelo = LogisticRegression(class_weight="balanced", max_iter=1000)
    modelo.fit(X_train_escalado, train["Class"].to_numpy())

    X_val_crudo = val[FEATURES].to_numpy()
    coef = np.array(artefacto["coeficientes"])
    scores_manual = 1 / (1 + np.exp(-(X_val_crudo @ coef + artefacto["intercepto"])))

    scores_sklearn = modelo.predict_proba(escalador.transform(val[FEATURES]))[:, 1]

    assert np.max(np.abs(scores_manual - scores_sklearn)) < 1e-9


def test_guardar_artefacto_round_trip(tmp_path):
    df = _dataset_sintetico(n=1500)
    train, val, _ = split_temporal(df)
    artefacto = calibrar(train, val)

    ruta = tmp_path / "artefacto.json"
    guardar_artefacto(artefacto, ruta)

    cargado = json.loads(ruta.read_text(encoding="utf-8"))
    assert cargado == artefacto


def test_cargar_dataset_csv_vacio_lanza_error(tmp_path):
    ruta = tmp_path / "vacio.csv"
    columnas = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    pd.DataFrame(columns=columnas).to_csv(ruta, index=False)

    with pytest.raises(DatasetInvalido, match="no tiene filas"):
        cargar_dataset(ruta)


def test_cargar_dataset_columnas_faltantes_lanza_error(tmp_path):
    ruta = tmp_path / "incompleto.csv"
    pd.DataFrame({"Time": [1.0], "Amount": [10.0], "Class": ["'0'"]}).to_csv(ruta, index=False)

    with pytest.raises(DatasetInvalido, match="columnas requeridas"):
        cargar_dataset(ruta)


def test_cargar_dataset_orden_estable_en_empates_de_time(tmp_path):
    # Dos filas con el mismo Time -- el orden relativo original del CSV debe preservarse.
    ruta = tmp_path / "empate.csv"
    columnas = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    fila_a = [5.0] + [0.0] * 28 + [111.0, "'0'"]  # Amount=111 marca "la fila A"
    fila_b = [5.0] + [0.0] * 28 + [222.0, "'0'"]  # Amount=222 marca "la fila B", viene después en el CSV
    pd.DataFrame([fila_a, fila_b], columns=columnas).to_csv(ruta, index=False)

    df = cargar_dataset(ruta)

    assert df["Amount"].tolist() == [111.0, 222.0]  # A sigue antes que B, no se reordenaron


def test_calibrar_sin_positivos_en_train_lanza_error():
    df = _dataset_sintetico(n=200, seed=1)
    df_sin_fraude = df.copy()
    df_sin_fraude["Class"] = 0
    train, val, _ = split_temporal(df_sin_fraude)

    with pytest.raises(DatasetInvalido, match="train"):
        calibrar(train, val)


def test_calibrar_sin_positivos_en_val_lanza_error():
    df = _dataset_sintetico(n=2000, seed=1)
    train, val, _ = split_temporal(df)
    val = val.copy()
    val["Class"] = 0  # fuerza el caso "sin fraude en validación"

    with pytest.raises(DatasetInvalido, match="val"):
        calibrar(train, val)
