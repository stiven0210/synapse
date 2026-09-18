import json

import numpy as np
import pandas as pd
import pytest

from src.calibrador import DatasetInvalido, FEATURES, _mejor_umbral_por_f1, cargar_dataset, calibrar, guardar_artefacto, split_temporal
from src.features_recursivas import calcular_features_recursivas_batch


def _dataset_sintetico(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset with a real, separable signal: fraud has systematically
    different Amount and V1, enough for logistic regression to find a
    non-trivial decision boundary. Includes the recursive features -- same
    pipeline as `cargar_dataset` in production."""
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

    assert df["Class"].tolist() == [1, 0]  # reordered by Time (1.0 before 2.0)
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
    # With real separation between classes, validation AUC should be clearly > 0.5 (chance).
    df = _dataset_sintetico(n=3000)
    train, val, _ = split_temporal(df)
    artefacto = calibrar(train, val)
    assert artefacto["metricas_validacion"]["auc"] > 0.8


def test_coeficientes_desescalados_coinciden_exactamente_con_modelo_escalado():
    # Verifies the algebraic identity score = w·((x-mu)/sigma)+b == (w/sigma)·x + (b - w·mu/sigma)
    # -- not an approximation, must match to floating-point precision.
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
    # Two rows with the same Time -- the CSV's original relative order must be preserved.
    ruta = tmp_path / "empate.csv"
    columnas = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    fila_a = [5.0] + [0.0] * 28 + [111.0, "'0'"]  # Amount=111 marks "row A"
    fila_b = [5.0] + [0.0] * 28 + [222.0, "'0'"]  # Amount=222 marks "row B", comes after in the CSV
    pd.DataFrame([fila_a, fila_b], columns=columnas).to_csv(ruta, index=False)

    df = cargar_dataset(ruta)

    assert df["Amount"].tolist() == [111.0, 222.0]  # A still comes before B, not reordered


def test_calibrar_sin_positivos_en_train_lanza_error():
    df = _dataset_sintetico(n=200, seed=1)
    df_sin_fraude = df.copy()
    df_sin_fraude["Class"] = 0
    train, val, _ = split_temporal(df_sin_fraude)

    with pytest.raises(DatasetInvalido, match="train"):
        calibrar(train, val)


def test_mejor_umbral_por_f1_no_queda_topado_cuando_el_optimo_esta_sobre_0_99():
    # Real finding (see _mejor_umbral_por_f1's docstring): with strong separation,
    # all the scores -- from both classes -- can fall above 0.99. The old
    # version (fixed grid up to 0.99) could never separate this case: every
    # candidate <= 0.99 predicts EVERYTHING as positive (F1 forced to 0.5714, see
    # the hand computation below). The truly optimal threshold (0.998) does
    # separate it perfectly.
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.991, 0.993, 0.995, 0.998, 0.9995])

    umbral = _mejor_umbral_por_f1(y_true, scores)

    y_pred = (scores >= umbral).astype(int)
    assert list(y_pred) == [0, 0, 0, 1, 1]  # perfect separation -- F1 = 1.0
    assert umbral > 0.99  # the finding: the real optimum is outside the range the old grid explored


def test_calibrar_sin_positivos_en_val_lanza_error():
    df = _dataset_sintetico(n=2000, seed=1)
    train, val, _ = split_temporal(df)
    val = val.copy()
    val["Class"] = 0  # forces the "no fraud in validation" case

    with pytest.raises(DatasetInvalido, match="val"):
        calibrar(train, val)
