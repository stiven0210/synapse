import numpy as np
import pandas as pd
import pytest

from src.features_recursivas import calcular_features_recursivas_batch
from src.validacion_cruzada import generar_folds_walk_forward, validar_walk_forward_multi_fold


def _dataset_sintetico(n: int, seed: int = 0, fraude_solo_antes_de: int | None = None) -> pd.DataFrame:
    """Real, separable signal (same pattern as test_calibrador.py). If
    `fraude_solo_antes_de` is given, fraud only occurs in rows with an index
    below that cutoff -- to deliberately force a late fold to have no
    positive cases in validation."""
    rng = np.random.default_rng(seed)
    es_fraude = rng.random(n) < 0.05
    if fraude_solo_antes_de is not None:
        es_fraude[fraude_solo_antes_de:] = False

    data = {"Time": np.arange(n, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n) + (0.5 if i == 1 else 0) * es_fraude
    data["Amount"] = np.where(es_fraude, rng.normal(500, 100, n), rng.normal(50, 20, n)).clip(min=0)
    data["Class"] = es_fraude.astype(int)
    return calcular_features_recursivas_batch(pd.DataFrame(data))


def test_generar_folds_walk_forward_ventana_expansiva_sin_traslape():
    df = _dataset_sintetico(n=6000)
    folds = generar_folds_walk_forward(df, n_folds=5, frac_train_inicial=0.5)

    assert len(folds) == 5
    tamanos_train = [len(train) for train, _ in folds]
    assert tamanos_train == sorted(tamanos_train)  # train expands fold by fold, never shrinks

    for train, val in folds:
        assert train["Time"].max() < val["Time"].min()  # val always after train -- no future leakage

    # validation blocks don't overlap between consecutive folds
    for (_, val_a), (_, val_b) in zip(folds, folds[1:]):
        assert val_a["Time"].max() < val_b["Time"].min()


def test_generar_folds_walk_forward_dataset_muy_chico_lanza_error():
    df = _dataset_sintetico(n=20)
    with pytest.raises(ValueError, match="no alcanza"):
        generar_folds_walk_forward(df, n_folds=10, frac_train_inicial=0.9)


def test_validar_walk_forward_con_senal_estable_da_auc_alto_y_consistente():
    df = _dataset_sintetico(n=6000, seed=1)  # fraud spread across the whole dataset
    resumen = validar_walk_forward_multi_fold(df, n_folds=5, frac_train_inicial=0.5)

    assert resumen["n_folds_fallidos"] == 0
    assert resumen["n_folds_exitosos"] == 5
    assert resumen["auc_media"] > 0.8
    assert resumen["auc_desviacion_estandar"] < 0.1  # the signal is real and stable over time, not a one-split fluke


def test_validar_walk_forward_fold_sin_positivos_en_validacion_se_registra_sin_reventar_los_demas():
    # Fraud only in the first 70% of the dataset -- folds whose validation
    # block falls after that cutoff have no positive cases at all.
    df = _dataset_sintetico(n=6000, seed=2, fraude_solo_antes_de=4200)
    resumen = validar_walk_forward_multi_fold(df, n_folds=5, frac_train_inicial=0.5)

    # folds 0 and 1 (val ends at 3600 and 4200) do have fraud; 2, 3, and 4 don't.
    assert resumen["n_folds_exitosos"] == 2
    assert resumen["n_folds_fallidos"] == 3
    assert {f["fold"] for f in resumen["folds_fallidos"]} == {2, 3, 4}
    assert "auc_media" in resumen  # the successful folds do get aggregated


def test_validar_walk_forward_todos_los_folds_fallidos_no_agrega_metricas_inexistentes():
    df = _dataset_sintetico(n=6000, seed=3, fraude_solo_antes_de=1)  # practically no fraud in validation
    resumen = validar_walk_forward_multi_fold(df, n_folds=5, frac_train_inicial=0.5)

    assert resumen["n_folds_exitosos"] == 0
    assert resumen["n_folds_fallidos"] == 5
    assert "auc_media" not in resumen  # nothing to average -- no number gets made up
