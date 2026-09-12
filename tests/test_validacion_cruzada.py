import numpy as np
import pandas as pd
import pytest

from src.features_recursivas import calcular_features_recursivas_batch
from src.validacion_cruzada import generar_folds_walk_forward, validar_walk_forward_multi_fold


def _dataset_sintetico(n: int, seed: int = 0, fraude_solo_antes_de: int | None = None) -> pd.DataFrame:
    """Señal real y separable (mismo patrón que test_calibrador.py). Si
    `fraude_solo_antes_de` se da, el fraude solo ocurre en filas con índice
    menor a ese corte -- para forzar deliberadamente que un fold tardío no
    tenga ningún caso positivo en validación."""
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
    assert tamanos_train == sorted(tamanos_train)  # el train se expande fold a fold, nunca se achica

    for train, val in folds:
        assert train["Time"].max() < val["Time"].min()  # val siempre después de train -- sin fuga de futuro

    # los bloques de validación no se traslapan entre folds consecutivos
    for (_, val_a), (_, val_b) in zip(folds, folds[1:]):
        assert val_a["Time"].max() < val_b["Time"].min()


def test_generar_folds_walk_forward_dataset_muy_chico_lanza_error():
    df = _dataset_sintetico(n=20)
    with pytest.raises(ValueError, match="no alcanza"):
        generar_folds_walk_forward(df, n_folds=10, frac_train_inicial=0.9)


def test_validar_walk_forward_con_senal_estable_da_auc_alto_y_consistente():
    df = _dataset_sintetico(n=6000, seed=1)  # fraude repartido en todo el dataset
    resumen = validar_walk_forward_multi_fold(df, n_folds=5, frac_train_inicial=0.5)

    assert resumen["n_folds_fallidos"] == 0
    assert resumen["n_folds_exitosos"] == 5
    assert resumen["auc_media"] > 0.8
    assert resumen["auc_desviacion_estandar"] < 0.1  # la señal es real y estable en el tiempo, no un accidente de un split


def test_validar_walk_forward_fold_sin_positivos_en_validacion_se_registra_sin_reventar_los_demas():
    # Fraude solo en el primer 70% del dataset -- los folds cuyo bloque de
    # validación cae después de ese corte no tienen ningún caso positivo.
    df = _dataset_sintetico(n=6000, seed=2, fraude_solo_antes_de=4200)
    resumen = validar_walk_forward_multi_fold(df, n_folds=5, frac_train_inicial=0.5)

    # folds 0 y 1 (val termina en 3600 y 4200) sí tienen fraude; 2, 3 y 4 no.
    assert resumen["n_folds_exitosos"] == 2
    assert resumen["n_folds_fallidos"] == 3
    assert {f["fold"] for f in resumen["folds_fallidos"]} == {2, 3, 4}
    assert "auc_media" in resumen  # los folds exitosos sí se agregan


def test_validar_walk_forward_todos_los_folds_fallidos_no_agrega_metricas_inexistentes():
    df = _dataset_sintetico(n=6000, seed=3, fraude_solo_antes_de=1)  # prácticamente sin fraude en validación
    resumen = validar_walk_forward_multi_fold(df, n_folds=5, frac_train_inicial=0.5)

    assert resumen["n_folds_exitosos"] == 0
    assert resumen["n_folds_fallidos"] == 5
    assert "auc_media" not in resumen  # nada que promediar -- no se inventa un número
