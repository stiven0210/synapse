"""Multi-fold walk-forward cross-validation — diagnostic, never
production. `calibrador.split_temporal()` + `calibrar()` produce THE
current artifact with a single train/val/test split (only one artifact
can exist at a time, that doesn't change). But a single split doesn't say
whether the calibration *process* is stable over time, or whether the
reported AUC/threshold depends on which particular cut was used.

Multi-fold walk-forward (expanding window, never random K-fold -- that
would leak the future into the past on data with real temporal order, the
same reason `split_temporal` is already walk-forward) runs calibration
several times over successive cuts and reports the mean/standard deviation
of AUC and threshold across folds.
"""
import numpy as np
import pandas as pd

from src.calibrador import DatasetInvalido, calibrar


def generar_folds_walk_forward(df: pd.DataFrame, n_folds: int, frac_train_inicial: float = 0.5) -> list:
    """Expanding window: fold `i` trains on everything before cut `i` and
    validates on the next block -- never the other way around.
    `frac_train_inicial` is the minimum training size before folds start
    being generated (validating with almost no history makes no sense)."""
    n = len(df)
    inicio_val = int(n * frac_train_inicial)
    tamano_bloque = (n - inicio_val) // n_folds
    if tamano_bloque <= 0:
        raise ValueError(
            f"dataset de {n} filas no alcanza para {n_folds} folds después del "
            f"{frac_train_inicial:.0%} inicial de entrenamiento"
        )

    folds = []
    for i in range(n_folds):
        fin_train = inicio_val + i * tamano_bloque
        fin_val = fin_train + tamano_bloque
        folds.append((df.iloc[:fin_train], df.iloc[fin_train:fin_val]))
    return folds


def validar_walk_forward_multi_fold(df: pd.DataFrame, n_folds: int = 5, frac_train_inicial: float = 0.5) -> dict:
    """Runs `calibrar()` over each walk-forward fold and aggregates the
    metrics. A fold without enough positive cases (`DatasetInvalido` --
    fraud is 0.17% of the real dataset, a small block might have none) is
    recorded as failed instead of crashing the whole validation; that's
    also real information, not an error to hide."""
    folds = generar_folds_walk_forward(df, n_folds=n_folds, frac_train_inicial=frac_train_inicial)

    exitosos, fallidos = [], []
    for i, (train, val) in enumerate(folds):
        try:
            artefacto = calibrar(train, val)
        except DatasetInvalido as e:
            fallidos.append({"fold": i, "error": str(e), "n_train": len(train), "n_val": len(val)})
            continue
        exitosos.append({
            "fold": i,
            "n_train": len(train),
            "n_val": len(val),
            "umbral_decision": artefacto["umbral_decision"],
            **artefacto["metricas_validacion"],
        })

    resumen = {
        "folds_exitosos": exitosos,
        "folds_fallidos": fallidos,
        "n_folds_exitosos": len(exitosos),
        "n_folds_fallidos": len(fallidos),
    }
    if not exitosos:
        return resumen

    aucs = [r["auc"] for r in exitosos]
    umbrales = [r["umbral_decision"] for r in exitosos]
    resumen.update({
        "auc_media": float(np.mean(aucs)),
        "auc_desviacion_estandar": float(np.std(aucs)),
        "umbral_media": float(np.mean(umbrales)),
        "umbral_desviacion_estandar": float(np.std(umbrales)),
    })
    return resumen
