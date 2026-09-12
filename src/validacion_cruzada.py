"""Validación cruzada walk-forward multi-fold — diagnóstico, nunca
producción. `calibrador.split_temporal()` + `calibrar()` producen EL
artefacto vigente con un único split train/val/test (solo puede existir un
artefacto a la vez, eso no cambia). Pero un único split no dice si el
*proceso* de calibración es estable a través del tiempo, o si el AUC/umbral
reportado depende de qué corte particular se usó.

Walk-forward multi-fold (ventana expansiva, nunca K-fold aleatorio -- eso
filtraría futuro hacia el pasado en datos con orden temporal real, la misma
razón por la que `split_temporal` ya es walk-forward) corre la calibración
varias veces sobre cortes sucesivos y reporta media/desviación estándar del
AUC y del umbral entre folds.
"""
import numpy as np
import pandas as pd

from src.calibrador import DatasetInvalido, calibrar


def generar_folds_walk_forward(df: pd.DataFrame, n_folds: int, frac_train_inicial: float = 0.5) -> list:
    """Ventana expansiva: el fold `i` entrena con todo lo anterior al corte
    `i` y valida con el siguiente bloque -- nunca al revés. `frac_train_inicial`
    es el tamaño mínimo de entrenamiento antes de empezar a generar folds
    (no tiene sentido validar con casi nada de historia)."""
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
    """Corre `calibrar()` sobre cada fold walk-forward y agrega las
    métricas. Un fold sin casos positivos suficientes (`DatasetInvalido` --
    el fraude es 0.17% del dataset real, un bloque chico puede no tener
    ninguno) se registra como fallido en vez de reventar toda la
    validación; eso también es información real, no un error a ocultar."""
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
