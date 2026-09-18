"""Tree calibrator (Domain 2, slow layer) — trains the Gradient Boosting
model validated in `docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` (sections
5-7 and 15.1-16: final model `HistGradientBoostingClassifier(max_depth=3,
learning_rate=0.05, max_iter=200)`, 6 features, isotonic calibration) on
`data/raw/sparkov_2013_2026.csv`, and produces the artifact from
`artefacto_arboles.py` -- the v3 lookup table included, not the raw trees
(see that section 6, third round: the table is exact by construction and
faster than evaluating trees on the hot path).

`src/calibrador.py` (Domain 1) is left untouched -- reused as-is:
`split_temporal` (purely positional, doesn't depend on columns) and
`calcular_features_recursivas_batch`/`EstadoRecursivoGlobal` from
`features_recursivas.py` for `conteo_ventana_global`, which is the SAME
global metric from Domain 1, not a new definition.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from src.calibrador import DatasetInvalido, split_temporal
from src.features_recursivas import calcular_features_recursivas_batch
from src.features_recursivas_cuenta import (
    calcular_features_recursivas_cuenta_batch,
    calcular_frecuencia_categoria_expandida_batch,
    calcular_frecuencia_poblacional_categoria,
)

VERSION_ARTEFACTO = 1
# 6th feature added in section 16 of the doc (`frecuencia_categoria_expandida`, improvement
# validated in section 15.1: AUC-PR 0.914+-0.030 -> 0.947+-0.027 in 5-fold walk-forward).
FEATURES_ARBOLES = [
    "amt", "hora", "conteo_ventana_global", "monto_ewma_cuenta", "huella_categoria_cuenta",
    "frecuencia_categoria_expandida",
]
# Subset to pass to `deriva.py::evaluar_deriva()` -- excludes `frecuencia_categoria_expandida`
# on purpose (section 19 of the doc: it's non-stationary by design, triggering recalibration on
# 85% of year-over-year transitions vs. 23% for the rest -- a constant false alarm, not real
# deterioration signal). No dedicated recalibration trigger for Domain 2 yet -- this
# constant leaves the right set already decided for when one gets built.
FEATURES_MONITOREO_DERIVA = [f for f in FEATURES_ARBOLES if f != "frecuencia_categoria_expandida"]
COLUMNAS_CRUDAS_REQUERIDAS = {"cc_num", "amt", "unix_time", "category", "is_fraud"}

# Winning configuration from tuning Step 4 (docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md),
# updated in section 21: learning_rate 0.05 -> 0.1 (validated with 5-fold walk-forward,
# AUC-PR 0.947+-0.027 -> 0.957+-0.024, a real +0.0098 improvement, better or tied in 4 of 5 folds).
MAX_DEPTH = 3
LEARNING_RATE = 0.1
MAX_ITER = 200
RANDOM_STATE = 42


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """Loads and sorts `sparkov_2013_2026.csv` -- does NOT compute the
    recursive features yet (unlike `calibrador.cargar_dataset`):
    `huella_categoria_cuenta` needs `frecuencia_poblacional_categoria`
    learned on TRAIN, and TRAIN is defined positionally over this same
    dataframe -- see `construir_features` and `entrenar_pipeline_completo`."""
    df = pd.read_csv(ruta)
    if df.empty:
        raise DatasetInvalido(f"{ruta} no tiene filas")
    faltantes = COLUMNAS_CRUDAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise DatasetInvalido(f"{ruta} no tiene las columnas requeridas: {faltantes}")
    # kind="stable": same reason as calibrador.cargar_dataset (Domain 1) -- unix_time with
    # 1-second resolution and high transaction frequency produces real ties.
    return df.sort_values("unix_time", kind="stable").reset_index(drop=True)


def construir_features(df: pd.DataFrame, frecuencia_categoria: dict) -> pd.DataFrame:
    """Adds the 6 final features over the ENTIRE continuous history (before
    splitting into train/val/test -- same principle as Domain 1: recursive
    state isn't reset at an arbitrary cut). `n_categorias` for
    `frecuencia_categoria_expandida` is derived from `frecuencia_categoria`
    (learned from TRAIN) -- the same known-category vocabulary the
    footprint's population fallback already uses, without adding a new
    field to the artifact."""
    df_global = calcular_features_recursivas_batch(df.rename(columns={"amt": "Amount", "unix_time": "Time"}))
    df = df.copy()
    df["conteo_ventana_global"] = df_global["conteo_ventana_global"].to_numpy()
    df = calcular_features_recursivas_cuenta_batch(df, frecuencia_categoria=frecuencia_categoria)
    df = calcular_frecuencia_categoria_expandida_batch(df, n_categorias=len(frecuencia_categoria))
    return df


def _mejor_umbral_por_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Intentional local copy of `calibrador._mejor_umbral_por_f1` (private,
    not imported across different domain modules -- see the module
    docstring). Same logic: searches over the actually observed scores, no
    capped fixed grid."""
    precisiones, recalls, umbrales = precision_recall_curve(y_true, scores)
    if len(umbrales) == 0:
        return 0.5
    precisiones, recalls = precisiones[:-1], recalls[:-1]
    denominador = precisiones + recalls
    f1s = np.divide(2 * precisiones * recalls, denominador, out=np.zeros_like(denominador), where=denominador > 0)
    return float(umbrales[np.argmax(f1s)])


def _extraer_umbrales_por_feature(modelo: HistGradientBoostingClassifier, features: list) -> dict:
    """Walks the internal nodes of the 175 trees and groups, by feature
    index, the set of real thresholds (`num_threshold`) the model learned
    -- these are the cut points that define the v3 table's exact bins (see
    `docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`, section 6, third
    round)."""
    umbrales_por_indice = {i: set() for i in range(len(features))}
    for arboles_iteracion in modelo._predictors:
        for arbol in arboles_iteracion:
            for nodo in arbol.nodes:
                if not nodo["is_leaf"]:
                    umbrales_por_indice[int(nodo["feature_idx"])].add(float(nodo["num_threshold"]))
    return {features[i]: sorted(umbrales_por_indice[i]) for i in range(len(features))}


def _construir_tabla_busqueda(modelo: HistGradientBoostingClassifier, features: list, umbrales_por_feature: dict) -> tuple:
    """Evaluates the model ONCE per bin combination (vectorized via
    `predict_proba`, not in pure Python -- this runs at calibration time,
    not on the hot path). Each representative is chosen so that
    `bisect.bisect_left(umbrales, representante)` returns exactly the bin
    index it belongs to (verified with a unit test in
    `tests/test_calibrador_arboles.py`); since no tree distinguishes
    between two values within the same interval between consecutive
    thresholds, the table ends up exact by construction."""
    representantes_por_feature = []
    for f in features:
        umbrales = umbrales_por_feature[f]
        if umbrales:
            representantes_por_feature.append(list(umbrales) + [umbrales[-1] + 1.0])
        else:
            representantes_por_feature.append([0.0])

    forma = [len(r) for r in representantes_por_feature]
    tamano_total = 1
    for n in forma:
        tamano_total *= n

    grillas = np.meshgrid(*[np.array(r) for r in representantes_por_feature], indexing="ij")
    X_repr = np.stack([g.ravel() for g in grillas], axis=1)  # (tamano_total, n_features), C order == row-major
    assert X_repr.shape == (tamano_total, len(features))

    raw_scores = modelo.predict_proba(X_repr)[:, 1]
    return forma, raw_scores.tolist()


def _calibrar_detalle(df_train: pd.DataFrame, df_val: pd.DataFrame, frecuencia_categoria: dict) -> tuple:
    """Same logic as `calibrar()`, but also returns the already-trained
    sklearn model -- needed for the bit-exact verification against the real
    `predict_proba()` in `tests/test_ejecutor_arboles.py` and validation
    scripts. `calibrar()` is the real public API (artifact only, same as
    Domain 1's `calibrador.calibrar`); this private function exists only
    so tests don't have to retrain twice."""
    if df_train["is_fraud"].sum() == 0:
        raise DatasetInvalido("df_train no tiene ningún caso positivo -- no hay nada que aprender")
    if df_val["is_fraud"].sum() == 0:
        raise DatasetInvalido("df_val no tiene ningún caso positivo -- el umbral por F1 no se puede calibrar razonablemente")

    X_train = df_train[FEATURES_ARBOLES].to_numpy(dtype=float)
    y_train = df_train["is_fraud"].to_numpy()
    X_val = df_val[FEATURES_ARBOLES].to_numpy(dtype=float)
    y_val = df_val["is_fraud"].to_numpy()

    modelo = HistGradientBoostingClassifier(
        max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE, max_iter=MAX_ITER,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    modelo.fit(X_train, y_train)

    scores_val_crudos = modelo.predict_proba(X_val)[:, 1]

    # Isotonic is ALWAYS fit on VAL, never on TEST (tuning Step 5) -- needed
    # because the raw scores are poorly calibrated in the mid-to-high range (the
    # sample_weight from class_weight="balanced" affects probabilities, not the ranking).
    isotonica = IsotonicRegression(out_of_bounds="clip")
    scores_val_calibrados = isotonica.fit_transform(scores_val_crudos, y_val)

    umbral = _mejor_umbral_por_f1(y_val, scores_val_calibrados)
    y_pred_val = (scores_val_calibrados >= umbral).astype(int)

    metricas = {
        "precision_val": float(precision_score(y_val, y_pred_val, zero_division=0)),
        "recall_val": float(recall_score(y_val, y_pred_val, zero_division=0)),
        "f1_val": float(f1_score(y_val, y_pred_val, zero_division=0)),
        "auc_roc_val": float(roc_auc_score(y_val, scores_val_calibrados)),
        "auc_pr_val": float(average_precision_score(y_val, scores_val_calibrados)),
        "n_transacciones_validacion": int(len(y_val)),
    }

    umbrales_por_feature = _extraer_umbrales_por_feature(modelo, FEATURES_ARBOLES)
    forma, tabla_plana = _construir_tabla_busqueda(modelo, FEATURES_ARBOLES, umbrales_por_feature)

    return {
        "version": VERSION_ARTEFACTO,
        "fecha_calibracion": datetime.now(timezone.utc).isoformat(),
        "modelo": "gradient_boosting",
        "features": FEATURES_ARBOLES,
        "umbrales_por_feature": umbrales_por_feature,
        "tabla_busqueda_forma": forma,
        "tabla_busqueda_plana": tabla_plana,
        "calibracion_isotonica": {"x": isotonica.X_thresholds_.tolist(), "y": isotonica.y_thresholds_.tolist()},
        "umbral_decision": umbral,
        "metricas_validacion": metricas,
        "frecuencia_poblacional_categoria": frecuencia_categoria,
    }, modelo


def calibrar(df_train: pd.DataFrame, df_val: pd.DataFrame, frecuencia_categoria: dict) -> dict:
    """Public API: trains and returns only the artifact from
    `artefacto_arboles.py` (same as Domain 1's `calibrador.calibrar`).
    `df_train`/`df_val` must already carry the 6 features
    (`construir_features`)."""
    artefacto, _modelo = _calibrar_detalle(df_train, df_val, frecuencia_categoria)
    return artefacto


def entrenar_pipeline_completo(ruta_dataset: Path) -> dict:
    """Orchestrates the full pipeline: loads, does a positional 60/20/20
    split ONLY to learn `frecuencia_poblacional_categoria` from raw TRAIN,
    builds features over the entire continuous history, splits again with
    the same boundaries, and calibrates. Returns
    `{"artefacto":..., "modelo":..., "df_train":..., "df_val":..., "df_test":...}`
    -- `df_test` and `modelo` are for validation scripts/tests, never for
    serving decisions (that's exclusively `EjecutorArboles`)."""
    df = cargar_dataset(ruta_dataset)
    df_train_crudo, _, _ = split_temporal(df)
    frecuencia_categoria = calcular_frecuencia_poblacional_categoria(df_train_crudo)

    df_features = construir_features(df, frecuencia_categoria)
    df_train, df_val, df_test = split_temporal(df_features)

    artefacto, modelo = _calibrar_detalle(df_train, df_val, frecuencia_categoria)
    return {"artefacto": artefacto, "modelo": modelo, "df_train": df_train, "df_val": df_val, "df_test": df_test}
