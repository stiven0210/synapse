"""Calibrador de árboles (Dominio 2, capa lenta) — entrena el Gradient
Boosting validado en `docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` (secciones
5-7 y 15.1-16: modelo final `HistGradientBoostingClassifier(max_depth=3,
learning_rate=0.05, max_iter=200)`, 6 features, calibración isotónica) sobre
`data/raw/sparkov_2013_2026.csv`, y produce el artefacto de
`artefacto_arboles.py` -- tabla de búsqueda v3 incluida, no los árboles
crudos (ver esa sección 6, tercera ronda: la tabla es exacta por
construcción y más rápida que evaluar árboles en caliente).

No se toca `src/calibrador.py` (Dominio 1) -- se reutiliza sin modificar
`split_temporal` (puramente posicional, no depende de columnas) y
`calcular_features_recursivas_batch`/`EstadoRecursivoGlobal` de
`features_recursivas.py` para `conteo_ventana_global`, que es la MISMA
métrica global de Dominio 1, no una nueva definición.
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
# 6ta feature agregada en la sección 16 del doc (`frecuencia_categoria_expandida`, mejora
# validada en la sección 15.1: AUC-PR 0.914+-0.030 -> 0.947+-0.027 en 5-fold walk-forward).
FEATURES_ARBOLES = [
    "amt", "hora", "conteo_ventana_global", "monto_ewma_cuenta", "huella_categoria_cuenta",
    "frecuencia_categoria_expandida",
]
COLUMNAS_CRUDAS_REQUERIDAS = {"cc_num", "amt", "unix_time", "category", "is_fraud"}

# Configuración ganadora del Paso 4 del afinamiento (docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md),
# actualizada en la sección 21: learning_rate 0.05 -> 0.1 (validado con 5-fold walk-forward,
# AUC-PR 0.947+-0.027 -> 0.957+-0.024, mejora real de +0.0098, mejor o empatado en 4 de 5 folds).
MAX_DEPTH = 3
LEARNING_RATE = 0.1
MAX_ITER = 200
RANDOM_STATE = 42


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """Carga y ordena `sparkov_2013_2026.csv` -- NO calcula todavía las
    features recursivas (a diferencia de `calibrador.cargar_dataset`):
    `huella_categoria_cuenta` necesita `frecuencia_poblacional_categoria`
    aprendida en TRAIN, y TRAIN se define por posición sobre este mismo
    dataframe -- ver `construir_features` y `entrenar_pipeline_completo`."""
    df = pd.read_csv(ruta)
    if df.empty:
        raise DatasetInvalido(f"{ruta} no tiene filas")
    faltantes = COLUMNAS_CRUDAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise DatasetInvalido(f"{ruta} no tiene las columnas requeridas: {faltantes}")
    # kind="stable": mismo motivo que calibrador.cargar_dataset (Dominio 1) -- unix_time con
    # resolución de 1 segundo y alta frecuencia de transacciones produce empates reales.
    return df.sort_values("unix_time", kind="stable").reset_index(drop=True)


def construir_features(df: pd.DataFrame, frecuencia_categoria: dict) -> pd.DataFrame:
    """Agrega las 6 features finales sobre TODO el historial continuo (antes
    de partir en train/val/test -- mismo principio que Dominio 1: el estado
    recursivo no se reinicia en un corte arbitrario). `n_categorias` para
    `frecuencia_categoria_expandida` se deriva de `frecuencia_categoria`
    (aprendida de TRAIN) -- mismo vocabulario de categorías conocidas que ya
    usa el respaldo poblacional de la huella, sin agregar un campo nuevo al
    artefacto."""
    df_global = calcular_features_recursivas_batch(df.rename(columns={"amt": "Amount", "unix_time": "Time"}))
    df = df.copy()
    df["conteo_ventana_global"] = df_global["conteo_ventana_global"].to_numpy()
    df = calcular_features_recursivas_cuenta_batch(df, frecuencia_categoria=frecuencia_categoria)
    df = calcular_frecuencia_categoria_expandida_batch(df, n_categorias=len(frecuencia_categoria))
    return df


def _mejor_umbral_por_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Copia local intencional de `calibrador._mejor_umbral_por_f1` (privada,
    no se importa entre módulos de dominios distintos -- ver docstring del
    módulo). Misma lógica: busca sobre los scores realmente observados, sin
    grilla fija topada."""
    precisiones, recalls, umbrales = precision_recall_curve(y_true, scores)
    if len(umbrales) == 0:
        return 0.5
    precisiones, recalls = precisiones[:-1], recalls[:-1]
    denominador = precisiones + recalls
    f1s = np.divide(2 * precisiones * recalls, denominador, out=np.zeros_like(denominador), where=denominador > 0)
    return float(umbrales[np.argmax(f1s)])


def _extraer_umbrales_por_feature(modelo: HistGradientBoostingClassifier, features: list) -> dict:
    """Recorre los nodos internos de los 175 árboles y agrupa, por índice de
    feature, el conjunto de umbrales reales (`num_threshold`) que el modelo
    aprendió -- son los cortes que definen los bins exactos de la tabla v3
    (ver `docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`, sección 6, tercera
    ronda)."""
    umbrales_por_indice = {i: set() for i in range(len(features))}
    for arboles_iteracion in modelo._predictors:
        for arbol in arboles_iteracion:
            for nodo in arbol.nodes:
                if not nodo["is_leaf"]:
                    umbrales_por_indice[int(nodo["feature_idx"])].add(float(nodo["num_threshold"]))
    return {features[i]: sorted(umbrales_por_indice[i]) for i in range(len(features))}


def _construir_tabla_busqueda(modelo: HistGradientBoostingClassifier, features: list, umbrales_por_feature: dict) -> tuple:
    """Evalúa el modelo UNA VEZ por cada combinación de bin (vectorizado vía
    `predict_proba`, no en Python puro -- esto corre en calibración, no en el
    camino caliente). Cada representante se elige para que
    `bisect.bisect_left(umbrales, representante)` devuelva exactamente el
    índice de bin que le corresponde (verificado con test unitario en
    `tests/test_calibrador_arboles.py`); como ningún árbol distingue dos
    valores dentro del mismo intervalo entre umbrales consecutivos, la tabla
    queda exacta por construcción."""
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
    X_repr = np.stack([g.ravel() for g in grillas], axis=1)  # (tamano_total, n_features), orden C == row-major
    assert X_repr.shape == (tamano_total, len(features))

    raw_scores = modelo.predict_proba(X_repr)[:, 1]
    return forma, raw_scores.tolist()


def _calibrar_detalle(df_train: pd.DataFrame, df_val: pd.DataFrame, frecuencia_categoria: dict) -> tuple:
    """Misma lógica que `calibrar()`, pero además devuelve el modelo
    sklearn ya entrenado -- necesario para la verificación bit-exacta contra
    `predict_proba()` real en `tests/test_ejecutor_arboles.py` y en scripts
    de validación. `calibrar()` es la API pública real (solo el artefacto,
    igual que `calibrador.calibrar` de Dominio 1); esta función privada
    existe solo para no reentrenar dos veces en tests."""
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

    # Isotonic SIEMPRE ajustada en VAL, nunca en TEST (Paso 5 del afinamiento) -- necesario
    # porque los scores crudos están mal calibrados en el rango medio-alto (sample_weight
    # de class_weight="balanced" afecta las probabilidades, no el orden).
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
    """API pública: entrena y devuelve solo el artefacto de
    `artefacto_arboles.py` (igual que `calibrador.calibrar` de Dominio 1).
    `df_train`/`df_val` ya deben traer las 6 features (`construir_features`)."""
    artefacto, _modelo = _calibrar_detalle(df_train, df_val, frecuencia_categoria)
    return artefacto


def entrenar_pipeline_completo(ruta_dataset: Path) -> dict:
    """Orquesta el pipeline completo: carga, split posicional 60/20/20 SOLO
    para aprender `frecuencia_poblacional_categoria` desde TRAIN crudo,
    construye features sobre el historial continuo completo, vuelve a
    partir con los mismos límites, y calibra. Devuelve
    `{"artefacto":..., "modelo":..., "df_train":..., "df_val":..., "df_test":...}`
    -- `df_test` y `modelo` son para scripts de validación/tests, nunca para
    servir decisiones (eso es exclusivamente `EjecutorArboles`)."""
    df = cargar_dataset(ruta_dataset)
    df_train_crudo, _, _ = split_temporal(df)
    frecuencia_categoria = calcular_frecuencia_poblacional_categoria(df_train_crudo)

    df_features = construir_features(df, frecuencia_categoria)
    df_train, df_val, df_test = split_temporal(df_features)

    artefacto, modelo = _calibrar_detalle(df_train, df_val, frecuencia_categoria)
    return {"artefacto": artefacto, "modelo": modelo, "df_train": df_train, "df_val": df_val, "df_test": df_test}
