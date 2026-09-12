"""Detección de deriva (drift) — responde la pregunta de cuándo hace falta
recalibrar (`CicloDecision.recargar_artefacto()` es el mecanismo; este
módulo es la señal que decide *cuándo* usarlo). Compara la distribución de
una feature en los datos con que se calibró (referencia) contra datos más
recientes (actual), con dos métricas complementarias:

- **PSI (Population Stability Index)**: cuánto se movió la distribución
  completa, en bins — umbrales estándar de la industria (0.1 / 0.25).
- **Test de Kolmogorov-Smirnov**: si la diferencia es estadísticamente
  significativa, con un p-value — más sensible que PSI a cambios de forma
  (no solo de nivel), pero sin la escala interpretable de PSI.

Se usan las dos porque miden cosas ligeramente distintas y ninguna es
estrictamente mejor — PSI es el estándar de la industria para decisiones de
negocio ("recalibrar sí/no"), KS es más sensible como señal temprana.
"""
import numpy as np
from scipy import stats

UMBRAL_PSI_MODERADO = 0.1  # por debajo: sin deriva significativa (estándar de la industria)
UMBRAL_PSI_SIGNIFICATIVO = 0.25  # por encima: recalibrar
UMBRAL_PVALUE_KS = 0.05


def calcular_psi(referencia: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI = Σ (pct_actual - pct_ref) · ln(pct_actual / pct_ref). Los bins
    se fijan por los percentiles de `referencia`, no de `actual` — el PSI
    mide cuánto se movió `actual` respecto a la referencia, no al revés."""
    referencia = np.asarray(referencia, dtype=float)
    actual = np.asarray(actual, dtype=float)

    percentiles = np.linspace(0, 100, n_bins + 1)
    bordes = np.unique(np.percentile(referencia, percentiles))
    if len(bordes) < 2:
        return 0.0  # referencia sin variación -- no se puede medir deriva

    bordes = bordes.copy()
    bordes[0] = -np.inf
    bordes[-1] = np.inf

    conteo_ref, _ = np.histogram(referencia, bins=bordes)
    conteo_actual, _ = np.histogram(actual, bins=bordes)

    pct_ref = conteo_ref / len(referencia)
    pct_actual = conteo_actual / len(actual)

    # Un bin vacío en cualquiera de los dos lados se trata como una fracción mínima (no 0)
    # para evitar log(0)/división por 0 -- práctica estándar del cálculo de PSI.
    epsilon = 1e-4
    pct_ref = np.where(pct_ref == 0, epsilon, pct_ref)
    pct_actual = np.where(pct_actual == 0, epsilon, pct_actual)

    return float(np.sum((pct_actual - pct_ref) * np.log(pct_actual / pct_ref)))


def interpretar_psi(psi: float) -> str:
    if psi < UMBRAL_PSI_MODERADO:
        return "sin_deriva_significativa"
    if psi < UMBRAL_PSI_SIGNIFICATIVO:
        return "deriva_moderada_monitorear"
    return "deriva_significativa_recalibrar"


def evaluar_ks(referencia: np.ndarray, actual: np.ndarray) -> dict:
    estadistico, pvalue = stats.ks_2samp(referencia, actual)
    return {"estadistico": float(estadistico), "pvalue": float(pvalue), "hay_deriva": bool(pvalue < UMBRAL_PVALUE_KS)}


def evaluar_deriva_score(scores_referencia: np.ndarray, scores_actual: np.ndarray, n_bins: int = 10) -> dict:
    """Igual que `evaluar_deriva()`, pero sobre el score de salida del
    modelo -- no una feature de entrada. Complementa, no reemplaza, el
    monitoreo por feature: el score agrega el efecto neto de deriva en
    TODAS las features a través del modelo en un solo número, así que
    puede moverse de forma significativa aunque ninguna feature individual
    cruce sola el umbral de PSI (el caso que el monitoreo por feature, por
    diseño, no puede ver)."""
    psi = calcular_psi(scores_referencia, scores_actual, n_bins=n_bins)
    return {"psi": psi, "psi_interpretacion": interpretar_psi(psi), "ks": evaluar_ks(scores_referencia, scores_actual)}


def evaluar_deriva(df_referencia, df_actual, columnas: list, n_bins: int = 10) -> dict:
    """Corre PSI + KS sobre cada columna dada, devuelve un reporte por
    columna y un resumen de cuántas muestran deriva significativa."""
    reporte = {}
    for col in columnas:
        psi = calcular_psi(df_referencia[col].to_numpy(), df_actual[col].to_numpy(), n_bins=n_bins)
        ks = evaluar_ks(df_referencia[col].to_numpy(), df_actual[col].to_numpy())
        reporte[col] = {"psi": psi, "psi_interpretacion": interpretar_psi(psi), "ks": ks}

    n_con_deriva_psi = sum(1 for r in reporte.values() if r["psi_interpretacion"] == "deriva_significativa_recalibrar")

    return {
        "por_feature": reporte,
        "n_features_con_deriva_psi": n_con_deriva_psi,
        "recomendacion_recalibrar": n_con_deriva_psi > 0,
    }


def ponderar_deriva_por_coeficiente(reporte_deriva: dict, features: list, coeficientes: list) -> dict:
    """Enriquece el reporte de `evaluar_deriva()` con la magnitud del
    coeficiente de cada feature en el artefacto vigente -- ayuda a
    distinguir deriva en una feature que el modelo realmente usa (coeficiente
    grande) de deriva en una que casi no importa (coeficiente cerca de 0).

    **No cambia `recomendacion_recalibrar`** (sigue siendo el estándar de
    industria de PSI > 0.25 por feature, sin alterar) -- introducir un
    umbral nuevo sobre la contribución ponderada sería un parámetro no
    justificado empíricamente (ver `CLAUDE.md`: "ningún parámetro no
    justificado tiene valor por defecto silencioso"). Esto es información
    adicional para que un humano juzgue si la recalibración recomendada la
    empujan features estructuralmente importantes o ruido en features que
    el modelo casi no pesa -- `contribucion_ponderada_features_con_deriva`
    en [0, 1]: qué fracción del peso total del modelo (suma de |coeficiente|)
    está en las features que sí muestran deriva significativa."""
    coef_por_feature = dict(zip(features, coeficientes))
    peso_total = sum(abs(c) for c in coeficientes)

    por_feature = {}
    for nombre, datos in reporte_deriva["por_feature"].items():
        coef = coef_por_feature.get(nombre)
        peso_relativo = (abs(coef) / peso_total) if (coef is not None and peso_total > 0) else None
        por_feature[nombre] = {**datos, "coeficiente": coef, "peso_relativo_en_score": peso_relativo}

    features_con_deriva = [
        nombre for nombre, datos in reporte_deriva["por_feature"].items()
        if datos["psi_interpretacion"] == "deriva_significativa_recalibrar"
    ]
    contribucion = (
        sum(abs(coef_por_feature.get(nombre, 0.0)) for nombre in features_con_deriva) / peso_total
        if peso_total > 0 else 0.0
    )

    return {
        "por_feature": por_feature,
        "n_features_con_deriva_psi": reporte_deriva["n_features_con_deriva_psi"],
        "recomendacion_recalibrar": reporte_deriva["recomendacion_recalibrar"],
        "contribucion_ponderada_features_con_deriva": contribucion,
    }
