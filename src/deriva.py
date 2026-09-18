"""Drift detection — answers the question of when recalibration is needed
(`CicloDecision.recargar_artefacto()` is the mechanism; this module is the
signal that decides *when* to use it). Compares a feature's distribution
in the data it was calibrated on (reference) against more recent data
(current), with two complementary metrics:

- **PSI (Population Stability Index)**: how much the whole distribution
  moved, in bins — standard industry thresholds (0.1 / 0.25).
- **Kolmogorov-Smirnov test**: whether the difference is statistically
  significant, with a p-value — more sensitive than PSI to shape changes
  (not just level), but without PSI's interpretable scale.

Both are used because they measure slightly different things and neither
is strictly better — PSI is the industry standard for business decisions
("recalibrate yes/no"), KS is more sensitive as an early signal.
"""
import numpy as np
from scipy import stats

UMBRAL_PSI_MODERADO = 0.1  # below: no significant drift (industry standard)
UMBRAL_PSI_SIGNIFICATIVO = 0.25  # above: recalibrate
UMBRAL_PVALUE_KS = 0.05


def calcular_psi(referencia: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI = Σ (pct_actual - pct_ref) · ln(pct_actual / pct_ref). Bins are
    fixed by `referencia`'s percentiles, not `actual`'s — PSI measures how
    much `actual` moved relative to the reference, not the other way
    around."""
    referencia = np.asarray(referencia, dtype=float)
    actual = np.asarray(actual, dtype=float)

    percentiles = np.linspace(0, 100, n_bins + 1)
    bordes = np.unique(np.percentile(referencia, percentiles))
    if len(bordes) < 2:
        return 0.0  # reference has no variation -- drift can't be measured

    bordes = bordes.copy()
    bordes[0] = -np.inf
    bordes[-1] = np.inf

    conteo_ref, _ = np.histogram(referencia, bins=bordes)
    conteo_actual, _ = np.histogram(actual, bins=bordes)

    pct_ref = conteo_ref / len(referencia)
    pct_actual = conteo_actual / len(actual)

    # An empty bin on either side is treated as a minimal fraction (not 0)
    # to avoid log(0)/division by 0 -- standard practice for PSI calculation.
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
    """Same as `evaluar_deriva()`, but on the model's output score -- not
    an input feature. Complements, doesn't replace, per-feature
    monitoring: the score aggregates the net drift effect across ALL
    features through the model into a single number, so it can move
    significantly even if no individual feature alone crosses the PSI
    threshold (the case per-feature monitoring, by design, can't see)."""
    psi = calcular_psi(scores_referencia, scores_actual, n_bins=n_bins)
    return {"psi": psi, "psi_interpretacion": interpretar_psi(psi), "ks": evaluar_ks(scores_referencia, scores_actual)}


def evaluar_deriva(df_referencia, df_actual, columnas: list, n_bins: int = 10) -> dict:
    """Runs PSI + KS on each given column, returns a per-column report and
    a summary of how many show significant drift."""
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
    """Enriches `evaluar_deriva()`'s report with the magnitude of each
    feature's coefficient in the current artifact -- helps distinguish
    drift in a feature the model actually uses (large coefficient) from
    drift in one that barely matters (coefficient near 0).

    **Doesn't change `recomendacion_recalibrar`** (still the industry
    standard of PSI > 0.25 per feature, unaltered) -- introducing a new
    threshold on the weighted contribution would be an empirically
    unjustified parameter (see `CLAUDE.md`: "no unjustified parameter gets
    a silent default value"). This is additional information for a human
    to judge whether the recommended recalibration is being driven by
    structurally important features or noise in features the model barely
    weighs -- `contribucion_ponderada_features_con_deriva` in [0, 1]: what
    fraction of the model's total weight (sum of |coefficient|) sits in
    the features that do show significant drift."""
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
