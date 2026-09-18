"""Recalibration trigger — connects drift detection (`src/deriva.py`) with
the recalibration mechanism (`src/calibrador.py` + `src/puente.py`). This
is the periodic job the work plan flagged as missing: compares the
reference window (the one the current artifact was calibrated on) against
a recent production window, and if there's significant drift in any
feature OR in the model's output score, recalibrates over the recent
window and publishes the new artifact.

Doesn't decide in real time and doesn't touch the Executor's recursive
state — a running decision process (`CicloDecision`) must call
`recargar_artefacto()` on its own after seeing this job publish a new
version; that mechanism already exists and is `ciclo.py`'s
responsibility, not this module's.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.artefacto import ArtefactoInvalido, calcular_scores
from src.calibrador import DatasetInvalido, calibrar, split_temporal
from src.deriva import evaluar_deriva, evaluar_deriva_score, ponderar_deriva_por_coeficiente
from src.puente import leer_vigente, publicar


@dataclass
class ResultadoDisparador:
    recomendacion_recalibrar: bool
    n_features_con_deriva: int
    contribucion_ponderada_features_con_deriva: float | None
    deriva_score_psi: float | None
    deriva_score_interpretacion: str | None
    se_recalibro: bool
    version_artefacto_nueva: int | None
    error: str | None


@dataclass
class _AnalisisConArtefactoVigente:
    contribucion_ponderada: float | None
    deriva_score_psi: float | None
    deriva_score_interpretacion: str | None
    recomienda_por_score: bool


def _analizar_con_artefacto_vigente(reporte_features: dict, df_referencia: pd.DataFrame, df_actual: pd.DataFrame, ruta_artefacto: Path) -> _AnalisisConArtefactoVigente:
    """Weighs the per-feature drift and evaluates the output score's
    drift, both using the CURRENT artifact (before any possible
    recalibration) -- see `deriva.ponderar_deriva_por_coeficiente` and
    `deriva.evaluar_deriva_score`. If there's no readable current artifact
    yet (first run, corrupt file), neither can be computed -- this doesn't
    block the per-feature drift evaluation, which remains the main
    criterion in that case."""
    try:
        artefacto_vigente = leer_vigente(ruta_artefacto)
    except (FileNotFoundError, ArtefactoInvalido):
        return _AnalisisConArtefactoVigente(None, None, None, False)

    contribucion_ponderada = ponderar_deriva_por_coeficiente(
        reporte_features, features=artefacto_vigente["features"], coeficientes=artefacto_vigente["coeficientes"]
    )["contribucion_ponderada_features_con_deriva"]

    scores_referencia = calcular_scores(artefacto_vigente, df_referencia)
    scores_actual = calcular_scores(artefacto_vigente, df_actual)
    reporte_score = evaluar_deriva_score(scores_referencia, scores_actual)

    return _AnalisisConArtefactoVigente(
        contribucion_ponderada=contribucion_ponderada,
        deriva_score_psi=reporte_score["psi"],
        deriva_score_interpretacion=reporte_score["psi_interpretacion"],
        recomienda_por_score=reporte_score["psi_interpretacion"] == "deriva_significativa_recalibrar",
    )


def evaluar_y_recalibrar_si_hace_falta(
    df_referencia: pd.DataFrame,
    df_actual: pd.DataFrame,
    columnas_deriva: list,
    ruta_artefacto: Path,
) -> ResultadoDisparador:
    """`df_referencia` is the window the current artifact was calibrated
    on; `df_actual` is the recent production window to evaluate and, if
    needed, the one used to recalibrate (internally split into train/val
    via `split_temporal`, same as in the original calibration).

    Recommends recalibrating if EITHER signal is significant: per-feature
    drift (`evaluar_deriva`, PSI > 0.25 on any one) or output score drift
    (`evaluar_deriva_score`) -- the score aggregates the net drift effect
    across all features and can move even if none crosses the threshold
    alone (see `docs/PLAN_DE_TRABAJO.md`).

    If `df_actual` doesn't have enough positive cases to recalibrate
    safely (`DatasetInvalido`), nothing is published — the current
    artifact stays intact and the error is reported for human review
    instead of failing silently or degrading the model with a bad
    calibration."""
    reporte_features = evaluar_deriva(df_referencia, df_actual, columnas=columnas_deriva)
    analisis = _analizar_con_artefacto_vigente(reporte_features, df_referencia, df_actual, ruta_artefacto)
    recomienda_recalibrar = reporte_features["recomendacion_recalibrar"] or analisis.recomienda_por_score

    if not recomienda_recalibrar:
        return ResultadoDisparador(
            recomendacion_recalibrar=False,
            n_features_con_deriva=reporte_features["n_features_con_deriva_psi"],
            contribucion_ponderada_features_con_deriva=analisis.contribucion_ponderada,
            deriva_score_psi=analisis.deriva_score_psi,
            deriva_score_interpretacion=analisis.deriva_score_interpretacion,
            se_recalibro=False,
            version_artefacto_nueva=None,
            error=None,
        )

    try:
        train, val, _ = split_temporal(df_actual)
        artefacto_nuevo = calibrar(train, val)
        publicar(artefacto_nuevo, ruta_artefacto)
    except DatasetInvalido as e:
        return ResultadoDisparador(
            recomendacion_recalibrar=True,
            n_features_con_deriva=reporte_features["n_features_con_deriva_psi"],
            contribucion_ponderada_features_con_deriva=analisis.contribucion_ponderada,
            deriva_score_psi=analisis.deriva_score_psi,
            deriva_score_interpretacion=analisis.deriva_score_interpretacion,
            se_recalibro=False,
            version_artefacto_nueva=None,
            error=str(e),
        )

    return ResultadoDisparador(
        recomendacion_recalibrar=True,
        n_features_con_deriva=reporte_features["n_features_con_deriva_psi"],
        contribucion_ponderada_features_con_deriva=analisis.contribucion_ponderada,
        deriva_score_psi=analisis.deriva_score_psi,
        deriva_score_interpretacion=analisis.deriva_score_interpretacion,
        se_recalibro=True,
        version_artefacto_nueva=artefacto_nuevo["version"],
        error=None,
    )
