"""Disparador de recalibración — conecta la detección de deriva
(`src/deriva.py`) con el mecanismo de recalibración (`src/calibrador.py` +
`src/puente.py`). Es el job periódico que el plan de trabajo señalaba como
faltante: compara la ventana de referencia (con la que se calibró el
artefacto vigente) contra una ventana reciente de producción, y si hay
deriva significativa en alguna feature O en el score de salida del modelo,
recalibra sobre la ventana reciente y publica el nuevo artefacto.

No decide en tiempo real y no toca el estado recursivo del Ejecutor — un
proceso de decisión en marcha (`CicloDecision`) debe llamar
`recargar_artefacto()` por su cuenta después de ver que este job publicó
una versión nueva; ese mecanismo ya existe y es responsabilidad de
`ciclo.py`, no de este módulo.
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
    """Pondera la deriva por feature y evalúa la deriva del score de
    salida, ambas usando el artefacto VIGENTE (antes de una posible
    recalibración) -- ver `deriva.ponderar_deriva_por_coeficiente` y
    `deriva.evaluar_deriva_score`. Si no hay artefacto vigente legible
    todavía (primera corrida, archivo corrupto), ninguna de las dos se
    puede calcular -- no bloquea la evaluación de deriva por feature, que
    sigue siendo el criterio principal en ese caso."""
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
    """`df_referencia` es la ventana con la que se calibró el artefacto
    vigente; `df_actual` es la ventana reciente de producción a evaluar y,
    si hace falta, la que se usa para recalibrar (partida internamente en
    train/val vía `split_temporal`, igual que en la calibración original).

    Recomienda recalibrar si CUALQUIERA de las dos señales es significativa:
    deriva por feature (`evaluar_deriva`, PSI > 0.25 en alguna) o deriva del
    score de salida (`evaluar_deriva_score`) -- el score agrega el efecto
    neto de deriva en todas las features y puede moverse aunque ninguna
    cruce sola el umbral (ver `docs/PLAN_DE_TRABAJO.md`).

    Si `df_actual` no tiene casos positivos suficientes para recalibrar con
    seguridad (`DatasetInvalido`), no se publica nada — el artefacto vigente
    queda intacto y el error se reporta para revisión humana en vez de
    fallar en silencio o degradar el modelo con una calibración mala."""
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
