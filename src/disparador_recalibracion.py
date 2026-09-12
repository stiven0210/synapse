"""Disparador de recalibración — conecta la detección de deriva
(`src/deriva.py`) con el mecanismo de recalibración (`src/calibrador.py` +
`src/puente.py`). Es el job periódico que el plan de trabajo señalaba como
faltante: compara la ventana de referencia (con la que se calibró el
artefacto vigente) contra una ventana reciente de producción, y si hay
deriva significativa en alguna feature (PSI > 0.25), recalibra sobre la
ventana reciente y publica el nuevo artefacto.

No decide en tiempo real y no toca el estado recursivo del Ejecutor — un
proceso de decisión en marcha (`CicloDecision`) debe llamar
`recargar_artefacto()` por su cuenta después de ver que este job publicó
una versión nueva; ese mecanismo ya existe y es responsabilidad de
`ciclo.py`, no de este módulo.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.calibrador import DatasetInvalido, calibrar, split_temporal
from src.deriva import evaluar_deriva
from src.puente import publicar


@dataclass
class ResultadoDisparador:
    recomendacion_recalibrar: bool
    n_features_con_deriva: int
    se_recalibro: bool
    version_artefacto_nueva: int | None
    error: str | None


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

    Si `df_actual` no tiene casos positivos suficientes para recalibrar con
    seguridad (`DatasetInvalido`), no se publica nada — el artefacto vigente
    queda intacto y el error se reporta para revisión humana en vez de
    fallar en silencio o degradar el modelo con una calibración mala."""
    reporte = evaluar_deriva(df_referencia, df_actual, columnas=columnas_deriva)

    if not reporte["recomendacion_recalibrar"]:
        return ResultadoDisparador(
            recomendacion_recalibrar=False,
            n_features_con_deriva=reporte["n_features_con_deriva_psi"],
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
            n_features_con_deriva=reporte["n_features_con_deriva_psi"],
            se_recalibro=False,
            version_artefacto_nueva=None,
            error=str(e),
        )

    return ResultadoDisparador(
        recomendacion_recalibrar=True,
        n_features_con_deriva=reporte["n_features_con_deriva_psi"],
        se_recalibro=True,
        version_artefacto_nueva=artefacto_nuevo["version"],
        error=None,
    )
