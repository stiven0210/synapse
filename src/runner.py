"""Daily runner — the "real deployment, even if small" that was missing:
with no real production traffic, it simulates one by advancing through
the historical dataset in batches, one batch per invocation, with state
persisted across runs (the same as a real service that restarts every
day). This is what's needed for the triage agent's Level 3 question (does
it actually save human effort?) to ever have real data to evaluate against
— before this, no process ran long enough to accumulate any.

Each invocation (`correr_ciclo_diario`):
1. If it's the first time, calibrates an initial (bootstrap) artifact
   using the first `frac_bootstrap` of the dataset and publishes it.
2. Picks the Executor's recursive state back up where the previous run
   left off (never reset -- the same causal semantics as always) and
   processes the next batch of `tamano_lote` rows.
3. Evaluates drift (features + score) between the previous batch and this
   one, and recalibrates if needed (`disparador_recalibracion.py`).
4. If a triage agent is configured, attempts to triage the new
   operational escalations this batch left behind.

Not the hot path -- this is the maintenance job, running once a day (or
whatever cadence is scheduled), never inside `CicloDecision.decidir()`.
"""
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.agente_triage import AgenteTriage, CircuitoAbierto
from src.bitacora_decisiones import filtrar_escalamientos_operativos, leer_bitacora, registrar_decision
from src.calibrador import cargar_dataset, calibrar, split_temporal
from src.ciclo import CicloDecision
from src.disparador_recalibracion import ResultadoDisparador, evaluar_y_recalibrar_si_hace_falta
from src.ejecutor import Ejecutor
from src.features_recursivas import EstadoRecursivoGlobal
from src.limitador_llamadas import PresupuestoAgotado
from src.puente import publicar


def _serializar_estado(estado: EstadoRecursivoGlobal) -> dict:
    return {
        "lambda_ewma": estado.lambda_ewma,
        "ventana_seg": estado.ventana_seg,
        "monto_ewma": estado.monto_ewma,
        "tiempos_ventana": list(estado.tiempos_ventana),
        "ultimo_tiempo_visto": estado.ultimo_tiempo_visto,
    }


def _deserializar_estado(data: dict) -> EstadoRecursivoGlobal:
    estado = EstadoRecursivoGlobal(
        lambda_ewma=data["lambda_ewma"],
        ventana_seg=data["ventana_seg"],
        monto_ewma=data["monto_ewma"],
        tiempos_ventana=deque(data["tiempos_ventana"]),
    )
    estado.ultimo_tiempo_visto = data["ultimo_tiempo_visto"]  # init=False -- assigned after construction
    return estado


def guardar_estado_runner(indice_procesado: int, ejecutor: Ejecutor, ruta: Path) -> None:
    """Atomic write (temp file + replace), same pattern as
    `puente.publicar()` -- a run interrupted mid-write must not leave the
    state corrupted."""
    data = {"indice_procesado": indice_procesado, "estado_ejecutor": _serializar_estado(ejecutor.estado)}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta_temporal = ruta.with_name(ruta.name + ".tmp")
    ruta_temporal.write_text(json.dumps(data, indent=2), encoding="utf-8")
    ruta_temporal.replace(ruta)


def cargar_estado_runner(ruta: Path) -> dict | None:
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


@dataclass
class ResultadoCorrida:
    agotado: bool
    filas_procesadas: int
    indice_procesado: int
    resultado_disparador: ResultadoDisparador | None
    n_triages_intentados: int
    n_triages_exitosos: int
    n_triages_fallidos: int


def correr_ciclo_diario(
    ruta_dataset: Path,
    ruta_artefacto: Path,
    ruta_bitacora: Path,
    ruta_estado_runner: Path,
    tamano_lote: int = 5000,
    frac_bootstrap: float = 0.3,
    agente_triage: AgenteTriage | None = None,
    contexto_triage: Callable[[], dict] | dict | None = None,
) -> ResultadoCorrida:
    df = cargar_dataset(ruta_dataset)
    estado_runner = cargar_estado_runner(ruta_estado_runner)

    if estado_runner is None:
        # First run: bootstrap -- calibrates on the first frac_bootstrap
        # of the dataset (never all of it, to leave real history that
        # "arrives" in later runs) and publishes the initial artifact.
        fin_bootstrap = int(len(df) * frac_bootstrap)
        train, val, _ = split_temporal(df.iloc[:fin_bootstrap])
        artefacto = calibrar(train, val)
        publicar(artefacto, ruta_artefacto)
        ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)

        # Feeds the recursive state with the calibration history, without
        # logging it (it's history, not a new decision) -- same principle
        # as scripts/validacion_end_to_end.py: state is never reset at an
        # arbitrary cut. Processing "what comes in" starts AFTER the
        # bootstrap window, not over it.
        columnas_bootstrap = artefacto["features"] + ["Time"]
        for fila in df.iloc[:fin_bootstrap][columnas_bootstrap].to_dict("records"):
            ciclo.decidir(fila)
        indice_procesado = fin_bootstrap
    else:
        ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)
        estado_restaurado = _deserializar_estado(estado_runner["estado_ejecutor"])
        # There's no public way to inject a restored recursive state when
        # constructing CicloDecision -- the Executor is rebuilt directly, the same
        # pattern CicloDecision.recargar_artefacto() already uses internally
        # (`Ejecutor(artefacto=..., estado=self._ejecutor.estado)`). This is preferred
        # over touching ciclo.py (already through 2 audit rounds) for a need that
        # only this runner has.
        ciclo._ejecutor = Ejecutor(artefacto=ciclo._ejecutor.artefacto, estado=estado_restaurado)
        indice_procesado = estado_runner["indice_procesado"]

    if indice_procesado >= len(df):
        return ResultadoCorrida(
            agotado=True, filas_procesadas=0, indice_procesado=indice_procesado,
            resultado_disparador=None, n_triages_intentados=0, n_triages_exitosos=0, n_triages_fallidos=0,
        )

    fin_lote = min(indice_procesado + tamano_lote, len(df))
    lote = df.iloc[indice_procesado:fin_lote]
    columnas = ciclo._ejecutor.artefacto["features"] + ["Time"]

    # Critical audit finding: if an earlier run is interrupted (crash,
    # Task Scheduler killing the process, power loss) AFTER logging a
    # decision but BEFORE persisting the new index, the next run used to
    # resume from the old index and reprocess -- and re-log -- those same
    # rows, duplicating them in the log.
    # There's no way to make a write atomic across two different files
    # (log + state) without a real transactional log, so the damage is
    # bounded to a minimum: state is saved row by row (never once per full
    # batch), and before logging each row it's checked whether it was
    # already logged by an earlier interrupted attempt -- if so, it still
    # goes through `decidir()` again (the restored recursive state is the
    # one from BEFORE that row, it has to be re-derived to not lose causal
    # continuity) but it's not written to the log again.
    indices_ya_registrados = {e["indice_fila"] for e in leer_bitacora(ruta_bitacora) if e.get("indice_fila") is not None}

    inicio_corrida = datetime.now(timezone.utc).isoformat()
    for posicion, fila in enumerate(lote[columnas].to_dict("records")):
        indice_fila_actual = indice_procesado + posicion
        decision = ciclo.decidir(fila)
        if indice_fila_actual not in indices_ya_registrados:
            registrar_decision(decision, fila, ruta_bitacora, indice_fila=indice_fila_actual)
        guardar_estado_runner(indice_fila_actual + 1, ciclo._ejecutor, ruta_estado_runner)

    resultado_disparador = None
    if indice_procesado > 0:
        # Reference = the batch immediately before this one (the most "recent"
        # one already known); current = what was just processed.
        inicio_referencia = max(0, indice_procesado - tamano_lote)
        referencia = df.iloc[inicio_referencia:indice_procesado]
        resultado_disparador = evaluar_y_recalibrar_si_hace_falta(
            df_referencia=referencia, df_actual=lote,
            columnas_deriva=ciclo._ejecutor.artefacto["features"], ruta_artefacto=ruta_artefacto,
        )
        if resultado_disparador.se_recalibro:
            ciclo.recargar_artefacto()

    n_intentados, n_exitosos, n_fallidos = 0, 0, 0
    if agente_triage is not None:
        entradas_nuevas = [e for e in leer_bitacora(ruta_bitacora) if e["timestamp"] >= inicio_corrida]
        for entrada in filtrar_escalamientos_operativos(entradas_nuevas):
            contexto = contexto_triage() if callable(contexto_triage) else (contexto_triage or {})
            n_intentados += 1
            try:
                resultado = agente_triage.triar(entrada, contexto)
                if not resultado.descartado:
                    n_exitosos += 1
            except (CircuitoAbierto, PresupuestoAgotado):
                n_fallidos += 1  # falls back to the flat report -- the escalation is already in the log
            except Exception:
                # Audit finding: only the 2 types above used to be caught --
                # a real network/API error (timeout, 5xx) blows up uncaught and
                # takes down the ENTIRE job, even after real decisions and state
                # were already saved. Triage is advisory: no failure of its own
                # should take down a job that already did its main work.
                n_fallidos += 1

    return ResultadoCorrida(
        agotado=False, filas_procesadas=len(lote), indice_procesado=fin_lote,
        resultado_disparador=resultado_disparador,
        n_triages_intentados=n_intentados, n_triages_exitosos=n_exitosos, n_triages_fallidos=n_fallidos,
    )
