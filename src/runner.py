"""Runner diario — el "despliegue real, aunque sea chico" que faltaba: sin
tráfico real de producción, simula uno real avanzando por el dataset
histórico en lotes, un lote por invocación, con estado persistente entre
corridas (igual que un servicio real que se reinicia todos los días). Es lo
que hace falta para que el Nivel 3 del agente de triage (¿de verdad ahorra
esfuerzo humano?) tenga alguna vez datos reales que evaluar — antes de
esto, no había ningún proceso corriendo el tiempo suficiente para
acumularlos.

Cada invocación (`correr_ciclo_diario`):
1. Si es la primera vez, calibra un artefacto inicial (bootstrap) con el
   primer `frac_bootstrap` del dataset y lo publica.
2. Retoma el estado recursivo del Ejecutor donde quedó la corrida anterior
   (nunca se reinicia -- misma semántica causal de siempre) y procesa el
   siguiente lote de `tamano_lote` filas.
3. Evalúa deriva (features + score) entre el lote anterior y este, y
   recalibra si hace falta (`disparador_recalibracion.py`).
4. Si hay un agente de triage configurado, intenta triar los
   escalamientos operativos nuevos que dejó este lote.

No es el camino caliente -- es el job de mantenimiento, corre una vez al
día (o con la cadencia que se programe), nunca dentro de `CicloDecision.decidir()`.
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
    estado.ultimo_tiempo_visto = data["ultimo_tiempo_visto"]  # init=False -- se asigna después de construir
    return estado


def guardar_estado_runner(indice_procesado: int, ejecutor: Ejecutor, ruta: Path) -> None:
    """Escritura atómica (archivo temporal + reemplazo), mismo patrón que
    `puente.publicar()` -- una corrida interrumpida a mitad de escritura no
    debe dejar el estado corrupto."""
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
        # Primera corrida: bootstrap -- calibra con el primer frac_bootstrap
        # del dataset (nunca con todo, para dejar historia real que
        # "llegue" en corridas siguientes) y publica el artefacto inicial.
        fin_bootstrap = int(len(df) * frac_bootstrap)
        train, val, _ = split_temporal(df.iloc[:fin_bootstrap])
        artefacto = calibrar(train, val)
        publicar(artefacto, ruta_artefacto)
        ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)

        # Alimenta el estado recursivo con la historia de calibración, sin
        # registrarla en la bitácora (es historia, no una decisión nueva) --
        # mismo principio que scripts/validacion_end_to_end.py: el estado
        # nunca se reinicia en un corte arbitrario. El procesamiento de
        # "lo que llega" arranca DESPUÉS de la ventana de bootstrap, no
        # sobre ella.
        columnas_bootstrap = artefacto["features"] + ["Time"]
        for fila in df.iloc[:fin_bootstrap][columnas_bootstrap].to_dict("records"):
            ciclo.decidir(fila)
        indice_procesado = fin_bootstrap
    else:
        ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)
        estado_restaurado = _deserializar_estado(estado_runner["estado_ejecutor"])
        # No hay una vía pública para inyectar un estado recursivo restaurado al
        # construir CicloDecision -- se reconstruye el Ejecutor directamente, mismo
        # patrón que ya usa CicloDecision.recargar_artefacto() internamente
        # (`Ejecutor(artefacto=..., estado=self._ejecutor.estado)`). Se prefiere esto
        # a tocar ciclo.py (ya pasó 2 rondas de auditoría) por una necesidad que solo
        # tiene este runner.
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

    # Hallazgo crítico de auditoría: si una corrida anterior se interrumpe
    # (crash, Task Scheduler matando el proceso, corte de luz) DESPUÉS de
    # registrar una decisión pero ANTES de persistir el nuevo índice, la
    # siguiente corrida retomaba desde el índice viejo y reprocesaba -- y
    # re-registraba -- esas mismas filas, duplicándolas en la bitácora.
    # No hay forma de hacer atómica una escritura a través de dos archivos
    # distintos (bitácora + estado) sin un log transaccional real, así que
    # se acota el daño al mínimo posible: se guarda el estado fila por
    # fila (nunca una vez por lote completo), y antes de registrar cada
    # fila se comprueba si ya quedó registrada por un intento anterior
    # interrumpido -- si es así, igual se vuelve a pasar por `decidir()`
    # (el estado recursivo restaurado es el de ANTES de esa fila, hay que
    # re-derivarlo para no perder continuidad causal) pero no se vuelve a
    # escribir en la bitácora.
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
        # Referencia = el lote inmediatamente anterior a este (lo más "reciente"
        # que ya se conocía); actual = lo que se acaba de procesar.
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
                n_fallidos += 1  # se cae al reporte plano -- el escalamiento ya quedó en la bitácora
            except Exception:
                # Hallazgo de auditoría: solo se capturaban los 2 tipos de arriba --
                # un error real de red/API (timeout, 5xx) revienta sin capturar y
                # tumba TODO el job, incluso después de ya haber guardado decisiones
                # y estado reales. El triage es asesor: ninguna falla suya debe
                # tumbar el job que ya hizo su trabajo principal.
                n_fallidos += 1

    return ResultadoCorrida(
        agotado=False, filas_procesadas=len(lote), indice_procesado=fin_lote,
        resultado_disparador=resultado_disparador,
        n_triages_intentados=n_intentados, n_triages_exitosos=n_exitosos, n_triages_fallidos=n_fallidos,
    )
