"""Veto-escalation triage job — connects end to end what
`docs/PLAN_DE_TRABAJO.md` flagged as pending: read the real decision log,
filter the operational escalations (`SCORE_INVALIDO`/`ERROR_EJECUTOR`),
invoke `agente_triage.AgenteTriage`, and show the result to a human.

Since this run has no real production traffic, it first feeds the
decision log with the same 2 known-real-cause scenarios from
`tests/test_bitacora_decisiones.py` (a NaN feature, a missing feature) —
so the job has something real to triage instead of starting from an empty
file.

If `ANTHROPIC_API_KEY` isn't configured, it doesn't invent a fake call:
it explicitly reports that it can't triage with a real LLM and falls back
to the flat report (the log entry as-is, no hypothesis) — the same
graceful-degradation principle as the agent's circuit breaker.
"""
import json
import os
from pathlib import Path

from src.agente_triage import AgenteTriage, CircuitoAbierto
from src.bitacora_decisiones import filtrar_escalamientos_operativos, leer_bitacora, registrar_decision
from src.ciclo import CicloDecision
from src.limitador_llamadas import LimitadorLlamadasDiarias, PresupuestoAgotado
from src.puente import publicar

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ARTEFACTO = RAIZ / "data" / "artefacto_politica.json"
RUTA_BITACORA = RAIZ / "data" / "bitacora_decisiones.jsonl"
RUTA_REPORTE_DERIVA = RAIZ / "data" / "reporte_deriva.json"
RUTA_REPORTE_TRIAGE = RAIZ / "data" / "reporte_triage_veto.json"
RUTA_ESTADO_LIMITADOR = RAIZ / "data" / "estado_limitador_triage.json"

ARTEFACTO_DEMO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "regresion_logistica",
    "features": ["Amount"],
    "coeficientes": [0.1],
    "intercepto": -1.0,
    "umbral_decision": 0.5,
    "metricas_validacion": {},
}


def _alimentar_bitacora_con_escenarios_de_ejemplo(ciclo: CicloDecision) -> None:
    """With no real production traffic, reproduces the same 2
    known-real-cause scenarios used in tests/test_bitacora_decisiones.py,
    so the job has real escalations to triage."""
    escenarios = [
        {"Amount": float("nan"), "Time": 0.0},  # corrupt feature -> SCORE_INVALIDO
        {"Time": 1.0},  # missing "Amount" -> ERROR_EJECUTOR
        {"Amount": 20.0, "Time": 2.0},  # normal -> MODELO, shouldn't show up in triage
    ]
    for transaccion in escenarios:
        registrar_decision(ciclo.decidir(transaccion), transaccion, RUTA_BITACORA)


def _armar_contexto() -> dict:
    """Deterministic context available to ground the agent -- the most
    recent drift report, if it exists (see scripts/deteccion_deriva.py
    and scripts/recalibracion_automatica.py)."""
    if RUTA_REPORTE_DERIVA.exists():
        return {"reporte_deriva_reciente": json.loads(RUTA_REPORTE_DERIVA.read_text(encoding="utf-8"))}
    return {"reporte_deriva_reciente": None}


def main() -> None:
    publicar(ARTEFACTO_DEMO, RUTA_ARTEFACTO)
    ciclo = CicloDecision(ruta_artefacto=RUTA_ARTEFACTO)
    _alimentar_bitacora_con_escenarios_de_ejemplo(ciclo)

    entradas = leer_bitacora(RUTA_BITACORA)
    operativos = filtrar_escalamientos_operativos(entradas)
    print(f"Bitácora: {len(entradas)} decisiones registradas, {len(operativos)} escalamientos operativos a triar.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    reporte = {"n_escalamientos_operativos": len(operativos), "api_key_configurada": bool(api_key), "resultados": []}

    if not api_key:
        print("\nANTHROPIC_API_KEY no configurada -- no se puede triar con un LLM real.")
        print("Reporte plano (sin hipótesis) de cada escalamiento operativo:\n")
        for entrada in operativos:
            print(f"  [{entrada['tipo']}] {entrada['razon']} -- transacción: {entrada['transaccion']}")
            reporte["resultados"].append({"entrada": entrada, "triage": None, "motivo_sin_triage": "sin ANTHROPIC_API_KEY"})
        RUTA_REPORTE_TRIAGE.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReporte guardado en {RUTA_REPORTE_TRIAGE}")
        return

    from src.agente_triage import crear_cliente_claude

    cliente = LimitadorLlamadasDiarias(
        cliente_llm=crear_cliente_claude(), max_llamadas_por_dia=50, ruta_estado=RUTA_ESTADO_LIMITADOR,
    )
    agente = AgenteTriage(cliente_llm=cliente)
    contexto = _armar_contexto()

    for entrada in operativos:
        print(f"\n[{entrada['tipo']}] {entrada['razon']}")
        try:
            resultado = agente.triar(entrada, contexto)
        except (CircuitoAbierto, PresupuestoAgotado) as e:
            print(f"  Sin triage ({type(e).__name__}): {e} -- cayendo al reporte plano.")
            reporte["resultados"].append({"entrada": entrada, "triage": None, "motivo_sin_triage": str(e)})
            continue
        except Exception as e:
            # Audit finding: a real network/API error (timeout, 5xx) must not
            # bring down the whole script -- triage is advisory, the rest of
            # the escalations and the report already generated must keep going.
            print(f"  Sin triage (error del cliente LLM: {type(e).__name__}): {e} -- cayendo al reporte plano.")
            reporte["resultados"].append({"entrada": entrada, "triage": None, "motivo_sin_triage": f"{type(e).__name__}: {e}"})
            continue

        if resultado.descartado:
            print(f"  Hipótesis descartada ({resultado.detalle.get('capa_fallida')}) -- reporte plano, sin conclusión confiable.")
        else:
            print(f"  Hipótesis: {resultado.hipotesis}")
            print(f"  Severidad: {resultado.severidad} | Acción sugerida: {resultado.accion_sugerida} | Confianza: {resultado.confianza:.2f}")
        reporte["resultados"].append({
            "entrada": entrada,
            "triage": {
                "descartado": resultado.descartado,
                "hipotesis": resultado.hipotesis,
                "severidad": resultado.severidad,
                "accion_sugerida": resultado.accion_sugerida,
                "confianza": resultado.confianza,
                "detalle": resultado.detalle,
            },
        })

    RUTA_REPORTE_TRIAGE.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte guardado en {RUTA_REPORTE_TRIAGE}")


if __name__ == "__main__":
    main()
