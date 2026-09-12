r"""Punto de entrada para Task Scheduler (o cron) — una invocación = un
"día" de operación real. Pensado para correr sin supervisión: cada corrida
deja un renglón en `data/runner_log.jsonl` (append-only, igual que la
bitácora) para que se pueda revisar la historia días después sin haber
estado mirando la consola.

Registrar en Windows Task Scheduler (PowerShell, como administrador):

    $accion = New-ScheduledTaskAction -Execute "python" `
        -Argument "scripts\runner_diario.py" -WorkingDirectory "C:\proyectos\synapse"
    $disparador = New-ScheduledTaskTrigger -Daily -At 3am
    Register-ScheduledTask -TaskName "SYNAPSE-runner-diario" -Action $accion -Trigger $disparador

O por CLI (cmd.exe):

    schtasks /create /tn "SYNAPSE-runner-diario" /tr "python C:\proyectos\synapse\scripts\runner_diario.py" /sc daily /st 03:00
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.agente_triage import AgenteTriage, crear_cliente_claude
from src.limitador_llamadas import LimitadorLlamadasDiarias
from src.runner import correr_ciclo_diario

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "raw" / "creditcard.csv"
RUTA_ARTEFACTO = RAIZ / "data" / "artefacto_politica.json"
RUTA_BITACORA = RAIZ / "data" / "bitacora_decisiones.jsonl"
RUTA_ESTADO_RUNNER = RAIZ / "data" / "estado_runner.json"
RUTA_REPORTE_DERIVA = RAIZ / "data" / "reporte_deriva.json"
RUTA_LOG = RAIZ / "data" / "runner_log.jsonl"
RUTA_ESTADO_LIMITADOR = RAIZ / "data" / "estado_limitador_triage.json"

TAMANO_LOTE = 5000  # filas "nuevas" por corrida -- a este ritmo, el dataset completo (~285k filas) dura ~57 corridas


def _armar_contexto_triage() -> dict:
    if RUTA_REPORTE_DERIVA.exists():
        return {"reporte_deriva_reciente": json.loads(RUTA_REPORTE_DERIVA.read_text(encoding="utf-8"))}
    return {"reporte_deriva_reciente": None}


def _crear_agente_triage() -> AgenteTriage | None:
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    cliente = LimitadorLlamadasDiarias(
        cliente_llm=crear_cliente_claude(), max_llamadas_por_dia=50, ruta_estado=RUTA_ESTADO_LIMITADOR,
    )
    return AgenteTriage(cliente_llm=cliente)


def main() -> None:
    resultado = correr_ciclo_diario(
        ruta_dataset=RUTA_DATASET, ruta_artefacto=RUTA_ARTEFACTO, ruta_bitacora=RUTA_BITACORA,
        ruta_estado_runner=RUTA_ESTADO_RUNNER, tamano_lote=TAMANO_LOTE,
        agente_triage=_crear_agente_triage(), contexto_triage=_armar_contexto_triage,
    )

    renglon_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agotado": resultado.agotado,
        "filas_procesadas": resultado.filas_procesadas,
        "indice_procesado": resultado.indice_procesado,
        "recomendacion_recalibrar": resultado.resultado_disparador.recomendacion_recalibrar if resultado.resultado_disparador else None,
        "se_recalibro": resultado.resultado_disparador.se_recalibro if resultado.resultado_disparador else None,
        "n_triages_intentados": resultado.n_triages_intentados,
        "n_triages_exitosos": resultado.n_triages_exitosos,
        "n_triages_fallidos": resultado.n_triages_fallidos,
    }
    with RUTA_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(renglon_log, ensure_ascii=False) + "\n")

    if resultado.agotado:
        print(f"Dataset agotado en índice {resultado.indice_procesado} -- nada nuevo que procesar hoy.")
    else:
        print(f"Procesadas {resultado.filas_procesadas} filas (índice {resultado.indice_procesado}).")
        if resultado.resultado_disparador:
            print(f"Recomienda recalibrar: {resultado.resultado_disparador.recomendacion_recalibrar} "
                  f"| se recalibró: {resultado.resultado_disparador.se_recalibro}")
        print(f"Triage: {resultado.n_triages_exitosos}/{resultado.n_triages_intentados} hipótesis generadas "
              f"({resultado.n_triages_fallidos} fallidas por error del cliente).")
    print(f"Log: {RUTA_LOG}")


if __name__ == "__main__":
    main()
