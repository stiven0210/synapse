"""Bitácora de decisiones — registro estructurado y append-only de cada
`DecisionFinal`. Es la pieza que faltaba antes de poder triar nada: hoy
`CicloDecision.decidir()` devuelve la decisión pero nada la persiste, así
que no hay sobre qué investigar un escalamiento después del hecho.

Deliberadamente NO se invoca desde `CicloDecision.decidir()`: el camino
caliente está medido en microsegundos (ver README) y no debe pagar el costo
de I/O a disco en cada decisión. Registrar es responsabilidad explícita de
quien tiene el ciclo corriendo — mismo principio que
`CicloDecision.recargar_artefacto()` no se llama solo, alguien externo
decide cuándo.

`clasificar_razon()` distingue 4 tipos a partir del texto de `razon` que
producen `veto.py` y `ciclo.py` — de los cuales solo dos representan una
falla de *sistema* (el modelo no pudo producir una decisión, no que haya
decidido que algo es sospechoso):

- `MODELO`: el camino normal, el modelo decidió (sospechosa o no).
- `MONTO_EXCEDE_LIMITE`: invariante de negocio, autoexplicativo en el propio
  dato (el monto) — no necesita triage adicional.
- `SCORE_INVALIDO`: el Ejecutor devolvió un score fuera de rango (NaN u
  otro), sin lanzar excepción — típicamente una feature corrupta aguas
  arriba.
- `ERROR_EJECUTOR`: `Ejecutor.decidir()` lanzó una excepción (ej. falta una
  feature que el artefacto vigente espera) — típicamente un desajuste de
  esquema entre la fuente de transacciones y el artefacto.

`SCORE_INVALIDO` y `ERROR_EJECUTOR` son los candidatos reales para un futuro
agente de triage (ver `docs/PLAN_DE_TRABAJO.md`) — son incidentes operativos,
no juicios sobre una transacción.
"""
import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from src.veto import DecisionFinal


class TipoEscalamiento(str, Enum):
    MODELO = "modelo"
    MONTO_EXCEDE_LIMITE = "monto_excede_limite"
    SCORE_INVALIDO = "score_invalido"
    ERROR_EJECUTOR = "error_ejecutor"


TIPOS_OPERATIVOS = {TipoEscalamiento.SCORE_INVALIDO, TipoEscalamiento.ERROR_EJECUTOR}


def clasificar_razon(razon: str) -> TipoEscalamiento:
    """Acoplado a los textos exactos que producen `veto.py`/`ciclo.py` —
    si esos textos cambian sin actualizar esto, la clasificación se degrada
    en silencio a MODELO. Cubierto por
    `test_clasificar_razon_cubre_los_textos_reales_de_veto_y_ciclo`, que
    llama a `veto.evaluar()`/`CicloDecision.decidir()` de verdad en vez de
    copiar los strings a mano, para que un cambio de texto rompa el test en
    vez de misclasificar en silencio."""
    if "error en el ejecutor" in razon:
        return TipoEscalamiento.ERROR_EJECUTOR
    if "rango" in razon:
        return TipoEscalamiento.SCORE_INVALIDO
    if "excede el límite absoluto" in razon:
        return TipoEscalamiento.MONTO_EXCEDE_LIMITE
    return TipoEscalamiento.MODELO


def registrar_decision(decision: DecisionFinal, transaccion: dict, ruta: Path, timestamp: str | None = None) -> None:
    """Agrega una entrada a la bitácora (JSON Lines — cada escritura es una
    línea independiente, una entrada corrupta o una escritura interrumpida
    nunca invalida las entradas anteriores, a diferencia de reescribir un
    único archivo JSON completo)."""
    entrada = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "tipo": clasificar_razon(decision.razon).value,
        **asdict(decision),
        "transaccion": transaccion,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entrada) + "\n")


def leer_bitacora(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with ruta.open("r", encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def filtrar_escalamientos_operativos(entradas: list[dict]) -> list[dict]:
    """SCORE_INVALIDO y ERROR_EJECUTOR -- los tipos que representan una
    falla de sistema, candidatos reales a triage. MODELO y
    MONTO_EXCEDE_LIMITE quedan fuera a propósito (ver docstring del
    módulo)."""
    tipos = {t.value for t in TIPOS_OPERATIVOS}
    return [e for e in entradas if e["tipo"] in tipos]
