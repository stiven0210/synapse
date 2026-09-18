"""Decision log — structured, append-only record of every `DecisionFinal`.
This is the piece that was missing before anything could be triaged:
today `CicloDecision.decidir()` returns the decision but nothing persists
it, so there's nothing to investigate an escalation against after the
fact.

Deliberately NOT invoked from `CicloDecision.decidir()`: the hot path is
measured in microseconds (see README) and must not pay the cost of disk
I/O on every decision. Logging is the explicit responsibility of whoever
has the cycle running — same principle as `CicloDecision.recargar_artefacto()`
never being called on its own; someone external decides when.

`clasificar_razon()` distinguishes 4 types from the `razon` text produced
by `veto.py` and `ciclo.py` — of which only two represent a *system*
failure (the model couldn't produce a decision, as opposed to having
decided something is suspicious):

- `MODELO`: the normal path, the model decided (suspicious or not).
- `MONTO_EXCEDE_LIMITE`: a business invariant, self-explanatory from the
  data itself (the amount) — needs no further triage.
- `SCORE_INVALIDO`: the Executor returned an out-of-range score (NaN or
  otherwise) without raising an exception — typically an upstream
  corrupted feature.
- `ERROR_EJECUTOR`: `Ejecutor.decidir()` raised an exception (e.g. a
  feature the current artifact expects is missing) — typically a schema
  mismatch between the transaction source and the artifact.

`SCORE_INVALIDO` and `ERROR_EJECUTOR` are the real candidates for a future
triage agent (see `docs/PLAN_DE_TRABAJO.md`) — they're operational
incidents, not judgment calls about a transaction.
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
    """Coupled to the exact text produced by `veto.py`/`ciclo.py` — if that
    text changes without updating this, the classification silently
    degrades to MODELO. Covered by
    `test_clasificar_razon_cubre_los_textos_reales_de_veto_y_ciclo`, which
    calls the real `veto.evaluar()`/`CicloDecision.decidir()` instead of
    hand-copying the strings, so a text change breaks the test instead of
    silently misclassifying."""
    if "error en el ejecutor" in razon:
        return TipoEscalamiento.ERROR_EJECUTOR
    if "rango" in razon:
        return TipoEscalamiento.SCORE_INVALIDO
    if "excede el límite absoluto" in razon:
        return TipoEscalamiento.MONTO_EXCEDE_LIMITE
    return TipoEscalamiento.MODELO


def registrar_decision(
    decision: DecisionFinal, transaccion: dict, ruta: Path, timestamp: str | None = None, indice_fila: int | None = None
) -> None:
    """Appends an entry to the log (JSON Lines — each write is an
    independent line, a corrupt entry or an interrupted write never
    invalidates earlier entries, unlike rewriting a single complete JSON
    file -- see `leer_bitacora()`).

    `indice_fila` (the transaction's absolute position in its source,
    e.g. the runner's index) is optional, but when the caller knows it,
    it's what lets duplicates be detected and avoided if a run is
    interrupted between logging the decision and persisting how far it
    got (see `src/runner.py`, a real audit finding)."""
    entrada = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "tipo": clasificar_razon(decision.razon).value,
        **asdict(decision),
        "transaccion": transaccion,
        "indice_fila": indice_fila,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entrada) + "\n")


def leer_bitacora(ruta: Path) -> list[dict]:
    """Tolerates corrupt or truncated lines (e.g. a write interrupted
    halfway through the last line -- a real audit finding: it used to be
    that `json.loads` would blow up on that line and no entry, not even
    the earlier valid ones, could be read, contradicting the guarantee
    this module documents). A line that fails to parse is skipped, it
    doesn't invalidate the rest."""
    if not ruta.exists():
        return []
    entradas = []
    with ruta.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                entradas.append(json.loads(linea))
            except json.JSONDecodeError:
                continue
    return entradas


def filtrar_escalamientos_operativos(entradas: list[dict]) -> list[dict]:
    """SCORE_INVALIDO and ERROR_EJECUTOR -- the types that represent a
    system failure, the real candidates for triage. MODELO and
    MONTO_EXCEDE_LIMITE are deliberately left out (see the module
    docstring)."""
    tipos = {t.value for t in TIPOS_OPERATIVOS}
    return [e for e in entradas if e["tipo"] in tipos]
