"""Bridge — Phase 3. Atomic update of the policy artifact: writes to a
temporary file and only at the end atomically replaces the current one
(`Path.replace` → `os.replace`, atomic on both Windows and POSIX), so the
Executor never reads a half-written artifact while the Calibrator updates
it.
"""
import json
from pathlib import Path

from src.artefacto import validar_artefacto


def publicar(artefacto: dict, ruta: Path) -> None:
    """Validates against `ADR_001` before publishing — never publishes
    something the Executor would reject on reading it (invariant 1 of
    `ADR_002`)."""
    validar_artefacto(artefacto)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta_temporal = ruta.with_name(ruta.name + ".tmp")
    ruta_temporal.write_text(json.dumps(artefacto, indent=2), encoding="utf-8")
    ruta_temporal.replace(ruta)


def leer_vigente(ruta: Path) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"no hay artefacto publicado en {ruta}")
    artefacto = json.loads(ruta.read_text(encoding="utf-8"))
    validar_artefacto(artefacto)
    return artefacto
