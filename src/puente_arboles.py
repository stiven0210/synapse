"""Domain 2's Bridge — same atomic-write mechanism as `puente.py`
(Domain 1), validating against `artefacto_arboles.py` instead of
`artefacto.py`. `puente.py` is left untouched: it hardcodes an import of
Domain 1's `validar_artefacto`, and mixing both contracts into a single
module would undo the neutrality `artefacto.py`/`artefacto_arboles.py`
already achieve separately.
"""
import json
from pathlib import Path

from src.artefacto_arboles import validar_artefacto_arboles


def publicar(artefacto: dict, ruta: Path) -> None:
    """Validates against Domain 2's contract before publishing -- never
    publishes something `EjecutorArboles` would reject on reading it."""
    validar_artefacto_arboles(artefacto)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta_temporal = ruta.with_name(ruta.name + ".tmp")
    ruta_temporal.write_text(json.dumps(artefacto, indent=2), encoding="utf-8")
    ruta_temporal.replace(ruta)


def leer_vigente(ruta: Path) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"no hay artefacto de árboles publicado en {ruta}")
    artefacto = json.loads(ruta.read_text(encoding="utf-8"))
    validar_artefacto_arboles(artefacto)
    return artefacto
