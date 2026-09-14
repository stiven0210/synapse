"""Puente de Dominio 2 — mismo mecanismo de escritura atómica que `puente.py`
(Dominio 1), validando contra `artefacto_arboles.py` en vez de `artefacto.py`.
No se toca `puente.py`: importa `validar_artefacto` de Dominio 1 hardcoded,
mezclar ambos contratos en un solo módulo invertiría la neutralidad que
`artefacto.py`/`artefacto_arboles.py` ya logran por separado.
"""
import json
from pathlib import Path

from src.artefacto_arboles import validar_artefacto_arboles


def publicar(artefacto: dict, ruta: Path) -> None:
    """Valida contra el contrato de Dominio 2 antes de publicar -- nunca se
    publica algo que `EjecutorArboles` rechazaría al leerlo."""
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
