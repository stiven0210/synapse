"""Puente — Fase 3. Actualización atómica del artefacto de política:
escribe a un archivo temporal y solo al final reemplaza el vigente de
forma atómica (`Path.replace` → `os.replace`, atómico en Windows y POSIX),
para que el Ejecutor nunca lea un artefacto a medio escribir mientras el
Calibrador lo actualiza.
"""
import json
from pathlib import Path

from src.artefacto import validar_artefacto


def publicar(artefacto: dict, ruta: Path) -> None:
    """Valida contra `ADR_001` antes de publicar — nunca se publica algo
    que el Ejecutor rechazaría al leerlo (invariante 1 de `ADR_002`)."""
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
