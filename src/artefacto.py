"""Contrato del artefacto de política (`docs/ADR_001_artefacto_de_politica.md`)
— vive en un módulo neutral, no en `ejecutor.py` ni en `puente.py`. Corrige
un hallazgo de la auditoría: `puente.py` importaba la validación desde
`ejecutor.py`, invirtiendo la dirección natural de dependencia (el
contrato es compartido, no debería depender del consumidor rápido).
"""

CAMPOS_REQUERIDOS_ADR_001 = {
    "version", "fecha_calibracion", "modelo", "features",
    "coeficientes", "intercepto", "umbral_decision", "metricas_validacion",
}


class ArtefactoInvalido(Exception):
    """El artefacto no cumple `ADR_001` — nunca se opera con un artefacto
    que no valida (invariante 1 de `ADR_002`: sin artefacto válido, no hay
    decisión automática)."""


def _es_numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def validar_artefacto(artefacto: dict) -> None:
    """Valida forma (ADR_001) Y contenido — un artefacto "correcto en
    longitud" pero con coeficientes no numéricos o nombres de feature
    vacíos pasaba esta validación antes y solo explotaba con un error crudo
    dentro del camino caliente (`Ejecutor.decidir()`), en la primera
    transacción real en vez de al publicarlo."""
    faltantes = CAMPOS_REQUERIDOS_ADR_001 - artefacto.keys()
    if faltantes:
        raise ArtefactoInvalido(f"faltan campos requeridos por ADR_001: {faltantes}")

    features = artefacto["features"]
    coeficientes = artefacto["coeficientes"]

    if not isinstance(features, list) or not isinstance(coeficientes, list):
        raise ArtefactoInvalido("features y coeficientes deben ser listas")
    if len(features) != len(coeficientes):
        raise ArtefactoInvalido(f"features ({len(features)}) y coeficientes ({len(coeficientes)}) de distinta longitud")
    if not features:
        raise ArtefactoInvalido("features no puede estar vacío")
    if not all(isinstance(f, str) and f for f in features):
        raise ArtefactoInvalido("cada feature debe ser un nombre de columna no vacío")
    if not all(_es_numero(c) for c in coeficientes):
        raise ArtefactoInvalido("todos los coeficientes deben ser numéricos")
    if not _es_numero(artefacto["intercepto"]):
        raise ArtefactoInvalido("intercepto debe ser numérico")

    umbral = artefacto["umbral_decision"]
    if not _es_numero(umbral) or not (0.0 <= umbral <= 1.0):
        raise ArtefactoInvalido(f"umbral_decision fuera de [0,1] o no numérico: {umbral!r}")
