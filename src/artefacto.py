"""Contrato del artefacto de política (`docs/adr/0001-policy-artifact.md`)
— vive en un módulo neutral, no en `ejecutor.py` ni en `puente.py`. Corrige
un hallazgo de la auditoría: `puente.py` importaba la validación desde
`ejecutor.py`, invirtiendo la dirección natural de dependencia (el
contrato es compartido, no debería depender del consumidor rápido).
"""
import numpy as np

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


def _sigmoide_vectorizada(z: np.ndarray) -> np.ndarray:
    """Misma forma numéricamente estable que `Ejecutor._sigmoide`
    (evita overflow de `exp` para z muy negativo), vectorizada."""
    resultado = np.empty_like(z, dtype=float)
    positivos = z >= 0
    resultado[positivos] = 1.0 / (1.0 + np.exp(-z[positivos]))
    ez = np.exp(z[~positivos])
    resultado[~positivos] = ez / (1.0 + ez)
    return resultado


def calcular_scores(artefacto: dict, df) -> np.ndarray:
    """Aplica el artefacto a un DataFrame completo, vectorizado -- versión
    batch de la misma fórmula que `Ejecutor.decidir()` aplica
    incrementalmente fila por fila (verificado idéntico en
    `tests/test_artefacto.py`). Solo para diagnóstico/análisis offline
    (ej. `deriva.evaluar_deriva_score`, `scripts/comparacion_umbral_por_costo.py`)
    -- el camino caliente real sigue siendo exclusivamente
    `Ejecutor`/`CicloDecision`, esto nunca se usa ahí. Requiere que `df` ya
    tenga las features del artefacto calculadas (incluidas las recursivas,
    vía `features_recursivas.calcular_features_recursivas_batch`)."""
    X = df[artefacto["features"]].to_numpy(dtype=float)
    z = artefacto["intercepto"] + X @ np.asarray(artefacto["coeficientes"], dtype=float)
    return _sigmoide_vectorizada(z)
