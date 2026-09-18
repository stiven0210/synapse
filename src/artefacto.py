"""Policy artifact contract (`docs/adr/0001-policy-artifact.md`)
— lives in a neutral module, not in `ejecutor.py` or `puente.py`. Fixes
an audit finding: `puente.py` used to import the validation from
`ejecutor.py`, inverting the natural dependency direction (the contract
is shared, it shouldn't depend on the fast consumer).
"""
import numpy as np

CAMPOS_REQUERIDOS_ADR_001 = {
    "version", "fecha_calibracion", "modelo", "features",
    "coeficientes", "intercepto", "umbral_decision", "metricas_validacion",
}


class ArtefactoInvalido(Exception):
    """The artifact doesn't satisfy `ADR_001` — never operate on an
    artifact that fails validation (invariant 1 of `ADR_002`: no valid
    artifact, no automated decision)."""


def _es_numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def validar_artefacto(artefacto: dict) -> None:
    """Validates shape (ADR_001) AND content — an artifact "correct in
    length" but with non-numeric coefficients or empty feature names used
    to pass this validation and only blow up with a raw error inside the
    hot path (`Ejecutor.decidir()`), on the first real transaction instead
    of at publish time."""
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
    """Same numerically stable shape as `Ejecutor._sigmoide`
    (avoids `exp` overflow for very negative z), vectorized."""
    resultado = np.empty_like(z, dtype=float)
    positivos = z >= 0
    resultado[positivos] = 1.0 / (1.0 + np.exp(-z[positivos]))
    ez = np.exp(z[~positivos])
    resultado[~positivos] = ez / (1.0 + ez)
    return resultado


def calcular_scores(artefacto: dict, df) -> np.ndarray:
    """Applies the artifact to a full DataFrame, vectorized -- the batch
    version of the same formula `Ejecutor.decidir()` applies incrementally
    row by row (verified identical in `tests/test_artefacto.py`). Only for
    offline diagnostics/analysis (e.g. `deriva.evaluar_deriva_score`,
    `scripts/comparacion_umbral_por_costo.py`) -- the real hot path
    remains exclusively `Ejecutor`/`CicloDecision`, this is never used
    there. Requires `df` to already have the artifact's features computed
    (including the recursive ones, via
    `features_recursivas.calcular_features_recursivas_batch`)."""
    X = df[artefacto["features"]].to_numpy(dtype=float)
    z = artefacto["intercepto"] + X @ np.asarray(artefacto["coeficientes"], dtype=float)
    return _sigmoide_vectorizada(z)
