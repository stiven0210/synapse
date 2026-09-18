"""Expected-cost decision threshold — an analysis alternative to
`calibrador._mejor_umbral_por_f1`, which optimizes F1 (a statistical
metric, blind to the real cost of each type of error).

A false negative (undetected fraud) literally costs the amount of that
transaction -- that data DOES exist in the dataset (`Amount`), so the real
amount of each undetected fraud is used, not an invented average.

A false positive (blocking a legitimate transaction) has an
operational friction/review cost this dataset can't provide. This
function **has no default for `costo_falso_positivo`** -- the caller must
explicitly pass a real business number (see `CLAUDE.md`: "no unjustified
parameter gets a silent default value"). Passing a made-up value as if it
were validated data would be exactly the kind of silent default this
project prohibits.

**Doesn't replace the threshold `calibrar()` uses** (which stays F1, the
only one this project actually has data to justify for both error
classes) -- this is a comparison tool for human judgment, never a silent
production change.
"""
import numpy as np


def costo_esperado(y_true: np.ndarray, montos: np.ndarray, scores: np.ndarray, umbral: float, costo_falso_positivo: float) -> float:
    """Total cost = (false positives) x costo_falso_positivo + sum of the
    amounts of the false negatives (the real undetected amount)."""
    y_true = np.asarray(y_true)
    montos = np.asarray(montos)
    scores = np.asarray(scores)

    y_pred = (scores >= umbral).astype(int)
    falsos_positivos = (y_pred == 1) & (y_true == 0)
    falsos_negativos = (y_pred == 0) & (y_true == 1)

    return float(falsos_positivos.sum() * costo_falso_positivo + montos[falsos_negativos].sum())


def mejor_umbral_por_costo(
    y_true: np.ndarray, montos: np.ndarray, scores: np.ndarray, costo_falso_positivo: float, candidatos: np.ndarray | None = None
) -> dict:
    """Searches, over the given threshold `candidatos` (by default the
    observed scores, same as `_mejor_umbral_por_f1` since its fix), for the
    one that minimizes total expected cost."""
    if costo_falso_positivo <= 0:
        raise ValueError(
            f"costo_falso_positivo debe ser > 0 (recibido {costo_falso_positivo!r}) -- "
            "si de verdad no tiene costo, no hay nada que optimizar aquí"
        )

    scores = np.asarray(scores)
    candidatos = np.unique(scores) if candidatos is None else np.asarray(candidatos)

    curva = []
    mejor_umbral, mejor_costo = None, float("inf")
    for umbral in candidatos:
        costo = costo_esperado(y_true, montos, scores, float(umbral), costo_falso_positivo)
        curva.append({"umbral": float(umbral), "costo_esperado": costo})
        if costo < mejor_costo:
            mejor_umbral, mejor_costo = float(umbral), costo

    return {
        "umbral_optimo": mejor_umbral,
        "costo_esperado_minimo": mejor_costo,
        "costo_falso_positivo_asumido": costo_falso_positivo,
        "curva_costo_por_umbral": curva,
    }
