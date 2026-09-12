"""Umbral de decisión por costo esperado — alternativa de análisis a
`calibrador._mejor_umbral_por_f1`, que optimiza F1 (una métrica
estadística, ciega al costo real de cada tipo de error).

Un falso negativo (fraude no detectado) cuesta literalmente el monto de
esa transacción -- ese dato SÍ existe en el dataset (`Amount`), así que se
usa el monto real de cada fraude no detectado, no un promedio inventado.

Un falso positivo (bloquear una transacción legítima) tiene un costo de
fricción/revisión operativa que este dataset no puede dar. Esta función
**no tiene default para `costo_falso_positivo`** -- quien la llama debe
pasar explícitamente un número de negocio real (ver `CLAUDE.md`: "ningún
parámetro no justificado tiene valor por defecto silencioso"). Pasar un
valor inventado como si fuera un dato validado sería exactamente el tipo
de default silencioso que este proyecto prohíbe.

**No reemplaza el umbral que usa `calibrar()`** (que sigue siendo F1, el
único justificado con datos que este proyecto realmente tiene para ambas
clases de error) -- es una herramienta de comparación para juicio humano,
nunca un cambio silencioso de producción.
"""
import numpy as np


def costo_esperado(y_true: np.ndarray, montos: np.ndarray, scores: np.ndarray, umbral: float, costo_falso_positivo: float) -> float:
    """Costo total = (falsos positivos) x costo_falso_positivo + suma de
    los montos de los falsos negativos (el monto real no detectado)."""
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
    """Busca, sobre los `candidatos` de umbral dados (por defecto los
    scores observados, igual que `_mejor_umbral_por_f1` desde su
    corrección), el que minimiza el costo esperado total."""
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
