"""Compara el umbral de F1 (el que usa `calibrar()`, producción) contra el
umbral por costo esperado (`src/costo_decision.py`, solo análisis) sobre el
tramo de prueba real.

**`COSTO_FALSO_POSITIVO_PLACEHOLDER` es un supuesto ilustrativo, no un dato
de negocio real** -- este proyecto no tiene forma de conocer el costo real
de fricción/revisión de bloquear una transacción legítima. Sustituir este
número por el que de verdad aplique antes de tomar cualquier decisión con
esto.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.calibrador import calibrar, cargar_dataset, split_temporal
from src.costo_decision import costo_esperado, mejor_umbral_por_costo

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "raw" / "creditcard.csv"

COSTO_FALSO_POSITIVO_PLACEHOLDER = 5.0  # SUPUESTO ilustrativo -- reemplazar con un dato de negocio real


def _scores(artefacto: dict, df) -> np.ndarray:
    X = df[artefacto["features"]].to_numpy()
    z = artefacto["intercepto"] + X @ np.array(artefacto["coeficientes"])
    return 1.0 / (1.0 + np.exp(-z))


def _metricas(y_true, y_pred) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    df = cargar_dataset(RUTA_DATASET)
    train, val, test = split_temporal(df)
    artefacto = calibrar(train, val)

    scores_test = _scores(artefacto, test)
    y_true = test["Class"].to_numpy()
    montos = test["Amount"].to_numpy()

    umbral_f1 = artefacto["umbral_decision"]
    resultado_costo = mejor_umbral_por_costo(y_true, montos, scores_test, costo_falso_positivo=COSTO_FALSO_POSITIVO_PLACEHOLDER)
    umbral_costo = resultado_costo["umbral_optimo"]

    pred_f1 = (scores_test >= umbral_f1).astype(int)
    pred_costo = (scores_test >= umbral_costo).astype(int)

    reporte = {
        "advertencia": "costo_falso_positivo es un SUPUESTO ilustrativo, no un dato de negocio real",
        "costo_falso_positivo_asumido": COSTO_FALSO_POSITIVO_PLACEHOLDER,
        "umbral_f1": {
            "umbral": umbral_f1,
            "metricas": _metricas(y_true, pred_f1),
            "costo_esperado_real": costo_esperado(y_true, montos, scores_test, umbral_f1, COSTO_FALSO_POSITIVO_PLACEHOLDER),
        },
        "umbral_costo": {
            "umbral": umbral_costo,
            "metricas": _metricas(y_true, pred_costo),
            "costo_esperado_real": resultado_costo["costo_esperado_minimo"],
        },
    }

    print(json.dumps(reporte, indent=2))

    ruta_reporte = RAIZ / "data" / "reporte_comparacion_umbral_costo.json"
    ruta_reporte.write_text(json.dumps(reporte, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en {ruta_reporte}")
    print(f"\nRECORDATORIO: {COSTO_FALSO_POSITIVO_PLACEHOLDER} es un supuesto ilustrativo -- "
          "no tomar decisiones de producción con este número sin reemplazarlo por un costo real.")


if __name__ == "__main__":
    main()
