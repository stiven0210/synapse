"""Compares the F1 threshold (the one `calibrar()` uses, production) against
the expected-cost threshold (`src/costo_decision.py`, analysis only) over
the real test slice.

**`COSTO_FALSO_POSITIVO_PLACEHOLDER` is an illustrative assumption, not a
real business figure** -- this project has no way of knowing the real
friction/review cost of blocking a legitimate transaction. Replace this
number with the one that actually applies before making any decision with
this.
"""
import json
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

from src.artefacto import calcular_scores
from src.calibrador import calibrar, cargar_dataset, split_temporal
from src.costo_decision import costo_esperado, mejor_umbral_por_costo

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "raw" / "creditcard.csv"

COSTO_FALSO_POSITIVO_PLACEHOLDER = 5.0  # Illustrative ASSUMPTION -- replace with a real business figure


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

    scores_test = calcular_scores(artefacto, test)
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
