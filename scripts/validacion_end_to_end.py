"""Phase 5 — End-to-end validation (docs/PLAN_DE_TRABAJO.md).

Runs the full pipeline (Calibrator -> Bridge -> Executor -> Veto) over the
TEST slice (never touched in Phase 1), and compares it against a naive
baseline with no slow layer (a fixed threshold on Amount, uncalibrated and
with no recursive features) — the comparison that shows whether the
two-layer pattern delivers something measurable, not just elegant.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.calibrador import cargar_dataset, calibrar, split_temporal
from src.ciclo import CicloDecision
from src.puente import publicar

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "raw" / "creditcard.csv"
RUTA_ARTEFACTO = RAIZ / "data" / "artefacto_politica.json"


def _procesar(ciclo: CicloDecision, filas: list) -> list:
    return [ciclo.decidir(fila).es_sospechosa for fila in filas]


def main() -> None:
    df = cargar_dataset(RUTA_DATASET)
    train, val, test = split_temporal(df)

    artefacto = calibrar(train, val)
    publicar(artefacto, RUTA_ARTEFACTO)  # only real publication path -- CicloDecision only reads from here

    columnas = artefacto["features"] + ["Time"]
    ciclo = CicloDecision(ruta_artefacto=RUTA_ARTEFACTO)

    # Rebuild the recursive state by going through train+val in order (continuity --
    # state is never reset at an arbitrary cut, see Phase 2).
    for fila in train[columnas].to_dict("records"):
        ciclo.decidir(fila)
    for fila in val[columnas].to_dict("records"):
        ciclo.decidir(fila)

    filas_test = test[columnas].to_dict("records")
    y_test = test["Class"].to_numpy()

    inicio = time.perf_counter()
    predicciones_synapse = _procesar(ciclo, filas_test)
    duracion = time.perf_counter() - inicio
    microsegundos_por_decision = (duracion / len(filas_test)) * 1_000_000

    metricas_synapse = {
        "precision": float(precision_score(y_test, predicciones_synapse, zero_division=0)),
        "recall": float(recall_score(y_test, predicciones_synapse, zero_division=0)),
        "f1": float(f1_score(y_test, predicciones_synapse, zero_division=0)),
    }

    # Naive baseline: 99.9th percentile of Amount on TRAIN (never sees test), with
    # no real statistical calibration, no recursive features, no veto layer.
    umbral_baseline = float(np.percentile(train["Amount"], 99.9))
    predicciones_baseline = (test["Amount"] > umbral_baseline).to_numpy()
    metricas_baseline = {
        "precision": float(precision_score(y_test, predicciones_baseline, zero_division=0)),
        "recall": float(recall_score(y_test, predicciones_baseline, zero_division=0)),
        "f1": float(f1_score(y_test, predicciones_baseline, zero_division=0)),
    }

    reporte = {
        "n_test": len(filas_test),
        "n_fraudes_test": int(y_test.sum()),
        "synapse": metricas_synapse,
        "baseline_umbral_fijo": metricas_baseline,
        "umbral_baseline_amount": umbral_baseline,
        "latencia_microsegundos_por_decision": microsegundos_por_decision,
    }
    print(json.dumps(reporte, indent=2))

    ruta_reporte = RAIZ / "data" / "reporte_validacion_end_to_end.json"
    ruta_reporte.write_text(json.dumps(reporte, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en {ruta_reporte}")


if __name__ == "__main__":
    main()
