"""Fase 5 — Validación end-to-end (docs/PLAN_DE_TRABAJO.md).

Corre el pipeline completo (Calibrador -> Puente -> Ejecutor -> Veto) sobre
el tramo de PRUEBA (nunca tocado en Fase 1), y compara contra un baseline
ingenuo sin capa lenta (un umbral fijo sobre Amount, sin calibrar y sin
features recursivas) — la comparación que demuestra si el patrón de dos
capas aporta algo medible, no solo elegante.
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
    publicar(artefacto, RUTA_ARTEFACTO)  # única vía real de publicación -- CicloDecision solo lee de aquí

    columnas = artefacto["features"] + ["Time"]
    ciclo = CicloDecision(ruta_artefacto=RUTA_ARTEFACTO)

    # Reconstruye el estado recursivo pasando por train+val en orden (continuidad --
    # el estado nunca se reinicia en un corte arbitrario, ver Fase 2).
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

    # Baseline ingenuo: percentil 99.9 de Amount en TRAIN (nunca ve test), sin
    # calibración estadística real, sin features recursivas, sin capa de veto.
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
