"""Applies drift detection (`src/deriva.py`) to real data: train
(calibration reference) vs. test (the most "current" slice we have),
to see whether there's already natural drift over the dataset's ~48
hours — plus a positive-control test (artificially injected drift) to
confirm the detector does fire when it should.
"""
import json
from pathlib import Path

import numpy as np

from src.calibrador import FEATURES, cargar_dataset, split_temporal
from src.deriva import evaluar_deriva

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "raw" / "creditcard.csv"


def main() -> None:
    df = cargar_dataset(RUTA_DATASET)
    train, val, test = split_temporal(df)

    # Negative control: real train vs. test -- is there natural drift over ~48h?
    reporte_real = evaluar_deriva(train, test, columnas=FEATURES)
    resumen_real = {
        col: {"psi": round(r["psi"], 4), "interpretacion": r["psi_interpretacion"]}
        for col, r in reporte_real["por_feature"].items()
    }
    print("=== Deriva real: train vs. test ===")
    print(json.dumps(resumen_real, indent=2))
    print(f"Features con deriva significativa: {reporte_real['n_features_con_deriva_psi']} / {len(FEATURES)}")
    print(f"Recomienda recalibrar: {reporte_real['recomendacion_recalibrar']}")

    # Positive control: inject artificial drift into Amount (double the scale) --
    # the detector MUST flag it, otherwise the detector itself is broken.
    test_con_deriva = test.copy()
    test_con_deriva["Amount"] = test_con_deriva["Amount"] * 2.0 + 50.0
    reporte_inyectado = evaluar_deriva(train, test_con_deriva, columnas=["Amount"])
    print("\n=== Control positivo: Amount con deriva inyectada (x2 + 50) ===")
    print(json.dumps(reporte_inyectado["por_feature"]["Amount"], indent=2))
    assert reporte_inyectado["recomendacion_recalibrar"], "el detector debería marcar esta deriva inyectada"
    print("Control positivo OK: el detector sí dispara cuando hay deriva real.")

    ruta_reporte = RAIZ / "data" / "reporte_deriva.json"
    ruta_reporte.write_text(
        json.dumps({"real_train_vs_test": resumen_real, "control_positivo_amount": reporte_inyectado["por_feature"]["Amount"]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nReporte guardado en {ruta_reporte}")


if __name__ == "__main__":
    main()
