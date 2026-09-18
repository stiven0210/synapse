"""Automatic recalibration job — the trigger that `docs/PLAN_DE_TRABAJO.md`
flagged as missing: connects drift detection (`src/deriva.py`) with
recalibration (`src/disparador_recalibracion.py`) and with the Executor's
live reload (`CicloDecision.recargar_artefacto()`), over real data.

Simulates the production scenario: a `CicloDecision` already running with
the artifact calibrated on train/val (same as
`scripts/validacion_end_to_end.py`), and this job evaluating whether the
test window (the "most recent" one available) justifies recalibrating —
and if so, reloading the new artifact into the live cycle without losing
the accumulated recursive state.
"""
import json
from pathlib import Path

from src.calibrador import FEATURES, cargar_dataset, calibrar, split_temporal
from src.ciclo import CicloDecision
from src.disparador_recalibracion import evaluar_y_recalibrar_si_hace_falta
from src.puente import publicar

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "raw" / "creditcard.csv"
RUTA_ARTEFACTO = RAIZ / "data" / "artefacto_politica.json"


def main() -> None:
    df = cargar_dataset(RUTA_DATASET)
    train, val, test = split_temporal(df)

    # Initial production state: the artifact calibrated on train/val (same as Phase 5).
    artefacto_inicial = calibrar(train, val)
    publicar(artefacto_inicial, RUTA_ARTEFACTO)
    ciclo = CicloDecision(ruta_artefacto=RUTA_ARTEFACTO)
    print(f"Artefacto inicial publicado: version {artefacto_inicial['version']}, "
          f"calibrado {artefacto_inicial['fecha_calibracion']}")

    # The periodic job: compares the calibration reference (train) against
    # the most recent production window (test) -- same pair
    # scripts/deteccion_deriva.py uses for its report, but here it actually
    # acts if needed.
    resultado = evaluar_y_recalibrar_si_hace_falta(
        df_referencia=train, df_actual=test, columnas_deriva=FEATURES, ruta_artefacto=RUTA_ARTEFACTO,
    )
    print(f"\nDeriva evaluada: {resultado.n_features_con_deriva} / {len(FEATURES)} features con deriva significativa")
    if resultado.contribucion_ponderada_features_con_deriva is not None:
        print(f"Contribución ponderada por coeficiente (0=ruido, 1=todo el peso del modelo): "
              f"{resultado.contribucion_ponderada_features_con_deriva:.2f}")
    if resultado.deriva_score_psi is not None:
        print(f"Deriva del score de salida: PSI={resultado.deriva_score_psi:.4f} ({resultado.deriva_score_interpretacion})")
    print(f"Recomienda recalibrar: {resultado.recomendacion_recalibrar}")

    if resultado.recomendacion_recalibrar and resultado.se_recalibro:
        print(f"Recalibrado y publicado: version {resultado.version_artefacto_nueva}")
        recargo = ciclo.recargar_artefacto()
        print(f"CicloDecision.recargar_artefacto() -> {recargo}")
    elif resultado.recomendacion_recalibrar:
        print(f"Deriva detectada pero NO se recalibró (artefacto vigente intacto): {resultado.error}")
    else:
        print("Sin deriva significativa -- no hace falta recalibrar, artefacto vigente se mantiene.")

    reporte = {
        "n_features_con_deriva": resultado.n_features_con_deriva,
        "contribucion_ponderada_features_con_deriva": resultado.contribucion_ponderada_features_con_deriva,
        "deriva_score_psi": resultado.deriva_score_psi,
        "deriva_score_interpretacion": resultado.deriva_score_interpretacion,
        "recomendacion_recalibrar": resultado.recomendacion_recalibrar,
        "se_recalibro": resultado.se_recalibro,
        "version_artefacto_nueva": resultado.version_artefacto_nueva,
        "error": resultado.error,
    }
    ruta_reporte = RAIZ / "data" / "reporte_disparador_recalibracion.json"
    ruta_reporte.write_text(json.dumps(reporte, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en {ruta_reporte}")


if __name__ == "__main__":
    main()
