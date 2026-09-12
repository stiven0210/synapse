"""Corre la validación cruzada walk-forward multi-fold (`src/validacion_cruzada.py`)
sobre el dataset real, para saber si el AUC/umbral de Fase 1 (0.972, un
único split) es representativo del proceso de calibración o el resultado
de un corte particular con suerte.
"""
import json
from pathlib import Path

from src.calibrador import cargar_dataset
from src.validacion_cruzada import validar_walk_forward_multi_fold

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "raw" / "creditcard.csv"


def main() -> None:
    df = cargar_dataset(RUTA_DATASET)

    resumen = validar_walk_forward_multi_fold(df, n_folds=5, frac_train_inicial=0.5)

    print(f"Folds exitosos: {resumen['n_folds_exitosos']} / {resumen['n_folds_exitosos'] + resumen['n_folds_fallidos']}")
    for fold in resumen["folds_fallidos"]:
        print(f"  Fold {fold['fold']} fallido: {fold['error']}")

    if "auc_media" in resumen:
        print(f"\nAUC por fold: {[round(f['auc'], 4) for f in resumen['folds_exitosos']]}")
        print(f"AUC media: {resumen['auc_media']:.4f} +/- {resumen['auc_desviacion_estandar']:.4f}")
        print(f"Umbral media: {resumen['umbral_media']:.4f} +/- {resumen['umbral_desviacion_estandar']:.4f}")
    else:
        print("\nNingún fold tuvo suficientes casos positivos para calibrar.")

    ruta_reporte = RAIZ / "data" / "reporte_validacion_cruzada.json"
    ruta_reporte.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en {ruta_reporte}")


if __name__ == "__main__":
    main()
