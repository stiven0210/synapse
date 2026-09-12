"""Job de recalibración automática — el disparador que `docs/PLAN_DE_TRABAJO.md`
señalaba como faltante: conecta detección de deriva (`src/deriva.py`) con
recalibración (`src/disparador_recalibracion.py`) y con la recarga en vivo
del Ejecutor (`CicloDecision.recargar_artefacto()`), sobre datos reales.

Simula el escenario de producción: un `CicloDecision` ya corriendo con el
artefacto calibrado en train/val (igual que `scripts/validacion_end_to_end.py`),
y este job evaluando si la ventana de test (la "más reciente" disponible)
justifica recalibrar — y si es así, recargando el artefacto nuevo en el
ciclo vivo sin perder el estado recursivo acumulado.
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

    # Estado inicial de producción: el artefacto calibrado con train/val (igual que Fase 5).
    artefacto_inicial = calibrar(train, val)
    publicar(artefacto_inicial, RUTA_ARTEFACTO)
    ciclo = CicloDecision(ruta_artefacto=RUTA_ARTEFACTO)
    print(f"Artefacto inicial publicado: version {artefacto_inicial['version']}, "
          f"calibrado {artefacto_inicial['fecha_calibracion']}")

    # El job periódico: compara la referencia de calibración (train) contra
    # la ventana más reciente de producción (test) -- mismo par que usa
    # scripts/deteccion_deriva.py para el reporte, pero aquí sí actúa si hace falta.
    resultado = evaluar_y_recalibrar_si_hace_falta(
        df_referencia=train, df_actual=test, columnas_deriva=FEATURES, ruta_artefacto=RUTA_ARTEFACTO,
    )
    print(f"\nDeriva evaluada: {resultado.n_features_con_deriva} / {len(FEATURES)} features con deriva significativa")
    if resultado.contribucion_ponderada_features_con_deriva is not None:
        print(f"Contribución ponderada por coeficiente (0=ruido, 1=todo el peso del modelo): "
              f"{resultado.contribucion_ponderada_features_con_deriva:.2f}")
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
