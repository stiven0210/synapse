# ADR 001 — Forma del artefacto de política

**Estado:** Decidido.
**Contexto:** El Calibrador (capa lenta) necesita comunicarle al Ejecutor
(capa rápida) los parámetros aprendidos, sin que el Ejecutor tenga que
cargar el modelo de entrenamiento ni ninguna librería de ML pesada. El
artefacto es el único punto de contacto entre las dos capas — su forma
tiene que ser estable, versionada, y auto-descriptiva (el Ejecutor no debe
tener que adivinar en qué orden van los coeficientes).

## Decisión

El artefacto de política es un JSON con esta forma exacta:

```json
{
  "version": 1,
  "fecha_calibracion": "2026-09-12T00:00:00",
  "modelo": "regresion_logistica",
  "features": ["monto_ewma_ratio", "conteo_ventana_5min", "..."],
  "coeficientes": [0.42, -0.13, "..."],
  "intercepto": -1.85,
  "umbral_decision": 0.5,
  "metricas_validacion": {
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "auc": 0.0,
    "n_transacciones_validacion": 0
  }
}
```

- `features` fija el **orden exacto** en que el Ejecutor debe construir el
  vector de entrada — nunca por nombre en tiempo de decisión (evita
  búsquedas costosas en el camino rápido), el orden es contractual.
- `coeficientes` y `features` tienen la misma longitud y el mismo orden —
  se valida al cargar el artefacto (falla ruidoso si no calzan).
- `metricas_validacion` viaja con el artefacto para que cualquier capa de
  monitoreo pueda saber qué tan buena es la política vigente sin tener que
  re-consultar el proceso de calibración.
- `version` es un entero incremental simple (no semver — no hay
  compatibilidad hacia atrás que gestionar todavía, es un contrato interno
  entre dos módulos del mismo proyecto).

## Por qué JSON y no pickle/joblib

Mismo principio que en proyectos anteriores: JSON es legible, auditable a
simple vista, y no depende de que la versión exacta de scikit-learn que
calibró el modelo sea la misma que la que lee el Ejecutor (el Ejecutor ni
siquiera necesita tener scikit-learn instalado — solo multiplica números).
