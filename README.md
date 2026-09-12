# SYNAPSE

Framework de decisión de dos velocidades: una capa lenta (**Calibrador**)
que aprende sin límite de tiempo, y una capa rápida (**Ejecutor**) que
decide en microsegundos aplicando lo ya aprendido — sin volver a pensar.

**No es una idea nueva** (el patrón existe en control theory desde Kalman,
1960, y toda la industria de detección de fraude ya lo usa — Feedzai,
Featurespace, feature stores como Tecton/Feast). El objetivo aquí es una
implementación simple, abierta y bien entendida del mismo patrón, no
inventar algo sin precedentes.

## Estado

6 fases completas + auditoría post-implementación (12 hallazgos, 10
corregidos) + detección de deriva + disparador de recalibración automática
(deriva -> recalibración -> publicación -> recarga en vivo) + bitácora de
decisiones (prerequisito para triar escalamientos de Veto). **70 tests,
todos en verde.**

## Las piezas

1. **Calibrador** — analiza datos históricos, produce un artefacto de
   política versionado (JSON: pesos, umbrales). Nunca decide en tiempo real.
2. **Ejecutor** — mantiene un estado compacto actualizado de forma
   recursiva (O(1) por evento, `collections.deque`), aplica el artefacto
   vigente, decide en microsegundos. Nunca reentrena ni llama a red.
   **No es la API pública** — de bajo nivel a propósito, ver `CicloDecision`.
3. **Puente** — actualiza el artefacto de forma atómica (archivo temporal +
   reemplazo atómico), para que el Ejecutor nunca lea un artefacto a medio
   escribir.
4. **Veto** — invariantes duros independientes del modelo. Protege incluso
   si el Calibrador o el Ejecutor están mal calibrados.
5. **`CicloDecision`** (`src/ciclo.py`) — **único punto de entrada real**:
   fuerza que el artefacto pase siempre por el Puente, que el Veto siempre
   corra, y convierte cualquier fallo (artefacto ausente/corrupto, error
   del Ejecutor) en una decisión real de "escalar a revisión manual" en vez
   de propagar una excepción sin capturar. También expone
   `recargar_artefacto()` — el mecanismo para actualizar el modelo en vivo
   (nueva calibración por deriva detectada, nueva política, nuevo
   hallazgo), conservando el estado recursivo acumulado y sin reemplazar
   nada si la recarga falla (degradación con gracia).
6. **Deriva** (`src/deriva.py`) — PSI + Kolmogorov-Smirnov, responde
   *cuándo* hace falta llamar a `recargar_artefacto()`.

Sin capa de interfaces genérica todavía — se construye concreto para el
Dominio 1, se generaliza después de un segundo dominio real. Ver
`docs/PLAN_DE_TRABAJO.md`.

## Dominio 1: detección de fraude en transacciones

Dataset: Credit Card Fraud Detection (público, 284,807 transacciones, 492
fraudes, sin credenciales de pago). Ver `docs/PLAN_DE_TRABAJO.md` para la
limitación de datos conocida (sin identificador de tarjeta/cuenta) y cómo
se maneja.

**Resultados reales de validación** (`scripts/validacion_end_to_end.py`,
sobre el tramo de prueba nunca antes visto):

| | Precisión | Recall | F1 |
|---|---|---|---|
| SYNAPSE (dos capas) | 0.50 | 0.79 | 0.61 |
| Baseline (umbral fijo, sin calibrar) | 0.00 | 0.00 | 0.00 |

Latencia real medida: **8.6 microsegundos/decisión** de punta a punta
(Ejecutor + Veto) — el requisito real de la industria es p99 < 50
*milisegundos*, así que esto está muy por debajo del cuello de botella real
(no vale la pena optimizar más la velocidad; sí vale la pena seguir
afinando corrección y robustez).

**Deriva real detectada** (`scripts/deteccion_deriva.py`): 8 de 31 features
ya muestran deriva significativa entre train y test, pese a cubrir solo
~48 horas — ver `docs/PLAN_DE_TRABAJO.md` para el detalle.

## Estructura

```
synapse/
├── CLAUDE.md
├── docs/
│   ├── PLAN_DE_TRABAJO.md          # empezar aquí -- estado, auditoría, resultados reales
│   ├── ADR_001_artefacto_de_politica.md
│   └── ADR_002_capa_de_veto.md
├── data/raw/creditcard.csv
├── scripts/
│   ├── validacion_end_to_end.py
│   ├── deteccion_deriva.py
│   └── recalibracion_automatica.py   # disparador de recalibración, end-to-end
├── src/
│   ├── artefacto.py       # contrato compartido (validación ADR_001)
│   ├── calibrador.py
│   ├── ejecutor.py
│   ├── features_recursivas.py
│   ├── puente.py
│   ├── veto.py
│   ├── ciclo.py           # CicloDecision -- único punto de entrada real
│   ├── deriva.py          # PSI + Kolmogorov-Smirnov
│   ├── disparador_recalibracion.py   # conecta deriva -> calibrador -> puente
│   └── bitacora_decisiones.py        # registro de decisiones, base para triar Veto
└── tests/
```

## Cómo correr

```bash
pip install -r requirements.txt
pytest tests/ -q                              # 57 tests
python scripts/validacion_end_to_end.py       # métricas + latencia reales
python scripts/deteccion_deriva.py            # reporte de deriva real
```
