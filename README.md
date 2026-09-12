# SYNAPSE

Framework de decisión de dos velocidades: una capa lenta (**Calibrador**)
que aprende sin límite de tiempo, y una capa rápida (**Ejecutor**) que
decide en microsegundos aplicando lo ya aprendido — sin volver a pensar.

**No es una idea nueva** (el patrón existe en control theory desde Kalman,
1960, y toda la industria de detección de fraude ya lo usa — Feedzai,
Featurespace, feature stores como Tecton/Feast). El objetivo aquí es una
implementación simple, abierta y bien entendida del mismo patrón, no
inventar algo sin precedentes.

## Para quién es esto

No es un tutorial de introducción (asume que ya sabes por qué importa el
train/serve skew) ni está pensado para equipos que ya operan una
plataforma empresarial para esto (Feedzai, Featurespace, un feature store
tipo Tecton/Feast) — esos ya tienen el problema resuelto. El encaje real es
un ingeniero solo o un equipo chico construyendo esto con un asistente de
codificación con IA, en una escala o presupuesto donde comprar esa
plataforma no se justifica. Ese encaje no es solo cuestión de tamaño de
empresa: un equipo chico *dentro* de una organización grande y regulada
(un equipo de innovación de un banco construyendo internamente en vez de
comprar) encaja igual o mejor — una decisión de construir-vs-comprar bajo
escrutinio de compliance necesita el historial de ADRs, la auditoría
escrita, y la capa de Veto con fail-closed documentados aquí, no como
overhead sino como lo que permite que la decisión sobreviva una revisión.
La evidencia en la que se apoya este patrón (latencia medida en
microsegundos, precisión/recall reales, una auditoría escrita de bugs
encontrados *y* corregidos) está pensada para ese lector escéptico bajo
escrutinio, no para un pitch de marketing.

## Estado

6 fases completas + auditoría post-implementación (12 hallazgos, 10
corregidos) + detección de deriva (con ponderación por magnitud de
coeficiente) + disparador de recalibración automática (deriva ->
recalibración -> publicación -> recarga en vivo) + bitácora de decisiones +
agente de triage de escalamientos de Veto (LLM en capa lenta, con
grounding y auditor selectivo, nunca en el camino caliente) + validación
cruzada walk-forward multi-fold + umbral de decisión por costo esperado
(análisis, no reemplaza producción) + monitoreo de deriva del score de
salida (complementa la deriva por feature). Agente de triage validado con
`ANTHROPIC_API_KEY` real (2 bugs de integración encontrados y corregidos).
**118 tests, todos en verde.**

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
| SYNAPSE (dos capas) | 0.93 | 0.67 | 0.78 |
| Baseline (umbral fijo, sin calibrar) | 0.00 | 0.00 | 0.00 |

Latencia real medida: **8.3 microsegundos/decisión** de punta a punta
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
│   ├── ADR_002_capa_de_veto.md
│   └── ADR_003_agente_triage_veto.md
├── data/raw/creditcard.csv
├── scripts/
│   ├── validacion_end_to_end.py
│   ├── deteccion_deriva.py
│   ├── recalibracion_automatica.py   # disparador de recalibración, end-to-end
│   └── triage_veto.py                # bitácora -> agente de triage -> reporte para un humano
├── src/
│   ├── artefacto.py       # contrato compartido (validación ADR_001) + calcular_scores() batch
│   ├── calibrador.py
│   ├── ejecutor.py
│   ├── features_recursivas.py
│   ├── puente.py
│   ├── veto.py
│   ├── ciclo.py           # CicloDecision -- único punto de entrada real
│   ├── deriva.py          # PSI + Kolmogorov-Smirnov
│   ├── disparador_recalibracion.py   # conecta deriva -> calibrador -> puente
│   ├── bitacora_decisiones.py        # registro de decisiones, base para triar Veto
│   ├── agente_triage.py              # LLM en capa lenta: hipótesis sobre escalamientos operativos
│   ├── limitador_llamadas.py         # rate limiting diario del agente de triage
│   ├── validacion_cruzada.py         # walk-forward multi-fold, diagnóstico de estabilidad
│   └── costo_decision.py             # umbral por costo esperado, análisis -- no reemplaza producción
└── tests/
```

## Cómo correr

```bash
pip install -r requirements.txt
pytest tests/ -q                              # 57 tests
python scripts/validacion_end_to_end.py       # métricas + latencia reales
python scripts/deteccion_deriva.py            # reporte de deriva real
```
