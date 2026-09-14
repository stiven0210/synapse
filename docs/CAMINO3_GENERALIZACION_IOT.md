# Camino 3 — ¿Generaliza el patrón a un dominio distinto de fraude? (exploración)

**Estado:** primera pasada completa, con resultado real y positivo, pero con
una salvedad honesta de comparación (ver "Veredicto"). Ningún código de
producción escrito — igual que tuvo Dominio 2 antes de construirse, esto son
scripts exploratorios de una sola corrida (`data/reporte_camino3_skab.json`),
nada en `src/`.

**Por qué existe esto:** Dominio 1 y Dominio 2 prueban el patrón de dos
velocidades contra *fraude* con distintos datasets — nunca contra un dominio
genuinamente distinto. Esta exploración responde si Calibrador/Ejecutor/
Veto (arquitectura, no el código) tiene sentido fuera de fraude, usando
detección de fallas industriales en tiempo real como el dominio de prueba.

## 1. Dataset: SKAB (Skoltech Anomaly Benchmark)

De los 3 candidatos sugeridos (NASA Bearing, SKAB, Yahoo S5), se eligió
**SKAB**: disponible en GitHub (`github.com/waico/SKAB`, MIT) sin gate de
acceso ni cuenta — clonado directo, sin la fricción que bloqueó IEEE-CIS en
la sesión de Sparkov. NASA Bearing y Yahoo S5 no se probaron (no hizo falta,
SKAB resultó disponible al primer intento).

Se usó el subconjunto `valve1` (16 archivos CSV): una válvula industrial en
un testbed real, lecturas de 8 sensores a 1Hz (acelerómetros, corriente,
presión, temperatura, termocupla, voltaje, caudal), con `anomaly` (0/1) y
`changepoint` (0/1) etiquetados. Los 16 archivos son en realidad **una sola
sesión continua** partida en trozos de ~20 minutos (verificado: los
timestamps empalman exactamente entre archivo y archivo, solo 3 saltos
>2s en 18,160 filas) — se concatenaron y ordenaron por tiempo como un único
stream, igual que `calibrador.py::cargar_dataset` hace con Dominio 1.

**18,160 filas, tasa de anomalía 34.7%** — mucho más balanceado que fraude
(0.077%-3.5% en los datasets ya probados), consistente con ser una falla
mecánica común en un testbed, no un evento raro adversarial.

## 2. Features causales diseñadas para este dominio

Correlación exploratoria con `anomaly`: `Volume Flow RateRMS` domina
(-0.62, el experimento es literalmente el cierre progresivo de una válvula,
así que el caudal cae cuando hay falla), `Accelerometer2RMS` tiene señal
débil positiva (0.10), el resto es ruido (<0.07).

4 features, mismo patrón causal (shift, la fila actual nunca se ve a sí
misma) y mismo `lambda=0.98` que Dominio 1/2:

- `flow_actual`: lectura cruda de `Volume Flow RateRMS` (análogo a `amt`).
- `flow_ewma`: EWMA causal del caudal (análogo a `monto_ewma_global`).
- `flow_delta`: `flow_actual - flow_ewma` (caída súbita respecto a la
  tendencia reciente — feature nueva, sin análogo directo en fraude, tiene
  sentido físico específico de este dominio: una válvula cerrándose se ve
  como una caída sostenida, no un pico aislado).
- `accel2_ewma`: EWMA causal de la vibración del segundo acelerómetro.

**No se forzó una noción de "por entidad"** (análogo a "por cuenta" de
Dominio 2): el testbed es una sola máquina, no hay múltiples entidades
independientes en este dataset — habría sido una analogía forzada, así que
se descartó, tal como se pidió explícitamente en la directiva de esta tarea.

## 3. Resultado real (regresión logística, split temporal 60/20/20)

| | VAL | TEST |
|---|---|---|
| AUC-ROC | — | 0.936 |
| AUC-PR | 0.962 | 0.937 |
| Precisión (umbral por F1 en VAL) | — | 89.0% |
| Recall | — | 87.0% |
| F1 | — | 0.880 |

**Paridad batch/incremental: diferencia máxima 0.00 (exacta)** — el
`EstadoRecursivoIoT` incremental reproduce bit a bit el EWMA calculado en
batch sobre las 18,160 filas, mismo tipo de test que exige la skill
`two-speed-decision` para cualquier feature recursiva nueva.

**Veto mínimo diseñado para este dominio:** invariante físico simple
(`flow_actual < 0` es imposible en este testbed → veta sin importar el
score) — no se disparó ni una vez en test (los datos reales nunca traen un
caudal negativo), pero existe como red de seguridad igual que el invariante
de monto absoluto en Dominio 1/2.

**Latencia real medida** (`time.perf_counter()`, 3,632 decisiones del split
de test, Ejecutor en aritmética pura sin sklearn): **5.51 µs/decisión** —
en el mismo orden de magnitud que Dominio 1 (8.33 µs) y Dominio 2 real
(9.84 µs), confirma que el patrón de velocidad no es específico del
dominio de fraude, es una propiedad de la arquitectura.

## 4. Veredicto honesto — con una salvedad de comparación importante

**La arquitectura generaliza bien**: Calibrador (batch, sin límite de
tiempo) → artefacto → Ejecutor (aritmética pura, microsegundos) → Veto
(invariante físico independiente del modelo) mapea limpio a este dominio
sin forzar nada, y el resultado de detección (F1 0.88, AUC-PR 0.937) es
fuerte.

**Pero esta cifra NO es comparable directamente al leaderboard publicado
de SKAB** (mejor resultado real: Conv-AE, F1=0.78) por dos diferencias de
metodología, no de arquitectura:
1. El leaderboard evalúa modelos **no supervisados** (entrenados solo con
   `anomaly-free.csv`, sin ver ninguna etiqueta de fraude/falla durante el
   entrenamiento) — este experimento usó regresión logística **supervisada**
   (entrenada viendo las etiquetas reales de `anomaly` en TRAIN), que es un
   problema estrictamente más fácil.
2. El leaderboard se evalúa sobre el dataset completo (35 archivos: `valve1`
   + `valve2` + `other`) — este experimento usó solo `valve1` (16 de 35).

**Conclusión honesta:** no se puede afirmar "superamos el estado del arte
publicado de SKAB" con este resultado — sería la misma inflación que ya se
evitó conscientemente en Dominio 2 sección 12. Lo que sí queda validado:
con etiquetas disponibles (igual que en fraude, donde sí tenemos
`Class`/`is_fraud`), el patrón de dos velocidades detecta bien y decide
rápido en un dominio físicamente distinto — la arquitectura no es
fraude-específica. Comparar de forma justa contra el leaderboard (mismo
setup no supervisado, dataset completo) queda pendiente si se retoma esto.

## Pendiente si se decide continuar (no iniciado)

- Comparación justa contra el leaderboard: entrenar sin ver etiquetas
  (ej. un detector de outliers simple sobre `anomaly-free.csv` como
  referencia "normal") y evaluar con las métricas NAB del leaderboard, no
  solo precisión/recall estándar.
- Correr sobre `valve2` y `other` también (35 archivos completos), no solo
  `valve1`.
- Construir en `src/` como Dominio 2 — módulos paralelos
  (`calibrador_iot.py`, `ejecutor_iot.py`, etc.) si el usuario decide que
  vale la pena — no hecho en esta pasada, queda a su decisión.
- Explorar `changepoint` (63 eventos) como problema de detección de
  cambio de régimen en vez de clasificación punto a punto — distinto del
  problema binario resuelto aquí.

Script de esta exploración: scratchpad de la sesión (no persistido en el
repo). Reporte de métricas real: `data/reporte_camino3_skab.json`
(gitignored, igual que el resto de `data/*.json`).
