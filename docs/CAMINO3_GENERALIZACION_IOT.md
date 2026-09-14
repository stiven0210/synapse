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

## Comparación justa contra el leaderboard real de SKAB (2026-09-14)

Cierra el pendiente de arriba: comparación **no supervisada**, sobre los
**34 archivos con anomalía** (todo `data/` excepto `anomaly-free.csv`, que
el propio protocolo oficial de SKAB no usa para este problema), con la
**metodología oficial exacta** del repo (`core/utils.py::load_preprocess_skab`
para el split, `core/metrics.py::chp_score(metric="binary")` para F1/FAR/MAR),
reutilizando el mismo código que generó los resultados ya publicados en
`results/results-*.pkl` — no una métrica propia inventada.

**Split oficial reutilizado tal cual:** por archivo, las primeras 400 filas
(cronológicas, sin mezclar) son train; el resto es test. El train nunca
incluye las columnas `anomaly`/`changepoint` — coincide exactamente con el
principio de SYNAPSE (Calibrador aprende offline sin supervisión de
etiquetas).

**Detector no supervisado construido, estilo SYNAPSE:**
- **Calibrador** (offline, una vez por archivo): media y desviación
  estándar de las 8 features de sensor, sobre las 400 filas de train.
  Nunca ve `anomaly`/`changepoint`.
- **Ejecutor** (aritmética pura, O(1) por fila): z-score máximo entre las 8
  features de la fila nueva contra la media/desvío aprendidos. Anómalo si
  supera **3 desviaciones estándar** — convención estadística estándar de
  control de procesos, fijada de antemano, **nunca ajustada contra las
  etiquetas de test** (ni siquiera se miraron hasta evaluar al final).
- Mismo suavizado de predicción (`rolling(3).median()`) que usa el propio
  notebook de Isolation Forest del repo SKAB, para no darle a este método
  una ventaja de post-proceso que los demás no tienen.

**Bug real encontrado y corregido durante esta comparación:** la primera
versión emparejó los 34 datasets de mi carga con los de cada pickle
publicado **por posición de lista** — asumiendo que el orden coincidía.
No coincidía (verificado: los rangos de fecha del dataset `i` en mi carga
y en el pickle eran distintos, `os.walk` enumera archivos en orden distinto
según el filesystem). Esto habría producido una comparación inválida (con
números que además parecían razonables — el error no era detectable "a
ojo"). Corregido emparejando cada dataset por una firma real (largo +
primer timestamp + último timestamp de su índice de tiempo), no por
posición.

**Resultado real, con la metodología oficial y el emparejamiento corregido**
(F1 / FAR / MAR, orden descendente por F1; 2 de los 11 pickles —
`MSCRED` y `Vanilla_LSTM` — no pudieron alinearse por firma, 0/34
coincidencias, probablemente usan un preprocesamiento de ventana distinto
al split estándar de 400 filas; se excluyen de la tabla por no ser
confiables, no se fuerza la comparación):

| Método | F1 | FAR | MAR |
|---|---|---|---|
| Conv_AE (publicado) | 0.78 | 13.55% | 28.02% |
| MSET (publicado) | 0.78 | 39.73% | 14.13% |
| **SYNAPSE (z-score 3σ, no supervisado)** | **0.76** | **43.32%** | **14.95%** |
| T2-q (publicado) | 0.76 | 26.62% | 24.92% |
| LSTM_AE (publicado) | 0.74 | 29.96% | 25.92% |
| T2 (publicado) | 0.66 | 19.21% | 42.60% |
| Vanilla_AE (publicado) | 0.39 | 2.59% | 75.15% |
| Isolation_Forest (publicado) | 0.29 | 2.56% | 82.89% |
| Arima_anomaly_detection (publicado) | 0.00 | 0.01% | 100.00% |

**Latencia real del Ejecutor** (`time.perf_counter()`, 23,801 decisiones de
test reales): **13.05 µs/decisión** — mismo orden de magnitud que Dominio 1
(8.33 µs) y Dominio 2 (9.84 µs).

**Veredicto honesto:** con la comparación ahora sí justa (mismo split
oficial, misma métrica oficial, mismo código de evaluación, dataset
completo, cero supervisión), el patrón de dos velocidades de SYNAPSE —
implementado como el detector estadístico más simple posible (media,
desvío estándar, z-score, 3 líneas de aritmética) — queda **competitivo,
no superior**: 3er lugar de 9 métodos evaluables, a 2 puntos de F1 del
mejor (Conv_AE, una red neuronal). Es del mismo orden de FAR/MAR que MSET
(el otro método estadístico clásico del leaderboard, no una red neuronal),
lo cual tiene sentido — comparte familia de enfoque. El resultado confirma
que la arquitectura generaliza y que incluso su versión más simple no
queda descolgada del estado del arte, sin necesidad de inflar la
afirmación a "lo superamos".

## Intento de mejora del detector — Mahalanobis y suavizado EWMA (2026-09-14)

A pedido explícito de "¿no la podemos mejorar?", se probaron 2 mejoras
principiadas sobre el detector z-score/3-sigma (F1=0.76), manteniendo la
misma disciplina de nunca mirar las etiquetas de test para ajustar nada —
mismo split oficial de SKAB, misma métrica (`chp_score`, `metric="binary"`),
mismo emparejamiento por firma real contra los pickles publicados.

**Mejora 1 — Distancia de Mahalanobis** en vez de z-score independiente por
feature: usa la matriz de covarianza aprendida en TRAIN (vía
`np.linalg.pinv`, robusto a features con varianza ~0) para capturar
desviaciones conjuntas entre las 8 señales correlacionadas, en vez de mirar
cada una por separado. Umbral: valor crítico chi-cuadrado al 99% con 8
grados de libertad (`scipy.stats.chi2.ppf(0.99, df=8)` ≈ 20.09) — el
equivalente multivariado principiado del "3-sigma" univariado, nunca
ajustado contra test.

**Mejora 2 — Suavizado causal EWMA** (`lambda=0.98`, mismo valor de
referencia del resto de SYNAPSE) del score en vez de una lectura
instantánea — el estado del EWMA continúa sin reiniciar de TRAIN a TEST
(mismo principio de continuidad causal que `EstadoRecursivoGlobal`). El
umbral se deriva aplicando el mismo suavizado a los scores de TRAIN y
tomando media+3·desvío de esa distribución suavizada — también ciego a
test. Probada combinada con z-score y con Mahalanobis.

**Resultado real, las 4 variantes (F1/FAR/MAR, 34 archivos, métrica
oficial):**

| Variante | F1 | FAR | MAR |
|---|---|---|---|
| z-score, 3-sigma (baseline, reproducido idéntico) | 0.76 | 43.3% | 14.9% |
| Mahalanobis, chi-cuadrado 99% | 0.76 | 47.6% | 12.8% |
| z-score + EWMA causal | 0.75 | 62.7% | 6.9% |
| Mahalanobis + EWMA causal | 0.75 | 70.9% | 4.2% |

**Veredicto honesto: ninguna de las dos mejoras subió el F1.** Mahalanobis
queda **empatado** con z-score simple (0.76) — cambia el balance FAR/MAR
(menos misses, más falsas alarmas) pero el F1 neto es igual; con solo 8
features, capturar la correlación conjunta no aportó frente a mirar el
máximo desvío individual. El suavizado EWMA **empeoró levemente** (0.76→0.75)
en ambas variantes: sí reduce el MAR de forma notable (menos anomalías
perdidas, 14.9%→6.9% y 12.8%→4.2%), pero a costa de un FAR mucho mayor
(43-48%→63-71%) — la memoria del EWMA mantiene el score elevado por más
tiempo después de que termina una anomalía real, extendiendo períodos de
falsa alarma más de lo que gana en detección. Tiene sentido con la
naturaleza de las anomalías de SKAB (eventos mayormente puntuales/abruptos,
no derivas graduales) — el suavizado causal que sí ayuda en fraude (Dominio
1/2, donde el comportamiento normal es más ruidoso y la señal está en la
tendencia) no ayuda igual acá.

**Conclusión:** el detector más simple (z-score, 3-sigma, 3 líneas de
aritmética) sigue siendo el mejor de las 4 variantes propias, y el resultado
de F1=0.76 (3er lugar de 9, empatado con T2-q) se mantiene como el número
vigente — no se encontró una mejora real con este dataset, y se documenta
así en vez de forzar una narrativa de "sí mejoró".

## Pendiente si se decide continuar (no iniciado)

- Construir en `src/` como Dominio 2 — módulos paralelos
  (`calibrador_iot.py`, `ejecutor_iot.py`, etc.) si el usuario decide que
  vale la pena — no hecho en esta pasada, queda a su decisión.
- Explorar `changepoint` como problema de detección de cambio de régimen
  en vez de clasificación punto a punto — distinto del problema binario
  resuelto aquí.
- Investigar por qué `MSCRED` y `Vanilla_LSTM` no alinearon (probablemente
  usan una ventana de secuencia distinta al split de 400 filas) si se
  quiere una tabla completa de los 11 métodos.

Scripts de esta exploración: scratchpad de la sesión (no persistidos en el
repo). Reporte de métricas del experimento supervisado anterior:
`data/reporte_camino3_skab.json` (gitignored).
