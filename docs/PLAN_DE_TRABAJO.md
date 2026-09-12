# Plan de trabajo — SYNAPSE

**Estado general:** Fases 0 a 5 completas + auditoría post-implementación
(12 hallazgos, 10 corregidos).

## Fases (ver plan aprobado para el detalle completo de cada una)

### ⏳ Fase 0 — Fundamentos — en progreso
- [x] Estructura del repo, `CLAUDE.md`, `README.md`, `requirements.txt`.
- [x] Dataset descargado y validado: `data/raw/creditcard.csv` — 284,807
      filas, 492 fraudes (0.17%), columnas `Time`, `V1`-`V28` (PCA,
      anonimizadas), `Amount`, `Class`. Fuente: mirror de OpenML
      (`https://www.openml.org/data/get_csv/1673544/phpKo8OWT`), sin
      autenticación ni credenciales de pago.
- [x] `docs/ADR_001_artefacto_de_politica.md` — forma exacta del JSON que
      conecta Calibrador y Ejecutor.
- [x] `docs/ADR_002_capa_de_veto.md` — invariantes duros para esta
      iteración.

**⚠️ Hallazgo importante de Fase 0 — limitación de datos:** el dataset
**no tiene identificador de tarjeta/cuenta** (cada fila es una transacción
anonimizada independiente). Esto significa que el Ejecutor (Fase 2)
mantendrá estadísticos recursivos **globales** (sobre toda la población),
no por cuenta — la arquitectura de dos velocidades y la medición de
latencia siguen siendo válidas, pero se pierde la dimensión de
personalización por cuenta que un sistema de fraude real tendría. Detalle
completo y opciones en `ADR_002`.

### ✅ Fase 1 — Calibrador — completa
- [x] `src/calibrador.py`: carga y limpia el dataset (la columna `Class`
      viene con comillas literales por la conversión ARFF→CSV del mirror),
      split temporal walk-forward, regresión logística con
      `class_weight="balanced"` (0.17% de fraude).
- [x] **Decisión de diseño**: los coeficientes se des-escalan
      matemáticamente antes de guardarlos, para que el artefacto opere
      directo sobre features crudas — el Ejecutor nunca carga
      `StandardScaler` ni scikit-learn. Verificado exacto (diff < 1e-9
      contra el modelo escalado original, `test_coeficientes_desescalados_...`).
- [x] Umbral de decisión: el que maximiza F1 en el tramo de validación
      (no 0.5 fijo — con `class_weight="balanced"` el umbral óptimo real
      queda lejos de 0.5).
- [x] **Métricas reales en validación** (56,961 transacciones, 57 fraudes):
      precisión 0.46, recall 0.79, F1 0.58, **AUC 0.972**. Consistente con
      benchmarks publicados de regresión logística sobre este dataset.
- [x] 6 tests (`tests/test_calibrador.py`), incluyendo la verificación
      algebraica exacta del des-escalado.
- [x] **Corrección importante**: el modelo original solo usaba V1-V28+Amount
      — ninguna feature recursiva. Sin eso, el estado que mantendría el
      Ejecutor en Fase 2 sería decorativo (calculado, pero nunca usado en
      la decisión). Se agregó `src/features_recursivas.py` con dos features
      causales (`monto_ewma_global`, `conteo_ventana_global`) en modo batch
      y modo incremental — verificado que ambos modos producen exactamente
      los mismos números sobre la misma secuencia (`tests/test_features_recursivas.py`,
      5 tests) para descartar train/serve skew por construcción. El modelo
      recalibrado con estas dos features da AUC 0.972 (marginal sobre 0.971
      porque V1-V28 ya son muy predictivas por sí solas), con coeficientes
      no nulos para ambas — confirmado que sí aportan a la decisión.

### ✅ Fase 2 — Ejecutor — completa
- [x] `src/ejecutor.py`: `validar_artefacto` (invariante 1 de `ADR_002` —
      sin artefacto válido, no hay decisión), `Ejecutor.decidir()` —
      sigmoide numéricamente estable, aplica el artefacto sobre el estado
      leído ANTES de actualizarlo (misma semántica causal que el batch).
- [x] **Bug encontrado y corregido durante el testing, no en producción**:
      el primer intento de la prueba de consistencia end-to-end comparaba
      el Ejecutor arrancando con estado vacío justo en el corte
      train/val, contra features batch que ya cargaban el historial
      continuo desde el inicio del dataset — divergían hasta 24% en
      algunos scores. Corregido haciendo que el Ejecutor procese también
      `train` antes de evaluar `val`, igual que en producción real (el
      estado nunca se reinicia en un corte arbitrario). Con la corrección,
      coincide con el batch hasta 1e-9 (`test_ejecutor_coincide_exactamente_con_calculo_batch_end_to_end`).
- [x] **Benchmark real de latencia**: 10,000 decisiones,
      **2.15 microsegundos/decisión** medido con `time.perf_counter()`
      (`test_benchmark_latencia_real_microsegundos`) — no una estimación.
- [x] 9 tests (`tests/test_ejecutor.py`).

### ✅ Fase 3 — Puente — completa
- [x] `src/puente.py`: `publicar()` valida contra `ADR_001` antes de
      escribir (nunca publica algo inválido, ni parcialmente), escritura
      atómica (archivo temporal + `Path.replace`), `leer_vigente()` valida
      de nuevo al leer.
- [x] 6 tests (`tests/test_puente.py`): round-trip, artefacto inválido no
      escribe nada, sin archivo temporal residual, sobreescritura atómica,
      archivo faltante y archivo corrupto lanzan el error correcto.

### ✅ Fase 4 — Veto — completa
- [x] `src/veto.py`: los 3 invariantes de `ADR_002` — score inválido
      (NaN/fuera de rango/`None`) veta, monto sobre el límite absoluto
      veta **incluso si el modelo dice que no es sospechosa** (el caso que
      de verdad importa: el veto sobreescribe al modelo, nunca al revés).
- [x] 7 tests (`tests/test_veto.py`).

### ✅ Fase 5 — Validación end-to-end — completa
`scripts/validacion_end_to_end.py` corre el pipeline completo (Calibrador →
Puente → Ejecutor → Veto) sobre el tramo de **prueba** (56,962 transacciones,
75 fraudes — nunca tocado hasta ahora), con el estado recursivo construido
en orden real desde train→val→test.

**Resultado — SYNAPSE (dos capas) vs. baseline ingenuo (umbral fijo sobre
Amount, sin calibrar, sin features recursivas, sin veto):**

| | Precisión | Recall | F1 |
|---|---|---|---|
| SYNAPSE | 0.50 | 0.79 | 0.61 |
| Baseline (umbral fijo) | 0.00 | 0.00 | 0.00 |

El baseline **no detecta ni un solo fraude** de los 75 en el tramo de
prueba — verificado que no es un bug: el fraude más grande en test es de
$1,504.93, muy por debajo del umbral del percentil 99.9 de Amount en train
($3,000). Es un hallazgo real, no un artefacto de la comparación: en este
dataset el fraude se concentra en montos moderados, exactamente el patrón
que un umbral fijo sobre el monto no puede capturar y que SÍ recogen las
features V1-V28 (PCA) + las recursivas.

**Latencia real de punta a punta** (Ejecutor + Veto, 56,962 decisiones):
**8.03 microsegundos/decisión** — más que en el benchmark aislado de Fase 2
(2.15 µs) porque ahora incluye también la evaluación del Veto, pero sigue
siendo microsegundos, no milisegundos.

### Fase 6 (futura) — Extracción de contrato genérico
Solo después de un segundo dominio real construido.

## Gobernanza

- Ningún parámetro no justificado tiene valor por defecto silencioso.
- Split temporal siempre, nunca aleatorio, sobre datos con orden temporal.
- Cada actualizador/fórmula probado contra un caso numérico conocido de
  antemano.
- Todo benchmark de latencia reporta un número medido, no una estimación.

## Auditoría post-implementación (con el skill `rigorous-build`)

Con las 6 fases y 33 tests en verde, se hizo una auditoría adversarial con
ojos frescos (agente independiente, sin el contexto de quien escribió el
código) sobre los 5 módulos de `src/` y el script de Fase 5. Encontró 12
hallazgos reales; 10 corregidos, 2 quedan como menores documentados.

### Críticos (4/4 corregidos)

1. **Nada obligaba a que una decisión pasara por `veto.evaluar()`** — y
   como `nan >= umbral` es `False` en Python, un score inválido (`NaN`) se
   interpretaba como "no sospechosa" si algo usaba `Ejecutor.decidir()`
   directo, exactamente lo que `ADR_002` prohíbe. Corregido con
   `src/ciclo.py::CicloDecision` — único punto de entrada sancionado, el
   Veto siempre corre.
2. **"Revisar manualmente" por artefacto ausente/corrupto (ADR_002,
   invariante 1) no existía como código**, solo como una excepción sin
   capturar que tumbaba el proceso. `CicloDecision.decidir()` la captura y
   la convierte en una decisión real de escalamiento.
3. **Nada impedía construir un `Ejecutor` con un artefacto que nunca pasó
   por `puente.leer_vigente()`** — sin la escritura atómica ni la
   validación del Puente. `CicloDecision` solo se construye desde una ruta
   de archivo.
4. **`EstadoRecursivoGlobal.tiempos_ventana` usaba una lista con
   `.pop(0)`, O(n) no O(1)** pese a lo que decía el docstring — invisible
   en los benchmarks porque la tasa de datos (~1.6 tx/seg) nunca llenó la
   ventana lo suficiente para notarlo. Corregido con `collections.deque`
   (`popleft` O(1) real). De paso se encontró y corrigió un bug relacionado:
   un evento con `Time` fuera de orden (llegada tardía, común en streaming
   real) se quedaba atascado en medio de la ventana para siempre y
   contaminaba todos los conteos posteriores — ahora se rechaza
   explícitamente (`TiempoFueraDeOrden`).

### Importantes (6/6 corregidos)

5. Veto de "monto absoluto" no usaba `abs()` — un monto muy negativo
   (reembolso/chargeback fraudulento) nunca disparaba el veto pese a que
   `ADR_002` lo llama explícitamente "absoluto". Corregido.
6. El artefacto era mutable después de validarse — sin copia defensiva, una
   mutación externa del mismo dict (logging, monitoreo) rompía la garantía
   de "validado una vez, seguro para siempre". `Ejecutor` ahora guarda una
   copia (`copy.deepcopy`) del artefacto.
7. `validar_artefacto` solo validaba forma (longitudes, rangos), no
   contenido — un artefacto con un coeficiente no numérico o un nombre de
   feature vacío pasaba la validación y explotaba con un error crudo
   dentro del camino caliente. Ahora valida tipos explícitamente.
8. `puente.py` importaba la validación desde `ejecutor.py` — dirección de
   acoplamiento invertida (el intermediario dependía del consumidor
   rápido). Se extrajo `src/artefacto.py`, módulo neutral que comparten
   `ejecutor.py` y `puente.py`.
9. `sort_values("Time")` no era estable — con resolución de 1 segundo y
   >1.6 transacciones/segundo en promedio, hay empates frecuentes; el
   quicksort por defecto no preserva el orden original del CSV entre filas
   empatadas. Corregido con `kind="stable"`.
10. Sin manejo controlado de CSV vacío o con columnas faltantes — fallaba
    con errores crudos de pandas/numpy varias llamadas después, no en el
    punto de entrada. `cargar_dataset` ahora valida explícitamente
    (`DatasetInvalido`).

### Menor (1 corregido, 1 documentado sin código)

11. `_mejor_umbral_por_f1` no tenía protección propia si `df_val` no tiene
    ningún positivo — dependía de que `roc_auc_score` reventara como
    efecto colateral. `calibrar()` ahora valida explícitamente que
    `df_train` y `df_val` tengan al menos un caso positivo.
12. Contradicción de estilo documentada, sin corregir (impacto nulo):
    `ADR_001` describe el orden de `features` como "nunca por nombre en
    tiempo de decisión", pero `Ejecutor.decidir()` sí resuelve cada
    feature por nombre — correcto en resultado, la justificación de diseño
    escrita se contradice con la implementación. Pendiente de ajustar la
    prosa del ADR, no el código.

**Actualización de arquitectura**: `scripts/validacion_end_to_end.py` se
migró para usar `CicloDecision` en vez de invocar `Ejecutor`+`veto.evaluar()`
por separado — los mismos resultados de Fase 5 (precisión 0.50, recall
0.79, F1 0.61) se reproducen exactamente, ahora por la vía sancionada.

16 tests nuevos de esta auditoría. Suite completa: 49 tests.

## Cómo se actualiza un artefacto en producción (recalibración)

`CicloDecision.recargar_artefacto()` es el mecanismo: el Calibrador corre
de nuevo (con datos más recientes, o porque un monitoreo de deriva —
pendiente de construir, ver plan de pruebas en conversación — detectó que
el modelo se desactualizó), publica un artefacto nuevo vía `puente.publicar()`,
y `recargar_artefacto()` lo recoge — conservando el estado recursivo
acumulado (no se pierde historial por recalibrar) y **sin reemplazar el
artefacto vigente si la recarga falla** (archivo corrupto a mitad de
escritura, artefacto inválido) — degradación con gracia, nunca una caída.

## Detección de deriva (`src/deriva.py`) — el disparador de "cuándo recalibrar"

Implementa **PSI** (Population Stability Index, umbrales estándar de la
industria: <0.1 sin deriva, 0.1-0.25 moderada, >0.25 recalibrar) y el
**test de Kolmogorov-Smirnov** de dos muestras — se usan ambos porque miden
cosas distintas (PSI es el estándar de negocio, KS es más sensible como
señal temprana). 9 tests (`tests/test_deriva.py`), incluyendo control
negativo (misma distribución → PSI≈0) y positivo (desplazamiento de 5
desviaciones estándar → PSI>0.25, KS p<0.05).

**Resultado real** (`scripts/deteccion_deriva.py`, train vs. test del
dataset real): **8 de 31 features ya muestran deriva significativa** entre
los dos tramos, pese a que el dataset cubre solo ~48 horas. La más notable:
`conteo_ventana_global` (PSI=0.82) — tiene sentido, el volumen de
transacciones varía naturalmente por hora del día, y esa feature depende
directamente de la tasa de transacciones. `Amount` en cambio no muestra
deriva (PSI=0.008) — el monto promedio de las transacciones se mantuvo
estable. Confirmado con un control positivo (duplicar `Amount`
artificialmente) que el detector sí dispara cuando corresponde
(PSI=4.72, KS p≈0).

### ✅ Disparador de recalibración automática — completo

`src/disparador_recalibracion.py::evaluar_y_recalibrar_si_hace_falta()` es
el job que conecta `evaluar_deriva()` con `calibrador.calibrar()` y
`puente.publicar()`: compara la ventana de referencia (con la que se
calibró el artefacto vigente) contra una ventana reciente, y si hay deriva
significativa (PSI > 0.25 en alguna feature) recalibra sobre esa ventana
reciente y publica el artefacto nuevo. Si la ventana reciente no tiene
casos positivos suficientes para recalibrar con seguridad (`DatasetInvalido`),
**no publica nada** — el artefacto vigente queda intacto y el error se
reporta para revisión humana, en vez de degradar el modelo en silencio con
una calibración mala. La recarga en vivo (`CicloDecision.recargar_artefacto()`)
sigue siendo responsabilidad de quien tiene el ciclo corriendo — este
módulo solo publica, nunca reemplaza el estado recursivo del Ejecutor.

`scripts/recalibracion_automatica.py` corre el flujo completo sobre datos
reales: publica el artefacto inicial (train/val), evalúa deriva train vs.
test, y si corresponde recalibra y llama `recargar_artefacto()` sobre un
`CicloDecision` ya en marcha. **Resultado real**: detecta las mismas 8/31
features con deriva significativa que `deteccion_deriva.py`, recalibra,
publica, y la recarga en vivo confirma éxito (`True`) sin perder el estado
recursivo acumulado.

5 tests nuevos (`tests/test_disparador_recalibracion.py`): sin deriva no
recalibra ni publica, con deriva significativa recalibra y publica, deriva
detectada pero ventana sin casos positivos no publica y reporta el error,
y el artefacto vigente no se sobreescribe si la recalibración falla. Suite
completa: 61 tests.

**Lo que sigue pendiente, y es un problema de datos/infraestructura de
producción, no de este módulo**: la ventana "de referencia" y la ventana
"actual" hoy se pasan como parámetros explícitos (train/test del dataset
histórico) — falta que un proceso real en producción arme esas dos
ventanas desde tráfico en vivo (ej. últimos N días vs. los N días
anteriores) y programe la ejecución periódica (Task Scheduler / cron /
job en la nube).

## Pendiente del plan de pruebas más amplio (no bloqueante)

Quedan por construir, del plan de pruebas discutido en conversación: umbral
de decisión por costo esperado (no solo F1), validación cruzada temporal
con múltiples folds walk-forward, prueba de carga/concurrencia real del
Ejecutor, y el dataset con identificador de cuenta (IEEE-CIS) para probar
personalización por entidad.
