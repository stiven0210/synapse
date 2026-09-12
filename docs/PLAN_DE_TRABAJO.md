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
      queda lejos de 0.5, de hecho por encima de 0.99 — ver corrección más
      abajo, "Hallazgo real: grilla de umbral topada en 0.99").
- [x] **Métricas reales en validación** (56,961 transacciones, 57 fraudes):
      precisión 0.93, recall 0.70, F1 0.80, **AUC 0.972**. Consistente con
      benchmarks publicados de regresión logística sobre este dataset.
      (Corregido tras el hallazgo de la validación cruzada walk-forward —
      antes de la corrección: precisión 0.46, recall 0.79, F1 0.58, mismo AUC.)
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
| SYNAPSE | 0.93 | 0.67 | 0.78 |
| Baseline (umbral fijo) | 0.00 | 0.00 | 0.00 |

(Corregido tras el hallazgo de la validación cruzada walk-forward — antes
de la corrección: precisión 0.50, recall 0.79, F1 0.61. Ver "Hallazgo real:
grilla de umbral topada en 0.99" más abajo.)

El baseline **no detecta ni un solo fraude** de los 75 en el tramo de
prueba — verificado que no es un bug: el fraude más grande en test es de
$1,504.93, muy por debajo del umbral del percentil 99.9 de Amount en train
($3,000). Es un hallazgo real, no un artefacto de la comparación: en este
dataset el fraude se concentra en montos moderados, exactamente el patrón
que un umbral fijo sobre el monto no puede capturar y que SÍ recogen las
features V1-V28 (PCA) + las recursivas.

**Latencia real de punta a punta** (Ejecutor + Veto, 56,962 decisiones):
**8.3 microsegundos/decisión** — más que en el benchmark aislado de Fase 2
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

### ✅ Bitácora de decisiones — completa (prerequisito del triage de Veto)

Antes de evaluar si vale la pena un agente de triage sobre escalamientos de
Veto (ver análisis en conversación), hacía falta cerrar un hueco real:
`CicloDecision.decidir()` devuelve una `DecisionFinal` pero nada la
persistía — no había sobre qué investigar un escalamiento después del
hecho.

`src/bitacora_decisiones.py` — registro append-only (JSON Lines) de cada
decisión, con `clasificar_razon()` distinguiendo 4 tipos a partir del texto
real que producen `veto.py`/`ciclo.py`:

- `MODELO` — camino normal, no necesita triage.
- `MONTO_EXCEDE_LIMITE` — invariante de negocio, autoexplicativo en el
  propio dato (el monto), no necesita triage.
- `SCORE_INVALIDO` y `ERROR_EJECUTOR` — **los únicos dos candidatos reales
  a triage**: representan una falla de *sistema* (el modelo no pudo
  producir una decisión), no un juicio sobre una transacción.

Deliberadamente **no se invoca desde `CicloDecision.decidir()`** — el
camino caliente está medido en microsegundos y no debe pagar I/O a disco
en cada decisión; registrar es responsabilidad explícita de quien tiene el
ciclo corriendo, mismo principio que la recarga del artefacto no es
automática.

**Acoplamiento documentado explícitamente**: `clasificar_razon()` distingue
por el texto de `razon`, no por un campo estructurado en `DecisionFinal`
(no se tocó `veto.py` — ya pasó 2 rondas de auditoría). Si el texto cambia,
`test_clasificar_razon_cubre_los_textos_reales_de_veto_y_ciclo` (llama a
`veto.evaluar()` de verdad, no copia los strings a mano) rompe en vez de
misclasificar en silencio.

**2 escenarios de causa real conocida** (`tests/test_bitacora_decisiones.py`),
de punta a punta por el camino real (Ejecutor → Veto → CicloDecision →
bitácora, sin monkeypatchear nada, a diferencia de `test_ciclo.py`):

1. Transacción con una feature en `NaN` (dato corrupto aguas arriba) → el
   Ejecutor no lanza excepción pero el score resultante es NaN → se
   clasifica `SCORE_INVALIDO`.
2. Transacción a la que le falta una feature que el artefacto vigente
   espera (desajuste de esquema fuente-vs-artefacto) → `KeyError` dentro
   del Ejecutor → `CicloDecision` lo captura → se clasifica `ERROR_EJECUTOR`.

Estos 2 escenarios son la base para, más adelante, verificar si un agente
de triage señala la causa correcta — ver el plan de pruebas de 3 niveles
discutido en conversación (Nivel 1: guardrails deterministas con casos
conocidos; Nivel 2: causas simuladas como estas; Nivel 3: valor real solo
verificable con uso en producción, no offline).

9 tests nuevos. Suite completa: 70 tests.

### ✅ Agente de triage de escalamientos de Veto — completo

`docs/ADR_003_agente_triage_veto.md` documenta la decisión completa.
`src/agente_triage.py` implementa la misma disciplina de 3 capas del
agente de an earlier project (código nuevo e independiente, no importado —
`CLAUDE.md`): Capa 1 (schema determinista), Capa 2 (grounding determinista
contra los datos reales de la entrada/contexto), Capa 3 (auditor selectivo,
desacuerdo nunca se resuelve por mayoría). Nunca se invoca desde
`CicloDecision.decidir()` — corre después, sobre entradas de tipo
`SCORE_INVALIDO`/`ERROR_EJECUTOR` de la bitácora. Nunca decide ni bloquea
nada, solo propone una hipótesis para que un humano la verifique.

`src/limitador_llamadas.py` — rate limiting diario (mismo patrón que
`an earlier project/src/agents/rate_limiter.py`, código independiente). `CircuitoTriage`
(en `agente_triage.py`) — circuit breaker: si la tasa de descarte de los
últimos N triages supera el umbral, el agente se apaga solo antes de gastar
en una llamada más (`CircuitoAbierto`), el llamador cae al reporte plano
determinista.

**Validado según el plan de pruebas de 3 niveles:**
- Nivel 1 (cliente LLM falso, sin llamadas reales): schema inválido se
  descarta en Capa 1, grounding fallido dispara Capa 3, desacuerdo
  proponente/auditor descarta sin resolver por mayoría, circuit breaker se
  abre con tasa de descarte simulada y no gasta una llamada más una vez
  abierto.
- Nivel 2 (causa real conocida, reusando los 2 escenarios de
  `test_bitacora_decisiones.py`): una hipótesis que cita el dato real
  (`Amount` en NaN) pasa Capa 2 sin necesitar Capa 3; una hipótesis
  alucinada (causa inventada, sin respaldo en los datos reales) falla
  Capa 2, dispara Capa 3, y se descarta antes de llegar a un humano como
  conclusión confiable.
- Nivel 3 (valor real de uso): explícitamente sin validar — requiere
  incidentes reales en producción, no un dataset histórico offline.

20 tests nuevos (15 del agente + 5 del limitador). Suite completa: 90 tests.

### ✅ Job de triage de punta a punta — completo

`scripts/triage_veto.py` cierra el cableado operativo: lee la bitácora
real, filtra escalamientos operativos, arma el contexto (reporte de deriva
más reciente si existe), invoca `AgenteTriage` a través del rate limiter, y
muestra el resultado a un humano (consola + `data/reporte_triage_veto.json`).

**Sin tráfico real de producción disponible**, el job primero alimenta la
bitácora con los mismos 2 escenarios de causa real conocida de
`tests/test_bitacora_decisiones.py` (feature en NaN, feature faltante, y
una decisión normal de control) — así hay algo real que triar en vez de
partir de un archivo vacío.

**Degradación con gracia, verificada con una corrida real**: sin
`ANTHROPIC_API_KEY` en el entorno, el job no inventa una llamada falsa —
reporta explícitamente que no puede triar con un LLM real y cae al reporte
plano (cada escalamiento operativo tal cual, sin hipótesis). `CircuitoAbierto`
y `PresupuestoAgotado` se manejan igual si ocurren con un cliente real
configurado: se cae al reporte plano para esa entrada, nunca se detiene el
job completo.

**Actualización: probado con `ANTHROPIC_API_KEY` real — 2 bugs reales
encontrados y corregidos, ninguno detectable con el cliente falso de los
tests.**

1. **`crear_cliente_claude()` asumía `respuesta.content[0]` como texto.**
   El modelo puede devolver primero un bloque de razonamiento extendido
   (`ThinkingBlock`), y `.content[0].text` lanzaba `AttributeError`.
   Corregido: se buscan explícitamente los bloques de tipo `"text"` entre
   todo `respuesta.content`, sin asumir posición.
2. **El modelo envuelve el JSON en un bloque de código markdown**
   (` ```json ... ``` `) pese a que el prompt pide "Responde SOLO en
   JSON" — comportamiento común de LLMs, no un error del modelo.
   `_capa1_validar_schema` lanzaba `FalloCapa1` con
   `"Expecting value: line 1 column 1"` (JSON vacío tras el fence sin
   despojar). Corregido con `_despojar_bloque_markdown()`.

4 tests nuevos reproduciendo ambos casos con un cliente Anthropic
simulado (sin llamada de red real). Suite completa: 110 tests.

**Resultado real, de punta a punta, con Claude real** (los mismos 2
escenarios de causa conocida): ambas hipótesis correctas y bien fundamentadas,
pasando Capa 2 sin necesitar Capa 3 —
- `SCORE_INVALIDO` (Amount en NaN): *"El valor de Amount llegó como NaN...
  No hay evidencia en el contexto de que esto se deba a deriva de
  distribución"* — severidad media, `investigar_pipeline_datos`, confianza 0.55.
- `ERROR_EJECUTOR` (Amount faltante): *"El ejecutor falló porque el objeto
  'transaccion' no contiene el campo 'Amount'... probablemente causó un
  KeyError"* — severidad alta, `verificar_esquema_transaccion`, confianza 0.75.

Esto cierra la validación de Nivel 2 con un LLM real (antes solo probado
con cliente falso) — la calidad y el grounding de las hipótesis son
correctos en este caso concreto.

**Pendiente, explícitamente fuera de este alcance**: no existe todavía una
fuente de tráfico de producción real que alimente la bitácora — hoy solo
el job de ejemplo la alimenta con los 2 escenarios conocidos. El Nivel 3
(¿esto reduce de verdad el esfuerzo de un humano?) sigue sin poder
validarse sin incidentes reales en producción.

### ✅ Monitoreo de deriva del score de salida — completo

`evaluar_deriva()` solo miraba deriva por feature de entrada — un shift
pequeño repartido en muchas features (cada una por debajo de PSI 0.25)
puede mover el score combinado del modelo sin que ninguna lo muestre sola,
justo lo que el monitoreo por feature, por diseño, no puede ver.

`src/artefacto.py::calcular_scores()` — versión batch vectorizada de la
misma fórmula que `Ejecutor.decidir()` aplica incrementalmente (verificado
idéntico hasta 1e-9 en `tests/test_artefacto.py`, mismo principio de
train/serve parity que el resto del proyecto). Solo para diagnóstico
offline, nunca en el camino caliente.

`src/deriva.py::evaluar_deriva_score()` — PSI+KS sobre el score en vez de
una feature. `disparador_recalibracion.py` ahora combina ambas señales:
recomienda recalibrar si CUALQUIERA es significativa (feature o score),
usando el artefacto vigente para calcular los scores.

**Caso construido a mano que demuestra el valor real** (`tests/test_deriva.py`):
un shift de 0.11 desviaciones estándar repartido en 30 features dejó a
CADA UNA muy por debajo de PSI 0.25, pero el score combinado (media de las
30) cruzó a PSI=0.35. Reproducido también a través del disparador completo
(`tests/test_disparador_recalibracion.py`): 0 features cruzan solas, el
score sí, y el disparador recalibra igual.

**Resultado real, dirección contraria — igual de valioso**: sobre el
dataset real, 8/31 features muestran deriva significativa mientras el PSI
del score combinado da solo 0.07 (sin deriva significativa) — confirma que
las dos señales son genuinamente complementarias, no redundantes; pueden
discrepar en cualquier dirección.

También se eliminó una duplicación real: `scripts/comparacion_umbral_por_costo.py`
tenía su propia función de scoring copiada — ahora usa
`artefacto.calcular_scores()`, una sola fuente de verdad para "cómo se
aplica un artefacto a un DataFrame completo".

8 tests nuevos (4 en `test_artefacto.py`, 3 en `test_deriva.py`, 1 de
integración en `test_disparador_recalibracion.py`). Suite completa: 118 tests.

### ✅ Deriva ponderada por magnitud de coeficiente — completo

`evaluar_deriva()` trataba todas las features igual: cualquiera que
superara PSI 0.25 contaba lo mismo para `recomendacion_recalibrar`, sin
distinguir una feature que el modelo realmente pesa de una que casi no usa.
`deriva.ponderar_deriva_por_coeficiente()` enriquece el reporte con la
magnitud de coeficiente de cada feature en el artefacto vigente y calcula
`contribucion_ponderada_features_con_deriva` (0 a 1: qué fracción del peso
total del modelo está en las features que sí derivaron).

**Deliberadamente no reemplaza `recomendacion_recalibrar`** — cambiar el
disparador automático con un umbral nuevo sobre la contribución ponderada
sería un parámetro no justificado empíricamente. Es información adicional
para juicio humano, no una nueva regla automática.

`disparador_recalibracion.py` la calcula automáticamente usando los
coeficientes del artefacto vigente (antes de una posible recalibración) —
si no hay artefacto legible todavía, queda en `None` sin bloquear nada.
`scripts/recalibracion_automatica.py` la reporta.

**Resultado real** (mismo par train/test de `deteccion_deriva.py`): de las
8/31 features con deriva significativa, solo el **23%** del peso total del
modelo (suma de |coeficiente|) está en esas features — la deriva encontrada
no está concentrada en lo que el modelo más pesa.

5 tests nuevos (4 en `test_deriva.py`, incluyendo caso numérico armado a
mano y protección contra división por cero; 1 en
`test_disparador_recalibracion.py`). Suite completa: 95 tests.

### ✅ Validación cruzada walk-forward multi-fold — completo

`src/validacion_cruzada.py` responde si el AUC/umbral reportado en Fase 1
depende de qué corte particular se usó, o si el proceso de calibración es
estable en el tiempo. Ventana expansiva (nunca K-fold aleatorio, que
filtraría futuro hacia el pasado). Diagnóstico únicamente — no reemplaza
`split_temporal()`+`calibrar()`, que siguen siendo la única vía de
producción (solo puede existir un artefacto vigente a la vez).

**Resultado real** (`scripts/validacion_cruzada_walk_forward.py`, 5 folds
sobre el dataset real): AUC = 0.9775 ± 0.0066 entre folds — consistente y
estable, el 0.972 de un único split **no es un accidente de corte**.

**Hallazgo real durante esta validación: grilla de umbral topada en 0.99.**
Los 5 folds dieron el umbral óptimo exactamente en el límite de la grilla
de búsqueda (`np.linspace(0.01, 0.99, 99)` en `calibrador._mejor_umbral_por_f1`),
sin variación — señal de que el óptimo real quedaba fuera del rango
explorado, no de que 0.99 fuera genuinamente el mejor valor. Verificado a
mano: sobre el tramo de validación real, el F1 seguía subiendo de 0.58 (en
0.99) a 0.79 (en 0.99999). Causa: con `class_weight="balanced"` y
separación fuerte entre clases, las probabilidades se concentran cerca de
0 y 1.

**Corregido**: `_mejor_umbral_por_f1` ahora busca sobre los scores
realmente observados vía `precision_recall_curve` (O(n log n), umbral
óptimo exacto, no una aproximación de grilla) en vez de una grilla fija.
Esto **cambió las métricas reales reportadas en Fase 1 y Fase 5** (ver esas
secciones, ya actualizadas) — precisión subió de ~0.50 a ~0.93, F1 de ~0.61
a ~0.78 (recall bajó de ~0.79 a ~0.67, el balance neto mejora). El AUC no
cambia (es independiente del umbral) — sirvió como control de que la
corrección no tocó nada más.

6 tests nuevos (5 en `test_validacion_cruzada.py` + 1 caso numérico armado
a mano en `test_calibrador.py` que reproduce exactamente el bug: scores de
ambas clases por encima de 0.99, donde la grilla vieja fuerza F1=0.57 y la
nueva encuentra la separación perfecta). Suite completa: 101 tests.

### Umbral de decisión por costo esperado — construido con un supuesto de negocio explícito, sin reemplazar producción

Pendiente real (no se puede resolver sin datos de negocio que este proyecto
no tiene): el costo de un falso positivo (fricción/revisión de una
transacción legítima bloqueada) no está en ningún dataset público. El
costo de un falso negativo (fraude no detectado) sí — es el `Amount` real
de la transacción no detectada, no un promedio inventado.

`src/costo_decision.py` (`mejor_umbral_por_costo()`) **exige explícitamente**
`costo_falso_positivo` como parámetro, sin default — pasar un número
inventado como si fuera un dato validado violaría "ningún parámetro no
justificado tiene valor por defecto silencioso" (`CLAUDE.md`). **No
reemplaza el umbral de F1 que usa `calibrar()`** (el único justificado con
datos reales que este proyecto tiene) — es una herramienta de comparación,
no un cambio de producción.

`scripts/comparacion_umbral_por_costo.py` corre la comparación sobre el
tramo de prueba real con un supuesto **ilustrativo** de $5 por falso
positivo (marcado explícitamente en el código y en el reporte como no
validado). **Resultado real**: el umbral por costo (0.985) da F1 más bajo
que el de producción (0.61 vs 0.78 — precisión cae de 0.94 a 0.49) pero
**reduce el costo total esperado de 3,352 a 2,713** — captura más fraude
real (recall 0.67 → 0.80) porque, bajo ese supuesto, un falso positivo es
barato frente al costo de dejar pasar un fraude. Demuestra concretamente
que optimizar F1 y optimizar costo esperado no son lo mismo — la decisión
de cuál usar en producción sigue pendiente de un costo de falso positivo
real, no inventado.

11 tests nuevos desde el commit anterior (5 en `test_costo_decision.py`,
incluyendo el caso numérico armado a mano que demuestra la divergencia
F1-vs-costo; 5 en `test_validacion_cruzada.py`; 1 en `test_calibrador.py`
para la corrección del umbral de F1). Suite completa: 106 tests.

## Pendiente del plan de pruebas más amplio (no bloqueante)

El dataset IEEE-CIS (identificador de cuenta, para probar personalización
por entidad) sigue bloqueado — requiere la cuenta de Kaggle del usuario,
unirse a la competencia y un token de API real; no hay forma honesta de
sustituir eso con un supuesto.

### Hallazgo de la investigación de concurrencia — documentado, sin corrección de código

Se probó `CicloDecision`/`Ejecutor` bajo llamadas concurrentes reales desde
múltiples hilos (16 hilos, 300 llamadas c/u, con `time.sleep(0)` forzando
cesión del GIL para maximizar interleaving) — sin excepciones ni corrupción
observada en las corridas. Pero eso no es una garantía: `EstadoRecursivoGlobal`
no tiene ningún tipo de sincronización, y su corrección depende de que las
transacciones lleguen **en un único stream secuencial, en orden de `Time`**
(`TiempoFueraDeOrden` lo asume explícitamente). Que el GIL de CPython no haya
expuesto una corrupción visible en estas corridas no prueba que sea seguro
bajo otra carga/hardware/versión de Python.

**Conclusión, sin cambiar código:** `CicloDecision` está diseñado para un
único hilo procesando un stream secuencial — nunca se documentó
explícitamente esta restricción. No hace falta agregar locking: el punto
entero de decisiones O(1) en microsegundos es que un solo hilo ya cubre
volúmenes reales de fraude con margen enorme (el dataset promedia ~1.6
tx/seg; incluso a 10 microsegundos/decisión, un hilo cubre >90,000 tx/seg).
Si alguna vez hace falta paralelismo real, la forma correcta es particionar
por cuenta/entidad (cuando exista ese identificador, ver limitación de
Fase 0), nunca compartir una misma instancia de `Ejecutor` entre hilos.
Pendiente: dejar esta restricción explícita en `CLAUDE.md`/`ejecutor.py`
(no se tocó código de producción en esta ronda, solo se investigó y se
documenta el hallazgo).

### ✅ Runner diario — el "despliegue real, aunque sea chico" — completo

Sin tráfico real de producción, `src/runner.py::correr_ciclo_diario()`
simula uno real: avanza por el dataset histórico en lotes (uno por
invocación), con estado recursivo persistente entre corridas (igual que un
servicio real que se reinicia todos los días). Primera corrida hace
bootstrap (calibra con el primer `frac_bootstrap` del dataset, alimenta el
estado recursivo con esa historia sin registrarla como decisión nueva);
corridas siguientes retoman, evalúan deriva/recalibran
(`disparador_recalibracion.py`) contra el lote anterior, y trían los
escalamientos operativos nuevos si hay un agente configurado.

`scripts/runner_diario.py` es el punto de entrada para Task Scheduler/cron
(comandos de registro incluidos en su docstring). **Corrida real,
verificada**: procesó 10,000 filas reales del dataset en 2 invocaciones
manuales (índice 90,442 → 95,442 → 100,442 de ~284,807). Deliberadamente
**no se programó** en Task Scheduler — el usuario decidió dejarlo como
herramienta manual por ahora; programarlo es lo que le daría al Nivel 3 del
agente de triage (¿de verdad ahorra esfuerzo humano?) datos reales que
evaluar con el tiempo, pero eso implicaría un proceso corriendo sin
supervisión y gastando presupuesto de LLM real si se configura
`ANTHROPIC_API_KEY` en el entorno del sistema.

11 tests nuevos (`tests/test_runner.py`).

## Auditoría de los agentes (con ojos frescos, agente independiente)

Con el runner y el agente de triage recién construidos, se pidió una
auditoría adversarial de todo lo relacionado a agentes —
`agente_triage.py`, `limitador_llamadas.py`, `bitacora_decisiones.py`,
`runner.py`, y los 2 scripts de entrada — con la misma metodología que las
auditorías anteriores (agente sin contexto de quien escribió el código).

**Confirmado sólido**: ningún LLM es alcanzable desde `CicloDecision.decidir()`
(verificado por grep, no solo lectura); Capa 1/Capa 2/Capa 3 corren
exactamente cuando deben; el desacuerdo proponente/auditor nunca se
resuelve por mayoría; la serialización del estado recursivo hace
round-trip exacto; sin estado mutable compartido entre instancias.

**5 hallazgos reales, los 5 corregidos:**

1. **🔴 Crítico — un crash entre registrar una decisión y persistir el
   nuevo índice del runner duplicaba filas en la bitácora.** Reproducido
   de punta a punta antes de corregir: simulando la interrupción exacta,
   la siguiente corrida reprocesaba y re-registraba la misma fila.
   Corregido: el estado se guarda fila por fila (no una vez por lote), y
   cada `registrar_decision()` lleva un `indice_fila` explícito — al
   reanudar, si la fila que toca ya está en la bitácora (por un intento
   interrumpido anterior), se vuelve a pasar por `decidir()` (continuidad
   causal del estado recursivo) pero no se vuelve a registrar.
2. **🟡 Capa 2 (grounding) aprobaba gratis una hipótesis sin evidencia
   citada.** `evidencia_citada: []` devolvía "grounded" automáticamente —
   invertía el incentivo (no citar nada era más seguro para una hipótesis
   mala que citar algo verificable). Corregido: evidencia vacía ahora
   cuenta como grounding fallido, fuerza Capa 3.
3. **🟡 `leer_bitacora()` no toleraba una línea final truncada**, pese a
   que su propio docstring decía que sí — una escritura interrumpida
   (el mismo escenario del hallazgo 1) hacía que `json.loads`
   reventara y **ninguna** entrada se pudiera leer. Corregido: cada línea
   se parsea en su propio try/except, una corrupta se omite sin invalidar
   el resto.
4. **🟡 Solo se capturaban 2 tipos de excepción alrededor del triage**
   (`CircuitoAbierto`, `PresupuestoAgotado`) — un error real de red/API
   (timeout, 5xx) no capturado tumbaba todo el job, incluso después de que
   ya se habían guardado decisiones y estado reales. Corregido en
   `runner.py` y `scripts/triage_veto.py`: cualquier excepción del cliente
   LLM durante el triage se cuenta como fallida y el job/reporte
   continúa — el triage es asesor, nunca debe tumbar lo que ya funcionó.
5. **🟡 El presupuesto diario del rate limiter vivía solo en memoria**,
   pero los 2 puntos de entrada reales son procesos de un solo uso (por
   diseño, para Task Scheduler: "una invocación = un día") — un reintento
   manual el mismo día obtenía presupuesto fresco. Corregido:
   `LimitadorLlamadasDiarias` acepta `ruta_estado` opcional y persiste
   fecha/contador a disco entre procesos; ambos scripts de entrada ahora
   lo usan, compartiendo el mismo presupuesto diario real.

9 tests nuevos reproduciendo cada hallazgo antes de corregirlo. Suite
completa: 132 tests.
