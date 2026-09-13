# Dominio 2 — Personalización por cuenta (investigación en curso)

**Estado:** Exploración y validación completas con resultados reales. Integración
de producción (artefacto de árboles + Ejecutor compatible) pendiente de construir.

**Por qué existe esto:** Fase 0 (Dominio 1) documentó una limitación real: el
dataset de Credit Card Fraud no tiene identificador de cuenta, así que el
Ejecutor solo puede mantener estadísticos **globales**, nunca personalización
por cuenta. Esta investigación resuelve esa limitación con un segundo dataset
real, y de paso responde si vale la pena ir más allá del modelo lineal.

## 1. Dataset: Sparkov (sintético, con identificador de cuenta)

Generado localmente con [`Sparkov_Data_Generation`](https://github.com/namebrandon/Sparkov_Data_Generation)
(sin Kaggle, sin cuenta, sin verificación — alternativa real cuando IEEE-CIS
quedó bloqueado por soporte de Kaggle). Parámetros: 100 clientes,
2013-01-01 a 2026-09-12, semilla 42.

Combinado y reducido a las columnas útiles (61 archivos crudos → 1 CSV):
`data/raw/sparkov_2013_2026.csv` (119MB, gitignored igual que `creditcard.csv`).

- **1,170,945 transacciones, 99 cuentas únicas** (`cc_num`), 907 fraudes (0.077%).
- Columnas: `cc_num`, `amt`, `unix_time`/`trans_date`/`trans_time`, `category`,
  `is_fraud`, `lat`/`long` (cliente), `merch_lat`/`merch_long` (comercio).
- **Esquema totalmente distinto al Dominio 1** — no tiene V1-V28 (PCA), tiene
  features crudas/categóricas. `calibrador.py` (Dominio 1) no aplica tal cual;
  esto es un segundo dominio real, no un cambio de archivo.
- **Distancia geográfica cliente-comercio: descartada.** Probada explícitamente
  (76.5 km promedio en fraude vs. 75.1 km en legítimas) — sin señal en este
  generador. No se volvió a intentar.
- **12 de 99 cuentas son "perfiles de fraude" puros** (100% de sus 7-12
  transacciones son fraude — identidades sintéticas dedicadas, no clientes
  víctimas). Representan 119 de 907 fraudes. Toda validación de aquí en
  adelante se corrió **con y sin** estas cuentas para no confundir "detectar
  fraude" con "reconocer una cuenta ya conocida como mala".

## 2. Hallazgo real: trampa de no-estacionariedad en `category`

Con un one-hot ingenuo de `category` + split temporal 60/20/20, el AUC
combinado caía **por debajo de 0.5** (peor que azar). Causa encontrada y
verificada, no solo sospechada:

| Período | Transacciones `personal_care` | Fraudes |
|---|---|---|
| 2013-2023 (11 años) | ~24 en total | **todas** |
| 2025-2026 | 82,456 | 4 |

De 2013 a 2023 la categoría casi no tiene actividad legítima simulada; las
pocas apariciones tempranas son inyecciones de fraude (el generador puede
etiquetar fraude con cualquier categoría, sin importar si esa categoría tiene
volumen legítimo en ese momento). El modelo aprende "personal_care = fraude"
de esas ~24 filas y esa regla falla catastróficamente cuando la categoría se
vuelve normal en 2025-2026.

**Lección general (no específica de esta categoría):** con solo 99 cuentas en
13 años, cualquier categoría cuya actividad legítima no esté distribuida de
forma estable en el tiempo es una trampa para one-hot + split temporal —
mismo principio de fondo que "universo point-in-time" en cualquier sistema que
mira una distribución histórica no estacionaria como si fuera constante.

**Decisión:** `category` cruda queda fuera del modelo. Pendiente si se retoma:
una codificación point-in-time (tasa de fraude por categoría con ventana
móvil reciente, no un one-hot fijo ajustado una sola vez).

## 3. Feature nueva: huella de comportamiento por cuenta

`huella_categoria_cuenta`: de las últimas K=20 transacciones **de esa cuenta**
(causal, ventana deslizante por conteo — mismo patrón que `conteo_ventana_global`
pero por cuenta y por categoría en vez de por tiempo), qué fracción fueron en
la misma categoría que la transacción actual. Cuentas con <5 transacciones
previas usan la frecuencia poblacional como respaldo (0.04% de las filas) en
vez de un valor inventado.

**Validado, no solo diseñado:**
- Estable en el tiempo (promedio anual 0.98-0.99 en los 14 años) — no
  reproduce la trampa de la sección 2.
- Coeficiente negativo en regresión logística: categoría inusual *para esa
  cuenta específica* → más probable fraude (la intuición correcta).
- Con regresión logística: AUC-ROC pasó de 0.976 a 0.998 al agregarla.

## 4. AUC-ROC vs. AUC-PR — por qué el primer número engañaba

Benchmarks publicados reales: AUC-ROC 0.86-0.92 típico, **AUC-PR 0.51-0.66**
típico (la industria usa AUC-PR, no AUC-ROC, porque con fraude tan
desbalanceado el AUC-ROC se ve inflado fácilmente).

Con regresión logística (todas las features, incluida la huella):

| | VAL | TEST |
|---|---|---|
| AUC-ROC | 0.998 | 0.998 |
| **AUC-PR** | **0.161** | **0.263** |

**Por debajo** del benchmark publicado (0.16-0.26 vs. 0.51-0.66), pese al
AUC-ROC casi perfecto. El modelo lineal se estaba quedando corto.

## 5. Gradient Boosting — el techo era el modelo, no los datos

Diagnóstico: mismas features, mismo split, `HistGradientBoostingClassifier`
(sklearn, sin dependencia nueva) en vez de regresión logística.

| | AUC-PR (con perfiles de fraude) | AUC-PR (sin perfiles de fraude) |
|---|---|---|
| Regresión logística | 0.161 / 0.263 (val/test) | 0.144 / 0.198 |
| **Gradient Boosting** | 0.953 / 0.932 | 0.945 / 0.924 |

Prácticamente sin cambio al quitar las 12 cuentas triviales — confirma que es
señal real de comportamiento anómalo en cuentas normales, no memorización de
identidades de cuenta conocidas.

**En números de negocio (mejor umbral por F1, sin perfiles de fraude):**

| | VAL | TEST |
|---|---|---|
| Fraudes reales | 154 | 200 |
| Atrapados | 139 (90.3%) | 180 (90.0%) |
| Falsas alarmas | 24 | 29 |
| Precisión | 85.3% | 86.1% |

Muy por encima del benchmark publicado de AUC-PR (0.51-0.66) — con la
salvedad honesta de que sigue siendo un dataset **sintético**: ya se sabe que
el fraude simulado de Sparkov tiene una separación más limpia que el fraude
real (monto promedio 8x mayor en fraude). Esto valida la ingeniería de
features y que el modelo lineal era el cuello de botella, no una promesa de
desempeño idéntico contra fraude real adversarial.

## 6. Exportación a aritmética pura — validado exacto, no aproximado

Pregunta abierta: Gradient Boosting requiere una librería de ML en tiempo de
decisión, lo cual choca con el principio central del Ejecutor (aritmética
pura, sin librería pesada en el camino caliente).

**Resuelto:** se extrajo la estructura interna de los árboles de
`HistGradientBoostingClassifier` (`_predictors[i][0].nodes`, un array
estructurado con `feature_idx`, `num_threshold`, `left`, `right`, `is_leaf`,
`value`, `missing_go_to_left`) y se escribió un evaluador en Python puro
(comparaciones + sumas + un sigmoide, sin sklearn). Comparado contra
`predict_proba()` del modelo original: **diferencia máxima 2.8×10⁻¹⁷** —
idéntico, no una aproximación con pérdida.

**Implica:** Gradient Boosting sí es compatible con el Ejecutor, a costo de
ingeniería (no de arquitectura). Pendiente de construir como producción:

1. Nuevo formato de artefacto (lista de árboles serializada en JSON, en vez
   de `coeficientes`/`intercepto`).
2. Nuevo Ejecutor que recorra árboles en vez de producto punto — mismo
   contrato `decidir(transaccion) -> dict`.
3. Probablemente un `CicloDecision` paralelo (no tocar el ya auditado dos
   veces para una necesidad que solo tiene este dominio).
4. Tests verificando exactitud byte a byte contra sklearn, igual que la
   validación manual de esta sección.

### Latencia del evaluador (2026-09-13) — por debajo de Dominio 1, muy por encima de la industria

Medido con la misma metodología que `scripts/validacion_end_to_end.py`
(decisión por decisión, no por lotes), sobre los 175 árboles del modelo final
(`max_depth=3, learning_rate=0.05, max_iter=200`):

| Evaluador | Latencia/decisión |
|---|---|
| Ejecutor Dominio 1 (producto punto, referencia) | 8.33 µs |
| Árboles — nodos como diccionarios de Python | 157.3 µs |
| Árboles — **código generado** (umbrales horneados como literales, compilado una sola vez con `compile()`/`exec()`, cero diccionarios en el camino caliente) | **39.6 µs** |

La versión con diccionarios es 19x más lenta que Dominio 1; la de código
generado la deja en ~4.75x — la mejora (4x) viene de eliminar el *overhead*
de acceso a diccionarios en tiempo de decisión, no de cambiar el algoritmo.
Exactitud verificada igual en ambas: diferencia máxima 1.39×10⁻¹⁷ vs.
`predict_proba()` de sklearn.

**Conclusión honesta:** por debajo del estándar interno (Dominio 1), pero
39.6 µs = 0.04 ms sigue siendo ~2,500x más rápido que una ventana de
autorización típica de redes de tarjetas (decenas de milisegundos) — no es
un bloqueante real para producción, solo una diferencia de elegancia/paridad
interna. Reportes en `data/reporte_latencia_arboles_dominio2.json` (versión
diccionarios) y `data/reporte_latencia_arboles_codegen.json` (versión
código generado). El generador de código sigue siendo exploratorio (script
de una sola corrida, no un módulo de `src/` todavía) — si se retoma la
integración de producción, el punto 2 de la lista de arriba debería usar
este enfoque de código generado, no el de diccionarios.

### Segunda ronda de optimización (2026-09-13) — sin chequeo de NaN, y tabla de búsqueda (fallida)

**Quitar el chequeo de NaN de cada nodo:** las 5 features nunca son NaN en
este pipeline (verificado con `assert` antes de confiar en el resultado, no
solo supuesto) — el chequeo era código muerto. Resultado: **39.6 µs → 30.5 µs**
(1.3x más), acumulado 5.16x más rápido que la versión de diccionarios,
exactitud igual (1.39×10⁻¹⁷). Reporte en
`data/reporte_latencia_arboles_codegen_sin_nan.json`.

**Tabla de búsqueda precomputada (probada, descartada tal cual):** idea —
discretizar las 5 features en bins por cuantiles, evaluar el modelo una sola
vez por combinación (393,216 combos, hora indexada exacta 0-23), y en
decisión solo hacer `bisect` + un acceso a arreglo. Velocidad real:
**1.15 µs/decisión — más rápido que el Ejecutor de Dominio 1** (26.5x más
rápido que el código generado). **Pero destruye la detección**: al mismo
umbral, precisión 91.4%→18.8%, recall 80.0%→27.5%. Causa encontrada: el
binning por cuantiles colapsó `huella_categoria_cuenta` (la feature más
importante, 0.945) a **1 solo bin real** de los 16 pedidos, y
`conteo_ventana_global` a 2 de 8 — ambas features tienen distribuciones muy
concentradas/discretas donde los cortes por cuantiles caen todos en el mismo
valor. El techo de velocidad (1.15 µs) es real y vale la pena perseguir, pero
el binning ingenuo por cuantiles no sirve para estas features específicas —
pendiente rediseñar el binning (tratar `huella_categoria_cuenta` por sus
valores discretos reales, no por cuantiles continuos) antes de que esta vía
sea usable. Reporte en `data/reporte_tabla_lookup_dominio2.json`.

**Estado de la exploración de latencia (actualizado, ver tercera ronda abajo):**
30.5 µs (código generado sin NaN) fue el mejor resultado sin tabla. La tabla
de búsqueda se resolvió del todo en la tercera ronda — ver más abajo.

### Tercera ronda (2026-09-13) — tabla de búsqueda con umbrales reales del modelo: exacta y más rápida que Dominio 1

La v1 de la tabla (cuantiles ingenuos, 16 bins por feature) **destruía la
detección** (precisión 91.4%→18.8% al mismo umbral) porque el binning por
cuantiles colapsaba `huella_categoria_cuenta` (la feature más importante) a
1 solo bin real — esa feature tiene solo 40 valores discretos y el 97.7% de
las filas caen exactamente en 1.0, así que los cortes por cuantiles caían
todos en el mismo punto. La v2 (índices exactos para `hora`/`conteo_ventana_global`/
`huella_categoria_cuenta`, cuantiles solo para `amt`/`monto_ewma_cuenta`)
mejoró mucho (precisión 85.2% vs. 91.4%, 1.90 µs) pero seguía siendo una
aproximación.

**Solución definitiva (v3):** en vez de cuantiles o índices ad-hoc, los bins
de la tabla se definen con **los umbrales reales que el propio modelo
aprendió** — extraídos de los nodos de los 175 árboles (`amt`: 44 umbrales
únicos, `hora`: 14, `conteo_ventana_global`: **solo 1** — explica por qué su
importancia individual era tan baja, `monto_ewma_cuenta`: 26,
`huella_categoria_cuenta`: 18). Como el árbol nunca distingue dos valores
que caen en el mismo intervalo entre dos umbrales consecutivos, la tabla
queda **exacta por construcción**, no aproximada — y del tamaño mínimo
posible: 692,550 combinaciones (vs. millones si se usara resolución
arbitraria).

**Bug real encontrado y corregido en el camino:** la primera versión de v3
dio `diff_max=0.43` en 33 de 234,189 filas — causa: `bisect.bisect_right`
enruta un valor *exactamente igual* a un umbral al bin de la derecha, pero
el árbol compara con `<=` (va a la izquierda). Corregido a
`bisect.bisect_left`, reverificado sobre las 234,189 filas completas (no una
muestra): **MAE=0, diferencia máxima=0, correlación=1.000000** — reproducción
bit-perfecta.

| Evaluador | Latencia/decisión | Exactitud |
|---|---|---|
| Diccionarios | 157.3 µs | Exacta |
| Código generado, sin NaN | 30.5 µs | Exacta |
| Tabla v1 (cuantiles ingenuos) | 1.15 µs | Rota (91.4%→18.8% precisión) |
| Tabla v2 (índices parciales) | 1.90 µs | Aproximada (~6 pts de precisión) |
| **Tabla v3 (umbrales reales del modelo)** | **1.53 µs** | **Exacta (bit-perfecta)** |
| Ejecutor Dominio 1 (referencia) | 8.33 µs | — |

**Resultado final: 5.43x más rápido que el Ejecutor de Dominio 1, sin perder
ni un bit de precisión.** Reportes en `data/reporte_tabla_lookup_dominio2.json`
(v1), `data/reporte_tabla_lookup_v2.json` (v2), `data/reporte_tabla_lookup_v3_exacta.json`
(v3, definitivo). Script de construcción de la tabla sigue en el scratchpad
de la sesión — pendiente pasar a `src/` si se retoma la integración de
producción (sección 6, punto 2): la tabla v3 debería ser el mecanismo real
del nuevo Ejecutor de árboles, no el código generado ni los diccionarios.

**Nota para producción, si se retoma:** el tamaño de la tabla (692,550
combinaciones, ~5.5MB en float64) depende de cuántos umbrales aprenda el
modelo — si se reentrena con más árboles o distinto `max_depth`, hay que
reconstruir la tabla (no es estática); el mismo generador de umbrales +
tabla debe ser parte del pipeline de publicación del artefacto, no un paso
manual.

## 7. Afinamiento en curso — importancia de features (hallazgo, no concluido)

A pedido explícito del usuario, la integración de producción (sección 6)
queda **pausada hasta terminar de afinar el modelo** — "antes de empezar
cualquier dominio necesitamos afinar el modelo completamente".

`permutation_importance` (scoring=`average_precision`, sobre val) del modelo
Gradient Boosting de la sección 5:

| Feature | Importancia |
|---|---|
| **huella_categoria_cuenta** | **0.945** |
| **amt** | **0.552** |
| hora | 0.089 |
| monto_ewma_cuenta | 0.076 |
| conteo_ventana_global | 0.028 |
| monto_ewma_global | 0.005 |
| conteo_ventana_cuenta | 0.002 |

**Hallazgo honesto, sin resolver todavía:** dos features (huella + monto)
concentran casi toda la importancia. Las 4 features recursivas heredadas del
enfoque del Dominio 1 (EWMA/conteo, global y por cuenta) aportan muy poco una
vez que el modelo no lineal ya tiene huella+monto — puede ser redundancia
real (Gradient Boosting infiere de huella+monto lo mismo que esas features
aportarían) o una señal de que esas features necesitan mejor diseño para este
dominio específico. Sin conclusión aún.

## Plan de afinamiento — completo, los 5 pasos cerrados

### Paso 1 — Validación cruzada multi-fold

5 folds walk-forward (ventana expansiva, mismo criterio que
`validacion_cruzada.py` del Dominio 1), set de 7 features, `max_depth=6`:

AUC-PR por fold: 0.913, 0.948, 0.932, 0.962, 0.966 → **media 0.944 ± 0.020**.
Estable entre los 5 cortes temporales — no es un accidente de un solo split.

### Paso 2 — Chequeo de sobreajuste

Gap train-val por fold: +0.008, -0.002, -0.018, +0.017, +0.002 — insignificante,
y en 2 de 5 folds val superó a train. **Sin sobreajuste.**

### Paso 3 — Features de bajo aporte

Ablation multi-fold (3 sets de features):

| Set | AUC-PR media |
|---|---|
| A: completo (7 features) | 0.944 ± 0.020 |
| B: sin `monto_ewma_global` + `conteo_ventana_cuenta` (5 features) | 0.939 ± 0.016 |
| C: solo `amt` + `huella_categoria_cuenta` (2 features) | 0.821 ± 0.019 |

B ≈ A (diferencia dentro del ruido) → esas 2 features se descartan sin
pérdida real. C cae de verdad → `hora`, `monto_ewma_cuenta` y
`conteo_ventana_global` sí aportan valor en conjunto pese a que su
importancia individual (sección 7) se viera modesta.

**Set final de producción: 5 features** — `amt`, `hora`,
`conteo_ventana_global`, `monto_ewma_cuenta`, `huella_categoria_cuenta`.

### Paso 4 — Búsqueda de hiperparámetros

Grid de 12 combinaciones (`max_depth` ∈ {3,5,8}, `learning_rate` ∈
{0.05,0.1}, `max_iter` ∈ {200,300}) × 3 folds, sobre el set de 5 features.

Hallazgo real: **`max_depth=8` empeora y se vuelve inestable** (AUC-PR
0.84-0.90, std hasta 0.084) — sobreajuste claro con árboles profundos dado
que solo hay 99 cuentas. `max_depth=3` es mejor y más estable que el 6 usado
originalmente. `max_iter` no importa (200 = 300, converge antes).

**Configuración ganadora:** `max_depth=3, learning_rate=0.05, max_iter=200`.
Validación final de 5 folds con esta configuración: 0.943, 0.975, 0.975,
0.960, 0.978 → **AUC-PR 0.966 ± 0.013** — mejor y más estable que la
configuración original (0.944 ± 0.020).

### Paso 5 — Calibración de probabilidades

**Hallazgo real:** los scores crudos están severamente mal calibrados en el
rango medio-alto — consecuencia esperada de balancear artificialmente las
clases con `sample_weight` al entrenar (afecta las probabilidades, no el
orden, por eso el AUC-PR no lo delataba).

| Percentil de score (test) | Score crudo promedio | Tasa real de fraude |
|---|---|---|
| 99.5-99.9% | 45.5% | 0.85% |
| 99.9-99.99% | 98.9% | 79.9% |
| 99.99-100% | 99.95% | 100% |

Corregido con `IsotonicRegression` (ajustada en VAL, aplicada a TEST, nunca
al revés): **Brier score 0.00124 → 0.00013 (9.5x mejor)**, costo mínimo en
AUC-PR (0.973 crudo vs. 0.961 calibrado en ese split puntual — la validación
robusta de 5 folds, 0.966±0.013, es la cifra que manda). Percentiles
calibrados quedan cerca de la tasa real (0.59% / 79.2% / 100% vs. reales de
0.85% / 79.9% / 100%).

**Necesario si alguna vez se usa `costo_decision.py`** (Dominio 1) con este
modelo — esa herramienta asume que el score es una probabilidad real, no solo
un buen orden relativo.

## Estado: afinamiento completo, integración de producción no iniciada

Los 5 pasos quedaron cerrados con resultado positivo en cada uno. Modelo
final: Gradient Boosting, 5 features, `max_depth=3, learning_rate=0.05,
max_iter=200`, con calibración isotónica como paso de post-procesamiento.
**Ningún código de producción se ha escrito todavía** — todo lo de este
documento son scripts exploratorios de una sola vez, no módulos del repo.

## 8. Validación contra mercado LatAm real — corrección importante de prevalencia

Se buscó un dataset transaccional de fraude público y descargable a nivel
Latinoamérica: **no existe** (confirmado contra Kaggle, literatura académica y
portales de datos abiertos gubernamentales de Colombia/México/Brasil). Sí se
encontraron dos fuentes reales útiles:

**a) Benchmark académico real** (Rugeles Diaz et al. 2025, *Financial
Innovation*, datos reales de una pasarela de pago de 7 países LatAm,
221,292 transacciones de 2022, fraude real 1%): evaluación honesta sobre
datos desbalanceados (su Tabla 14, no la Tabla 11-13 que usa una muestra
rebalanceada y por eso infla las métricas) da **AUC 0.85-0.90, F1 0.60-0.65**
por segmento de comercio. El dataset crudo es propietario, no descargable.

**b) Dato oficial de tasa de fraude real** (Banco Central de Brasil, API
pública Pix, `EstatisticasFraudesPix`): fraude confirmado ≈ **4.5-5.4 casos
por cada 100,000 transacciones (≈0.005%)**, muy por debajo del 0.077% de
fraude que tiene nuestro dataset sintético Sparkov (907/1,170,945).

**Corrección honesta que esto obliga a hacer:** la precisión reportada en la
sección 7 (83-97% según el recall elegido) asume la prevalencia sintética de
Sparkov. Recalculando con la prevalencia real de mercado (~0.005%, vía
Bayes, manteniendo el mismo TPR/FPR del modelo):

| Recall | Precisión (prevalencia sintética 0.07%) | Precisión (prevalencia real ~0.005%) |
|---|---|---|
| 90% | 83% | 26% |
| 80% | 91% | 41% |
| 76% | 97% | 69% |

**Conclusión:** el modelo sigue siendo útil y comparable o mejor que el
benchmark académico real de LatAm (F1 0.60-0.65), pero la afirmación previa
de "muy por encima del mercado" estaba inflada por el desbalance artificial
del dataset sintético, no por una capacidad real superior del modelo. Antes
de cualquier despliegue real, hay que recalibrar el umbral de decisión (y
posiblemente `costo_decision.py`) contra la prevalencia real esperada en el
mercado objetivo, no contra la del dataset de entrenamiento.

## 9. Curva completa recalculada contra prevalencia real (2026-09-13)

Se rehizo el entrenamiento (mismo modelo final: Gradient Boosting,
`max_depth=3, learning_rate=0.05, max_iter=200`, 5 features, split temporal
60/20/20) para obtener TPR/FPR en toda la curva de umbrales del set de TEST
(234,189 filas, 200 fraudes reales), no solo los 3 puntos de la sección 8.
Precisión recalculada vía Bayes con la misma fórmula
(`TPR·π' / (TPR·π' + FPR·(1-π'))`), `π'` = 4.95×10⁻⁵ (punto medio del rango
BCB Pix 4.5-5.4/100k):

| Recall | Precisión (sintética) | Precisión (mercado real ~0.005%) |
|---|---|---|
| 5-40% | 100% | 100% |
| 45% | 97.8% | 72.3% |
| 50% | 98.0% | 74.3% |
| 60% | 96.0% | 58.4% |
| 70% | 94.6% | 50.5% |
| 80% | 91.4% | 38.2% |
| 90% | 83.0% | 22.1% |
| 95% | 67.0% | 10.5% |

Curva completa (19 puntos, cada 5% de recall) en
`data/reporte_precision_recall_prevalencia_real.json`.

**Hallazgo nuevo, no reportado antes:** entre 5% y 40% de recall el modelo
tiene **cero falsos positivos** (precisión 100% en ambas prevalencias) — casi
seguro es la detección trivial de las 12 cuentas "perfil de fraude puro"
(sección 1), no señal generalizable a fraude en cuentas normales. La caída
fuerte de precisión empieza justo después de 40-45% de recall, que es donde
se agotan esas cuentas triviales y el modelo pasa a detectar fraude real en
cuentas normales — ese es el tramo que importa para juzgar el modelo, no el
90% de precisión "aparente" en la parte baja de la curva.

Consistente con los 3 puntos previos de la sección 8 (recall 90/80% → 83%/91%
sintética, 22%/38% real — la sección 8 daba 26%/41%, diferencia esperable por
variación de split/semilla, no una contradicción).

## 10. Señales de red entre cuentas (colusión) — intento negativo (2026-09-13)

Pendiente antiguo de la sección "Pendiente explícito" de más abajo: probar si
existe estructura de colusión (cuentas conectadas por comercio/dispositivo
compartido) usando dimensión fractal de redes (box-covering, Song/Havlin/Makse
2005, *Nature* — método real y establecido para medir auto-similitud en
grafos, no una metáfora).

**Intento 1, descartado sin correr:** usar `merch_lat`/`merch_long` de Sparkov
como proxy de identidad de comercio. Verificado que **cada transacción tiene
una combinación única** de lat/long (1,170,945 de 1,170,945) — Sparkov le
pone ruido GPS a cada transacción, la ubicación no identifica comercio.
Recuperar el campo `merchant` real requeriría regenerar el dataset completo
con Sparkov_Data_Generation — **probado y descartado por lentitud real**: un
test de solo 2 clientes × 10 días tardó >13 minutos sin terminar en esta
máquina (el benchmark del README asume 64 núcleos/128 hilos). Además,
inspeccionando `datagen_transaction.py`, el comercio se asigna por
`random.sample()` independiente por transacción — no es reconstruible desde
los datos ya generados sin re-simular la secuencia aleatoria completa.

**Intento 2, ejecutado con IEEE-CIS (ya descargado, sin esperar nada):**
grafo cuenta(`card1`)–cuenta vía `DeviceInfo` compartido (identity table).
Filtrados nombres genéricos de SO/navegador (`Windows`, `iOS Device`,
`MacOS`, versiones de Firefox) que no son dispositivos reales — quedaron
26,529 filas con modelo de dispositivo específico, 1,203 dispositivos
compartidos por >1 cuenta. Grafo resultante: 2,254 nodos en la componente
conexa mayor, 96,773 aristas.

**Resultado: negativo.** Dimensión fractal del grafo real (3.91, R²=0.93)
**prácticamente igual** a la de un grafo de control aleatorio con la misma
secuencia de grados (4.05, R²=0.89) — la prueba diagnóstica central del
método (real vs. modelo nulo con mismos grados) no encontró diferencia. No
hay evidencia de estructura de colusión con este proxy. Causa probable:
`DeviceInfo` es un **modelo de teléfono** (ej. "Samsung Galaxy S7"), no un
identificador único de aparato físico — cientos de cuentas no relacionadas
"comparten" el mismo modelo simplemente por ser un teléfono popular, mismo
problema de fondo que descartó la vía de lat/long (proxy de popularidad, no
de identidad real).

**Para hacerlo bien de verdad** (no intentado, es ingeniería real no
trivial): combinar `card1+card2+card5+addr1+D1` como pseudo-identidad de
cuenta (técnica conocida de las soluciones ganadoras de esa competencia de
Kaggle) y varios campos `id_3x` juntos como huella de dispositivo, en vez de
un solo campo. Reporte completo en `data/reporte_grafo_fractal_ieee.json`.

## Pendiente explícito, sin iniciar

- Probar el modelo/enfoque contra datasets de fraude de Norteamérica,
  Centroamérica y Europa — **CERRADO 2026-09-13**: se probaron IEEE-CIS
  (Norteamérica, prevalencia 3.50%) y el dataset europeo clásico ULB/`creditcard.csv`
  (prevalencia 0.17%). El patrón de prevalencia-inflada-en-sintéticos/curados
  **no es específico de LatAm** — se repite en los tres continentes, con la
  brecha más extrema en IEEE-CIS (700x la tasa real vs. 14x en Sparkov y 34x
  en el europeo). Ver tabla comparativa en la sesión del 2026-09-13.
- Construir la integración de producción de la sección 6 (artefacto de
  árboles + Ejecutor compatible + calibración isotónica como parte del
  artefacto) — recién ahora que el afinamiento está cerrado, queda a
  decisión del usuario cuándo empezar.
- Validación exploratoria barata de señales de red entre cuentas (colusión) —
  requiere recuperar el campo `merchant`, descartado al combinar los archivos.
  **Intentado 2026-09-13, resultado negativo, ver detalle abajo.**
- Point-in-time encoding de `category` (sección 2), si se retoma esa feature.
- Huella de comportamiento con decaimiento tipo EWMA en vez de ventana de
  conteo fijo (refinamiento menor, no bloqueante).

## 11. Saerens-Latinne-Decaestecker (SLD) — corrección de prevalencia más rigurosa (2026-09-13)

Método estándar de la industria (EM iterativo) para estimar/corregir la
prevalencia real cuando difiere de la de entrenamiento — más riguroso que el
Bayes de un solo punto de la sección 8/9.

**Primer intento, falló (100% de error):** alimentar SLD con los scores
crudos del modelo (sin calibrar) hizo que el EM divergiera a un prior
estimado de ~0 en vez de la prevalencia real del test (0.0854%). Causa: SLD
asume probabilidades de entrada bien calibradas, y ya sabíamos (Paso 5 del
afinamiento) que los scores crudos NO lo están.

**Corregido:** calibrar con `IsotonicRegression` (ajustada en VAL, igual que
el Paso 5) antes de correr SLD, usando como prior de entrenamiento la
prevalencia real de VAL (no un 50/50 asumido). Con eso, SLD estimó la
prevalencia del test **sin ver las etiquetas**: 0.0896% vs. la real
0.0854% — **error relativo 4.87%**. Validación real y positiva: el método
funciona una vez que se respeta su supuesto de calibración.

**Error de interpretación propio, encontrado y corregido en el momento:**
intenté recalcular "precisión por nivel de recall" usando las probabilidades
reajustadas por SLD contra la prevalencia real asumida — el resultado
coincidía con la columna *sintética* de la sección 9 (91.4% a 80% recall),
no con la columna de mercado real (38.2%). Razón: la precisión TP/(TP+FP)
calculada sobre las etiquetas reales del test siempre refleja la
composición real de ESE test (prevalencia sintética) sin importar qué
probabilidad recalibrada se le asigne a cada fila — recalibrar el score no
cambia las etiquetas ni los conteos. **La curva de precisión/recall contra
prevalencia real de la sección 9 (Bayes sobre TPR/FPR) sigue siendo la
correcta y no cambia con esto.**

**Lo que SLD sí aporta, validado y nuevo:**
1. Estimación de la prevalencia real sin necesitar un dato externo (como el
   de BCB Pix) — útil si en el futuro no hay una tasa de mercado conocida a
   priori, solo datos de producción sin etiquetar.
2. Probabilidad recalibrada por transacción individual (no solo puntos
   agregados de una curva) — útil para un umbral absoluto de negocio ("marcar
   para revisión manual si probabilidad ajustada > X%"), complementario a la
   curva de la sección 9, no un reemplazo.

Reporte completo (con ambos intentos) en `data/reporte_sld_prior_shift.json`.
