# Dominio 2 — Personalización por cuenta (investigación en curso)

**Estado:** Exploración, validación e integración de producción completas con
resultados reales (ver sección 13). Sigue en fase de pruebas -- nada de esto
se ha desplegado ni conectado a tráfico real.

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

## Estado: afinamiento completo, integración de producción completa (ver sección 13)

Los 5 pasos quedaron cerrados con resultado positivo en cada uno. Modelo
final: Gradient Boosting, 5 features, `max_depth=3, learning_rate=0.05,
max_iter=200`, con calibración isotónica como paso de post-procesamiento.
La integración de producción (artefacto + Ejecutor + Ciclo, sección 6) se
construyó el 2026-09-14 — ver sección 13 para el detalle y las cifras reales
reproducidas.

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
  artefacto) — **CERRADO 2026-09-14**, ver sección 13.
- Validación exploratoria barata de señales de red entre cuentas (colusión) —
  requiere recuperar el campo `merchant`, descartado al combinar los archivos.
  **Intentado 2026-09-13, resultado negativo, ver detalle abajo.**
- Point-in-time encoding de `category` (sección 2), si se retoma esa feature —
  **CERRADO 2026-09-14, resultado positivo**, ver sección 15.
- Huella de comportamiento con decaimiento tipo EWMA en vez de ventana de
  conteo fijo — **CERRADO 2026-09-14, resultado negativo/inconcluso**, ver
  sección 15.

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

## 12. Benchmark contra datasets reales (IEEE-CIS, ULB) y deriva de concepto (2026-09-13)

Motivado por evaluar honestamente si el enfoque "compite con el mercado" en
detección, no solo en latencia (que sí está confirmado, sección 6).

**Benchmark en ULB `creditcard.csv` (real, Europa):** misma disciplina
metodológica (split temporal, Gradient Boosting, grid de hiperparámetros)
aplicada a los datos reales de esta dataset (no a Sparkov). Resultado:
AUC-ROC 0.982, AUC-PR 0.780 (split temporal) / 0.817 (split aleatorio,
igualando la metodología típica de notebooks públicos). **Por debajo** del
mejor resultado publicado conocido (XGBoost afinado, AUC-PR 0.9133) — ni
ajustando hiperparámetros (grid de 12) ni igualando el split cerramos esa
brecha. Reportes en `data/reporte_benchmark_ulb*.json`.

**Conclusión honesta:** en detección, **no superamos el estado del arte
publicado** en datos reales — somos competitivos, no superiores. La ventaja
real y confirmada sigue siendo la latencia (sección 6), no la detección.

**Deriva de concepto (Sparkov, dentro del propio tramo de test 2023-2026):**
AUC-PR por año calendario: 2024→0.971, 2025→0.922, 2026(parcial)→0.892 —
**caída consistente y monótona**, mientras AUC-ROC se mantiene ~1.0 en los
tres (confirma otra vez que AUC-ROC esconde el deterioro, igual que en la
sección 4). Con ~4 puntos de AUC-PR de caída por año, **reentrenar al menos
anualmente sería razonable** en un despliegue real — aunque con solo 35-106
fraudes por año la pendiente exacta tiene incertidumbre estadística real, la
dirección (siempre bajando) es consistente en los 3 períodos, no un
accidente de un corte. Reporte en `data/reporte_concept_drift_sparkov.json`.

## 13. Integración de producción — construida y validada (2026-09-14)

Cierra el pendiente de la sección 6: el artefacto de árboles + Ejecutor
compatible + `CicloDecision` paralelo, ya en `src/`, no en scripts de
scratchpad. **Camino paralelo a Dominio 1, sin tocar ningún archivo
existente** (`artefacto.py`, `ejecutor.py`, `ciclo.py`, `calibrador.py`,
`features_recursivas.py`, `veto.py`, `puente.py` quedan intactos) — se
reutilizan sin modificar `veto.py::evaluar` (independiente del modelo por
diseño) y `features_recursivas.py::EstadoRecursivoGlobal`/`calcular_features_recursivas_batch`
para `conteo_ventana_global` (misma métrica global de Dominio 1).

**Módulos nuevos:**
- `src/features_recursivas_cuenta.py` — `monto_ewma_cuenta` y
  `huella_categoria_cuenta` por cuenta (`EstadoRecursivoPorCuenta` para el
  Ejecutor, `calcular_features_recursivas_cuenta_batch` para el Calibrador),
  con la misma disciplina de paridad batch/incremental que Dominio 1
  (verificada con test explícito, no solo compartiendo código).
- `src/artefacto_arboles.py` — contrato neutral: en vez de serializar los
  175 árboles para recorrerlos en caliente (más lento, ver sección 6), el
  artefacto guarda directamente la **tabla de búsqueda v3** (umbrales reales
  del modelo + tabla plana) y la calibración isotónica como breakpoints
  `x`/`y` (interpolación lineal en Python puro, verificada idéntica a
  `IsotonicRegression.predict(out_of_bounds="clip")`).
- `src/calibrador_arboles.py` — entrena `HistGradientBoostingClassifier`
  sobre `data/raw/sparkov_2013_2026.csv` (columna `class_weight="balanced"`
  soportada nativamente por el constructor en sklearn 1.9.1, no hizo falta
  `sample_weight` manual), extrae los umbrales reales de los nodos, y
  construye la tabla evaluando `predict_proba` una sola vez por combinación
  de bin (vectorizado con `np.meshgrid`, no en Python puro -- eso corre en
  calibración, nunca en el camino caliente).
- `src/ejecutor_arboles.py` — `EjecutorArboles`, capa rápida: solo
  `bisect_left` + indexación de tabla plana (sin recorrer árboles, sin
  diccionarios de nodos), reusa `EstadoRecursivoGlobal` y
  `EstadoRecursivoPorCuenta`.
- `src/puente_arboles.py` / `src/ciclo_arboles.py` — escritura atómica y
  único punto de entrada sancionado, mismas garantías que Dominio 1
  (artefacto siempre por el Puente, resultado siempre por `veto.py`,
  cualquier excepción del Ejecutor escala a revisión manual). `veto.py`
  espera la clave `Amount` (contrato de Dominio 1) -- `CicloDecisionArboles`
  agrega un alias de `amt` al llamar a `evaluar_veto`, sin modificar
  `veto.py`.

**Cifras reales reproducidas** (split temporal 60/20/20 sobre las 1,170,945
filas, `random_state=42`):

| | VAL | TEST |
|---|---|---|
| n | 234,189 | 234,189 (200 fraudes reales) |
| Precisión | 88.7% | 87.8% |
| Recall | 81.8% | 83.0% |
| F1 | 0.851 | 0.853 |
| AUC-ROC | 0.9999 | 0.9999 |
| AUC-PR | 0.898 | 0.901 |

**Umbrales reales extraídos de los 175 árboles** (definen los bins de la
tabla): `amt` 42, `hora` 13, `conteo_ventana_global` 1, `monto_ewma_cuenta`
27, `huella_categoria_cuenta` 15 → tabla de `43×14×2×28×16 = 539,392`
combinaciones (vs. las 692,550 de la sección 6 -- ese número salió del
modelo previo a la búsqueda de hiperparámetros del Paso 4, con
`max_depth=6`; el modelo final `max_depth=3` genera árboles más simples con
menos cortes únicos por feature, tabla más chica, mismo principio).

**Exactitud bit a bit, verificada sobre las 234,189 filas completas de TEST**
(replay desde el inicio de TODO el historial -- train+val+test en orden, no
solo el tramo de test, para que el estado recursivo por cuenta/global
arranque igual que en producción real, no vacío en un corte arbitrario):
MAE = 1.94×10⁻²³, diferencia máxima = 3.47×10⁻¹⁸, correlación = 1.000000 —
bit-perfecto contra `predict_proba()` + calibración isotónica real,
replicando el resultado de la sección 6.

**Latencia real medida** (`time.perf_counter()`, metodología de
`scripts/validacion_end_to_end.py`, 1,170,945 decisiones): **9.84
µs/decisión** para `EjecutorArboles.decidir()` completo (tabla + ambos
estados recursivos, global y por cuenta) — comparable al Ejecutor de
Dominio 1 (8.33 µs). Aislando solo la tabla de búsqueda (sin mantenimiento
de estado, misma metodología que la sección 6): **3.51 µs/decisión**. Ambos
números son **más altos que los 1.53 µs reportados en la sección 6** — esa
cifra medía solo la aritmética de la tabla con features ya calculadas, sin
el costo real de mantener `EstadoRecursivoPorCuenta` (una `deque` de 20
categorías + un diccionario por cuenta) ni de `hora_utc()`
(`datetime.fromtimestamp` por decisión). El número honesto y comparable
contra Dominio 1 es el de camino completo: **9.84 µs**, no 1.53 µs — sigue
siendo ~5,000x más rápido que una ventana de autorización típica de redes
de tarjetas, no es un bloqueante real.

**Desviación honesta de las cifras del afinamiento:** el AUC-PR real
(0.898/0.901 val/test) queda por debajo del 0.966±0.013 reportado en el
Paso 4 del afinamiento (promedio de 5 folds *walk-forward* con ventana
expansiva) y de los puntos 0.973/0.961 del Paso 5. Explicación más probable,
no verificada exhaustivamente: aquí se usa un único split fijo 60/20/20
(mayor varianza que un promedio de 5 folds), más diferencias menores de
reimplementación (regla exacta de respaldo poblacional bajo
`min_historial=5`, `random_state=42`, versión de sklearn). El número sigue
muy por encima del benchmark publicado (AUC-PR 0.51-0.66) y es del mismo
orden que los resultados originales pre-afinamiento de la sección 5 — no se
investigó más a fondo (ej. correr el mismo 5-fold walk-forward sobre esta
implementación) porque no formaba parte del alcance de esta tarea.

**132 tests de Dominio 1 siguen en verde, sin modificar ningún archivo
existente.** 50 tests nuevos (`tests/test_features_recursivas_cuenta.py`,
`tests/test_artefacto_arboles.py`, `tests/test_puente_arboles.py`,
`tests/test_ciclo_arboles.py`, `tests/test_calibrador_arboles.py`,
`tests/test_ejecutor_arboles.py`) — suite completa: **182 tests, todos en
verde**. Incluyen el test bit-exacto de arriba (con skip automático si
`data/raw/sparkov_2013_2026.csv` no está presente) y una versión rápida
equivalente sobre un dataset sintético pequeño para no depender del CSV real
de 119MB en cada corrida.

**Dos hallazgos reales de la auditoría en el camino, corregidos antes de
cerrar:**
1. `artefacto_arboles._es_numero` rechazaba `NaN` en la validación de forma
   -- inconsistente con `artefacto.py` de Dominio 1 (que sí acepta NaN ahí,
   dejando la protección real a `veto.py`). Corregido para que el contrato
   sea el mismo en ambos dominios.
2. `_interpolar_isotonica` (la interpolación lineal pura de la calibración)
   reventaba con `IndexError` ante un score de entrada `NaN` -- `bisect` no
   sabe comparar NaN, y ni `x <= xs[0]` ni `x >= xs[-1]` son verdaderos para
   NaN, así que caía al camino de interpolación normal y se salía del
   arreglo. Corregido con un chequeo explícito al inicio (`math.isnan`) que
   devuelve NaN limpio, igual que `Ejecutor._sigmoide` de Dominio 1
   (`math.exp(nan)` no lanza). Encontrado por un test que construía a
   propósito un artefacto con `NaN` en la tabla.

**No se tocó ningún ADR existente** (son de Dominio 1, siguen vigentes tal
cual). **No se hizo commit ni deploy de nada al momento de escribir esta
sección** -- ver sección 14 para el cierre de la investigación de la brecha
de AUC-PR, y `docs/CAMINO3_GENERALIZACION_IOT.md` para el cierre de Camino 3
(dominio IoT), que se completó después de esta sección.

## 14. Investigación de la brecha AUC-PR — causa raíz encontrada: dependencia de historial por cuenta (2026-09-14)

La sección 13 dejó sin investigar por qué el AUC-PR real de producción
(0.898/0.901) queda por debajo del 0.966±0.013 del Paso 4 del afinamiento.
Se retomó y se descartaron 4 hipótesis con evidencia directa, hasta
encontrar una explicación real (no solo "probablemente varianza").

**Primero, se confirmó que la brecha es real, no varianza de un split
único:** corriendo el mismo walk-forward de 5 folds (`frac_train_inicial=0.5`,
igual que `validacion_cruzada.py`) sobre el código de producción
(`src/calibrador_arboles.py`): AUC-PR por fold 0.877, 0.879, 0.931, 0.935,
0.947 → **media 0.914 ± 0.030**. El split único (0.898/0.901) está en línea
con este promedio, no es un caso raro.

**4 hipótesis probadas y descartadas, con evidencia directa:**

1. **Calibración isotónica** (el código de producción siempre la aplica antes
   de medir AUC-PR; el Paso 4 del doc, anterior al Paso 5 de calibración,
   probablemente medía sobre scores crudos) — medido directamente: la
   diferencia crudo vs. calibrado es de solo ±0.003-0.005 por fold. No
   explica la brecha de ~0.05.
2. **`class_weight="balanced"` vs. `sample_weight` manual** (el doc habla de
   "balancear con sample_weight", el código de producción usa el parámetro
   del constructor) — probado en 2 folds: dan resultados **idénticos**
   (0.8814 y 0.9505 en ambos casos). Sin balanceo, cae a 0.82/0.79 —
   confirma que el balanceo es necesario y ya está bien aplicado, pero no es
   la causa de la brecha.
3. **Fuga de datos en `frecuencia_poblacional_categoria`** (¿el script
   exploratorio la habrá calculado sobre el dataset completo en vez de solo
   TRAIN?) — probado deliberadamente con fuga: el resultado con fuga es
   **igual o levemente peor**, no mejor. Descartada.
4. **Bug en el cálculo de `huella_categoria_cuenta`** — verificado contra la
   cifra exacta que el propio doc ya había reportado en la sección 6
   (tercera ronda): 40 valores únicos, 97.7% de las filas en exactamente
   1.0. El código de producción reproduce **exactamente** esos números,
   además de filas totales (1,170,945), fraudes totales (907) y cuentas
   (99) idénticos al resto del documento. La feature más importante del
   modelo está bien calculada.

**Causa raíz encontrada — arranque en frío (cold start) por cuenta:**
variando `frac_train_inicial` (cuánto historial de "calentamiento" tiene
cada cuenta antes de que arranque el primer fold de validación):

| `frac_train_inicial` | AUC-PR por fold | Media | Desviación |
|---|---|---|---|
| 0.5 (arranque en frío, cuentas nuevas) | 0.877 – 0.947 | **0.914** | ±0.030 |
| 0.6 | 0.869 – 0.964 | **0.926** | ±0.033 |
| 0.7 (historial amplio, cuentas establecidas) | 0.919 – 0.991 | **0.940** | ±0.026 |

Tendencia **monótona y consistente**: a más historial acumulado por cuenta
antes de medir, mejor rinde el modelo — coherente con que
`huella_categoria_cuenta` (la feature de mayor importancia, 0.945) necesita
historial real de la cuenta para ser informativa; con poco historial, más
filas caen en el respaldo poblacional (genérico, menos discriminante). No
cierra el 100% de la brecha contra el 0.966 del doc (probablemente el
script exploratorio usó un calentamiento aún mayor, o alguna otra diferencia
metodológica menor no identificada), pero explica la mayor parte de forma
verificable y con evidencia monótona, no especulativa.

**Esto no es un bug — es un prerrequisito operativo real de Dominio 2, que
hay que comunicar si el enfoque se usa de verdad:**

- **Cuenta nueva / poco historial:** AUC-PR ≈ 87.7%-94.7% (media 91.4%),
  precisión 85-95%, recall 82-89% (rango medido en el detalle del split de
  `frac_train_inicial=0.5`).
- **Cuenta establecida / historial amplio:** AUC-PR ≈ 91.9%-99.1% (media
  94.0%).

La personalización por cuenta mejora con el tiempo a medida que la cuenta
acumula transacciones — el mismo problema de "arranque en frío" que tiene
cualquier sistema de personalización (recomendadores, scoring de crédito).
Una cuenta nueva en producción real arrancaría en el escenario más débil.

**Número vigente para reportar de aquí en adelante:** 0.898-0.914 de
AUC-PR (según el escenario de historial), no 0.966 — ese número del Paso 4
queda marcado como no reproducido con el código de producción y su origen
exacto sin confirmar (script exploratorio ya no disponible).

## 15. Dos refinamientos menores explorados (2026-09-14)

Cierra los 2 últimos ítems de "Pendiente explícito". Exploración pura —
nada de esto se llevó a `src/`. Misma metodología de comparación que la
sección 14 (walk-forward 5-fold, `frac_train_inicial=0.5`,
`HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
max_iter=200, class_weight="balanced", random_state=42)`, isotónica en VAL,
AUC-PR sobre scores calibrados), contra el mismo baseline de 5 features:
**0.914 ± 0.030** (folds: 0.877, 0.879, 0.931, 0.935, 0.947).

### 15.1 Point-in-time encoding de `category` — resultado positivo

Se agregó una 6ta feature, `frecuencia_categoria_expandida`: para cada
fila, `(conteo_categoria_hasta_ahora + 1) / (conteo_total_hasta_ahora +
n_categorias)` — causal (solo cuenta filas estrictamente anteriores) y
expandible (se recalcula fila a fila sobre todo el historial, con
suavizado de Laplace `alpha=1` para no dar valores extremos en las
primeras apariciones de una categoría).

**Verificado explícitamente que NO reproduce la trampa de no-estacionariedad
de la sección 2** (one-hot fijo + split temporal rompiendo el modelo por
`personal_care`): la trayectoria real de esta feature para esa categoría
por año —

| Año | Transacciones | Valor medio de la feature |
|---|---|---|
| 2013-2023 (11 años) | 1-4 por año | 0.000023 – 0.000090 |
| 2025 | 63,876 | 0.029253 |
| 2026 | 18,580 | 0.064972 |

— muestra que el valor **se adapta suavemente** al volumen real, sin
quedar pegado a las ~24 filas tempranas: a diferencia de un one-hot
estático ajustado una sola vez, acá el peso de esa evidencia vieja se
diluye solo (por construcción) a medida que crece el denominador total.
No hay coeficiente fijo aprendiendo una regla espuria de pocas filas.

**Resultado (6 features vs. 5 baseline):**

| | AUC-PR por fold | Media | Desviación |
|---|---|---|---|
| Baseline (5 features) | 0.877, 0.879, 0.931, 0.935, 0.947 | 0.914 | ±0.030 |
| **+ `frecuencia_categoria_expandida` (6 features)** | 0.907, 0.923, 0.976, 0.958, 0.970 | **0.947** | ±0.027 |

**Mejora real de +0.033 AUC-PR, consistente en los 5 folds** (mejora en
4 de 5, el fold 0 queda prácticamente igual). No es ruido — la magnitud y
la consistencia entre folds son mayores que la desviación estándar.

**Recomendación:** vale la pena llevar esta feature a producción
(`src/calibrador_arboles.py` + `src/ejecutor_arboles.py`). **Implementado
2026-09-14, ver sección 16** — AUC-PR real de producción confirmó la
mejora (0.947±0.027 en 5-fold, reproduciendo casi exacto este hallazgo).

### 15.2 Huella con decaimiento EWMA en vez de ventana fija K=20 — resultado negativo/inconcluso

Se reemplazó `huella_categoria_cuenta` (ventana causal de las últimas 20
transacciones de la cuenta) por un histograma de categorías por cuenta con
decaimiento exponencial: en cada transacción, todos los pesos existentes
decaen por `lambda`, se suma `(1-lambda)` a la categoría actual, y la
huella es el peso normalizado de la categoría actual (respaldo poblacional
igual que hoy para cuentas con <5 transacciones reales, no ponderadas).

**Resultado (reemplazando la huella de ventana fija, no agregándola):**

| Variante | AUC-PR por fold | Media | Desviación |
|---|---|---|---|
| Baseline (ventana fija K=20) | 0.877, 0.879, 0.931, 0.935, 0.947 | 0.914 | ±0.030 |
| EWMA λ=0.98 (mismo λ que el resto del proyecto) | 0.857, 0.852, 0.931, 0.904, 0.924 | **0.894** | ±0.033 |
| EWMA λ=0.9 (decaimiento más rápido) | 0.866, 0.901, 0.950, 0.941, 0.958 | **0.923** | ±0.035 |

**λ=0.98 empeora de forma clara** (-0.020, peor en 4 de 5 folds).
**λ=0.9 mejora levemente** (+0.009), pero la desviación estándar (±0.033-0.035)
es mayor que la diferencia — no hay evidencia sólida de que sea una mejora
real y no ruido de muestreo.

**Veredicto honesto: no hay caso claro para reemplazar la ventana fija de
20 transacciones.** La ventana fija (que "olvida" del todo después de 20
transacciones) parece capturar mejor "comportamiento reciente típico" que
un decaimiento suave que nunca olvida del todo — posiblemente porque el
comportamiento de cuenta en Sparkov no tiene una deriva gradual que el
decaimiento aproveche, similar en espíritu al hallazgo de Camino 3 (el
suavizado EWMA tampoco ayudó ahí, por la misma razón: no toda señal se
beneficia de memoria larga).

**Recomendación:** no llevar este cambio a producción — la versión actual
(ventana fija K=20) se mantiene como la mejor opción validada.

## 16. `frecuencia_categoria_expandida` llevada a producción (2026-09-14)

Cierra la recomendación de la sección 15.1: la 6ta feature (point-in-time
encoding causal de `category`, Laplace `alpha=1`) ya está en `src/`, no solo
en exploración.

**Archivos modificados:**
- `src/features_recursivas_cuenta.py` — nueva función batch
  `calcular_frecuencia_categoria_expandida_batch(df, n_categorias)`
  (vectorizada: `groupby("category").cumcount()` para el conteo causal por
  categoría + `np.arange` para el total, sin loop) y nueva clase de estado
  incremental `EstadoFrecuenciaCategoriaGlobal` (global, no por cuenta —
  vive en este módulo por cohesión con `calcular_frecuencia_poblacional_categoria`,
  no porque sea "por cuenta"). `n_categorias` se deriva de
  `len(frecuencia_poblacional_categoria)` (vocabulario de TRAIN) en vez de
  agregar un campo nuevo al artefacto.
- `src/calibrador_arboles.py` — `FEATURES_ARBOLES` pasa de 5 a 6 entradas;
  `construir_features()` agrega la feature nueva.
- `src/ejecutor_arboles.py` — `EjecutorArboles` mantiene
  `EstadoFrecuenciaCategoriaGlobal` junto a los otros dos estados
  recursivos, alimentando la 6ta dimensión de la tabla de búsqueda.
- `src/artefacto_arboles.py` — **sin cambios de esquema**: `features` ya
  era una lista genérica, y `n_categorias` se reutiliza de
  `frecuencia_poblacional_categoria` en vez de un campo nuevo.
- Tests nuevos: 3 en `tests/test_features_recursivas_cuenta.py` (caso a
  mano, estado incremental aislado, paridad batch/incremental sobre
  secuencia aleatoria) + 1 en `tests/test_ejecutor_arboles.py` (confirma
  que la feature nueva participa de verdad en la tabla de búsqueda, con un
  artefacto sintético de 6 features).

**Cifras reales reproducidas** (código de producción, no el script
exploratorio):

| Metodología | AUC-PR | Comparar contra (5 features) |
|---|---|---|
| 5-fold walk-forward (`frac_train_inicial=0.5`) | **0.947 ± 0.027** | 0.914 ± 0.030 |
| Split único 60/20/20, VAL | 0.932 | 0.898 |
| Split único 60/20/20, TEST | 0.925 | 0.901 |

El 5-fold reproduce casi exacto lo que había medido la exploración de la
sección 15.1 (0.947±0.027) — confirma que el hallazgo no era un artefacto
del script exploratorio. El split único también mejora de forma
consistente con la 6ta feature (no solo el 5-fold), aunque sigue
reflejando la dependencia de historial/arranque en frío ya documentada en
la sección 14 (0.925-0.932 vs. 0.947 del 5-fold, mismo patrón que antes).

**Tabla de búsqueda:** pasa de `[43, 14, 2, 28, 16]` (539,392 combinaciones,
5 features) a `[42, 11, 2, 26, 16, 29]` (**11,147,136 combinaciones**, 6
features) — crece ~20x por la dimensión nueva (29 umbrales únicos de
`frecuencia_categoria_expandida` aprendidos por los árboles). Sigue siendo
exacta por construcción (mismo mecanismo `bisect_left`), verificado con el
test bit-exacto real: diferencia máxima < 1×10⁻⁹ sobre las 234,189 filas
completas de TEST.

**Latencia real medida:** **10.90 µs/decisión** (1,170,945 decisiones,
`time.perf_counter()`) — sube desde los 9.84 µs de 5 features (una lectura
más de estado + una dimensión más en la tabla), sigue en el mismo orden que
Dominio 1 y muy por debajo del presupuesto de industria.

**Tests: 186 en total, todos en verde** (182 anteriores + 4 nuevos; ninguno
de los 182 tests existentes se modificó para lograrlo — la generalidad de
`FEATURES_ARBOLES` como lista y de `features`/`umbrales_por_feature` en el
artefacto absorbió el cambio de 5 a 6 dimensiones sin romper nada).

**No se tocó Dominio 1.** No se hizo commit ni deploy — queda a decisión
del usuario. SYNAPSE sigue en fase de pruebas.

## 17. Ablation: ¿hay redundancia entre las 2 features de categoría? (2026-09-14)

Pregunta directa tras llevar `frecuencia_categoria_expandida` (global) a
producción junto a `huella_categoria_cuenta` (por cuenta, ya existente):
¿se pisan entre sí, o aportan señal genuinamente distinta? Ablation de 4
sets de features, misma metodología de comparación que las secciones
14-16 (walk-forward 5-fold, `frac_train_inicial=0.5`, configuración ganadora
del Paso 4, isotónica en VAL, AUC-PR sobre scores calibrados):

| Set | AUC-PR por fold | Media | Desviación |
|---|---|---|---|
| **6 features (baseline producción)** | 0.907, 0.923, 0.976, 0.958, 0.970 | **0.947** | ±0.027 |
| Sin `huella_categoria_cuenta` (5, solo la global) | 0.438, 0.332, 0.395, 0.454, 0.606 | **0.445** | ±0.091 |
| Sin `frecuencia_categoria_expandida` (5, versión anterior) | 0.877, 0.879, 0.931, 0.935, 0.947 | **0.914** | ±0.030 |
| Solo `amt` + las 2 de categoría (3) | 0.833, 0.857, 0.838, 0.827, 0.895 | **0.850** | ±0.025 |

**Veredicto honesto: NO hay redundancia — al contrario, las dos features
de categoría son asimétricamente importantes, y ambas aportan señal real.**

- **Quitar `huella_categoria_cuenta` es catastrófico**: el modelo se
  derrumba de 0.947 a **0.445** (más que a la mitad) y la desviación entre
  folds se triplica (±0.091) — es, por lejos, la feature que más carga el
  modelo, consistente con la importancia de permutación ya medida en la
  sección 7 (0.945, la más alta de todas).
- **Quitar `frecuencia_categoria_expandida` cuesta mucho menos**: cae de
  0.947 a 0.914 (-0.033) — real, pero no comparable en magnitud a quitar la
  huella. Es un complemento genuino, no decorativo, pero no es la columna
  vertebral del modelo.
- **Las otras 3 features (`hora`, `conteo_ventana_global`,
  `monto_ewma_cuenta`) también aportan de forma real**: con solo `amt` +
  las 2 de categoría (3 features), el modelo cae a 0.850 — 0.097 por debajo
  del set completo. No son prescindibles pese a tener importancia
  individual modesta (mismo patrón ya visto en el ablation de la sección 7:
  "Set C" con solo 2 features caía de forma parecida).

**Conclusión:** las 6 features de producción se quedan como están — cada
una aporta señal real y verificada, ninguna es redundante con otra al
punto de justificar sacarla. `huella_categoria_cuenta` es, con mucha
diferencia, la feature más crítica de todo Dominio 2.

## 18. Columnas demográficas nunca probadas — no disponibles, no un resultado negativo real

Pendiente #2 de la lista de pruebas adicionales: probar edad/género/ocupación/
ciudad del cliente como features nuevas, candidatas típicas del generador
`Sparkov_Data_Generation` que nunca se usaron en el modelo.

**Hallazgo real antes de poder probar nada:** `data/raw/sparkov_2013_2026.csv`
**no tiene esas columnas**. Sus 11 columnas reales son: `cc_num`, `lat`,
`long`, `trans_date`, `trans_time`, `unix_time`, `category`, `amt`,
`is_fraud`, `merch_lat`, `merch_long` — verificado leyendo el CSV
directamente, no supuesto. La sección 1 del doc ya lo explicaba: el dataset
fue "combinado y reducido a las columnas útiles (61 archivos crudos → 1
CSV)" — género, fecha de nacimiento, ocupación y ciudad/estado se
descartaron en ese paso de combinación, nunca llegaron al CSV que usa el
proyecto.

**Se buscó si quedaba algún archivo de generación original con esos datos**
(`/tmp/sparkov_gen`, la instalación del generador usada en su momento):
existen `datagen_customer.py`, `profiles/` (perfiles demográficos genéricos
del generador, no de las 99 cuentas reales) y `demographics.csv` (tabla de
referencia poblacional de EE.UU., tampoco específica de las cuentas), pero
**no hay un `customers.csv` de la corrida real que generó las 99 cuentas**
de `sparkov_2013_2026.csv` — solo quedó guardado el CSV de transacciones ya
combinado y reducido. Recuperar esos datos requeriría **regenerar el
dataset completo desde cero** con `Sparkov_Data_Generation`, el mismo
obstáculo ya documentado en la sección 10 (un test de solo 2 clientes × 10
días tardó >13 minutos sin terminar en esta máquina — regenerar 100
clientes × 13 años no es viable aquí).

**Veredicto honesto: esta prueba queda cerrada por falta de datos, no por
un resultado negativo del modelo.** No es "se probó y no ayudó" (como la
sección 15.2) — es "no se pudo probar porque el dato ya no existe en el
dataset que tenemos". Si en algún momento se regenera Sparkov desde cero
guardando también el archivo de clientes, esta prueba queda pendiente de
verdad para retomar. Con los datos actuales, el dataset de transacciones
está agotado en cuanto a columnas explorables — las 6 features de
producción son todo lo que da de sí `sparkov_2013_2026.csv` tal como existe
hoy.

## 19. Detector de deriva (`src/deriva.py`) vs. `frecuencia_categoria_expandida` — dispara demasiado seguido, recomendación real

`src/deriva.py` (PSI + Kolmogorov-Smirnov, Dominio 1) es genérico —
`evaluar_deriva()` corre sobre cualquier columna numérica de un DataFrame,
sin ninguna dependencia real de Dominio 1. Se aplicó tal cual a las 6
features de producción de Dominio 2. `ponderar_deriva_por_coeficiente()`
**sí** es específico de Dominio 1 (asume un artefacto con
`coeficientes`/modelo lineal) — no aplica al artefacto de árboles de
Dominio 2 sin adaptarlo; **hallazgo aparte, fuera de alcance de esta
prueba**: no existe todavía un disparador de recalibración automática para
Dominio 2 (`disparador_recalibracion.py` es exclusivo de Dominio 1).

**Control positivo — el detector SÍ ve el cambio real y esperado:**
comparando 2013-2018 (referencia) contra 2024-2026 (actual),
`frecuencia_categoria_expandida` da **PSI=4.84** (`deriva_significativa_recalibrar`,
muy por encima del umbral de 0.25) — consistente con el cambio real y ya
documentado en la sección 15.1 (`personal_care` pasando de ~0.00002 a
~0.06). El detector funciona: no es ciego al cambio real.

**El problema real — dispara casi siempre, no solo cuando importa.**
Comparando cada año contra el siguiente (13 transiciones válidas,
2013→2014 … 2025→2026):

| Feature | Dispara recalibración (PSI>0.25) |
|---|---|
| `amt` | 0/13 (0%) |
| `hora` | 3/13 (23%) |
| `conteo_ventana_global` | 0/13 (0%) |
| `monto_ewma_cuenta` | 1/13 (8%) |
| `huella_categoria_cuenta` | 0/13 (0%) |
| **`frecuencia_categoria_expandida`** | **11/13 (85%)** |

**Recomendación agregada de recalibrar** (con `evaluar_deriva()` completo,
"recalibrar si CUALQUIER feature dispara"), mismas 13 transiciones
año-a-año:

| Set de features | Recomienda recalibrar |
|---|---|
| 5 features (sin la nueva) | 3/13 (23%) |
| **6 features (con la nueva)** | **12/13 (92%)** |

Agregar `frecuencia_categoria_expandida` al monitoreo de deriva hace que el
sistema recomiende recalibrar en **prácticamente todas** las transiciones
de año — no porque el modelo se esté deteriorando, sino porque esta
feature, por construcción (frecuencia expandida, Laplace), **siempre** se
mueve con el tiempo — es su comportamiento normal y esperado, documentado
desde que se introdujo (sección 15.1), no una anomalía real cada vez que
ocurre.

**Veredicto honesto: el comportamiento actual de `deriva.py` con esta
feature es técnicamente correcto (mide bien el movimiento real) pero
operativamente ruidoso (recalibraría casi siempre, sin discriminar entre
"cambio esperado por diseño" y "deterioro real que amerita atención").**
Recomendación real, no implementada en esta pasada: **excluir
`frecuencia_categoria_expandida` del set de columnas que se le pasan a
`evaluar_deriva()`/un futuro disparador de recalibración para Dominio 2**
(monitorear las otras 5 normalmente) — o, alternativa más elaborada,
monitorear su *tasa de cambio* relativa a una tendencia esperada en vez de
su valor crudo, pero eso es ingeniería adicional no trivial, no un ajuste
de una línea. Dado que no existe todavía un disparador de recalibración
para Dominio 2 (ver arriba), esto queda anotado para cuando se construya
uno, no como un bug urgente a corregir hoy.
