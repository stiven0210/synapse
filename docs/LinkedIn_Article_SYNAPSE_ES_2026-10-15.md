# Tu modelo es demasiado lento para decidir. Tu camino rápido no tiene criterio. ¿Y si los separas?

**Para quién es:** para quien diseña u opera sistemas que deciden en tiempo real: pagos, fraude, monitoreo industrial, seguridad. No hace falta saber de machine learning; cada término técnico se explica la primera vez que aparece.

**Por qué lo escribo:** soy arquitecto de soluciones y SYNAPSE es mi proyecto personal de investigación. Busca responder una pregunta: **¿hay que elegir entre velocidad y criterio?** Comparto la respuesta para contrastarla con la experiencia de quienes operan estos sistemas a diario.

**En 30 segundos:**

- **La idea:** separar el componente que *aprende* (lento, sin límite de tiempo) del que *decide* (rápido, solo aplica lo aprendido), con reglas duras detrás que no dependen del modelo.
- **El resultado:** funcionó en tres problemas reales con datos públicos. Cada decisión tarda entre 8 y 13 microsegundos, más de 75 veces menos que un milisegundo.
- **El límite:** en ninguno supera al mejor método publicado en calidad de detección. La ganancia está en la latencia, la trazabilidad y el respaldo ante fallos del modelo.
- **Lo que te pido:** al final hay una pregunta concreta. Tu respuesta me sirve más que un "me gusta".

---

## El dilema: el mejor modelo no cabe en el tiempo disponible

Si tu sistema decide en tiempo real, probablemente ya convives con este problema: el modelo que mejor decide es el que menos tiempo tiene para hacerlo.

Tres ejemplos con la misma forma:

- **Pagos:** un banco aprueba o rechaza cada transacción de tarjeta en pocos milisegundos. Ese tiempo se reparte además entre red, validaciones y otros controles.
- **Industria:** un sensor lee vibración y presión varias veces por segundo, y una válvula que empieza a fallar no avisa.
- **Ciberseguridad:** hay que detectar una intrusión antes de que el atacante salte a otros equipos.

En los tres casos, el modelo que *debería* decidir (entrenado con historia y calibrado contra resultados reales) compite por un tiempo que no es suyo. Cada milisegundo que usa se le quita al resto del sistema.

La salida habitual es un compromiso: el camino rápido usa reglas fijas o umbrales estáticos, y el modelo bueno se ejecuta después, por lotes. Se gana velocidad y se pierde criterio. Si tu sistema funciona así, lo que sigue es una alternativa.

---

## La idea: que el componente que decide no tenga que pensar

La propuesta de SYNAPSE es dejar de pedirle a un mismo componente que aprenda y decida, porque cada tarea necesita un tiempo muy distinto. Aprender puede tardar horas: procesa historia, ajusta parámetros y valida contra datos reservados. Decidir tiene que tardar microsegundos (millonésimas de segundo): recibe un evento y responde antes del siguiente.

Por eso el sistema se divide en capas que **nunca comparten presupuesto de tiempo**:

- **Capa lenta (Calibrador).** Aprende fuera de línea, sin prisa, con el algoritmo que el problema pida. Nunca interviene cuando llega una solicitud. Su única salida es un archivo JSON pequeño y versionado, el **Artefacto de Política**, con los parámetros ya aprendidos.
- **Capa rápida (Ejecutor).** Aplica ese artefacto con aritmética pura: multiplicar, sumar, comparar. No ejecuta modelos pesados, no hace llamadas de red y no reentrena nada.
- **Puente.** Sustituye el artefacto de forma atómica (todo o nada), así el Ejecutor nunca lee un archivo a medio escribir.

Qué ganas con esta separación: puedes usar el mejor modelo que encuentres sin preocuparte por su velocidad, porque nunca corre en el camino crítico. Y como lo aprendido vive en un archivo versionado, sabes exactamente con qué versión se tomó cada decisión.

### El respaldo: qué pasa cuando el modelo se equivoca

Un modelo rápido y bien calibrado sigue siendo un modelo, y puede equivocarse con total confianza por una mala calibración, un cambio en los datos o un error silencioso. Esta parte del diseño existe para ese momento, y es la que casi ningún sistema tiene. Cada pieza cubre un fallo distinto:

- **Capa de Veto: el modelo da una respuesta absurda.** Son reglas duras que se evalúan después del modelo y siempre pueden anular su decisión. Si el puntaje no es un número válido (por ejemplo, un NaN, "no es un número"), no se trata como "no sospechoso": se escala a revisión humana. Si el monto supera un límite absoluto, se marca sin importar lo que opine el modelo. Si el artefacto está corrupto, no se publica.
- **Detector de Deriva: el mundo cambió y el modelo no.** Decide *cuándo* recalibrar: no por calendario, sino cuando dos pruebas estadísticas (PSI y Kolmogorov-Smirnov) indican que los datos ya no se parecen a los del entrenamiento.
- **Bitácora de Decisiones: alguien pregunta después por qué.** Registra cada decisión con la versión del artefacto que la produjo, para reconstruirla ante un auditor o un regulador.

La regla clave es que el veto no depende del modelo: sus reglas se cumplen aunque el Calibrador o el Ejecutor fallen.

### ¿Y la IA generativa? Fuera de la decisión, a propósito

Es la primera pregunta que me hacen, así que la respondo de entrada: **ningún LLM participa en la decisión.** Un LLM tarda segundos, depende de la red y puede responder distinto ante la misma entrada. Son justo las tres cosas que el camino rápido no puede permitirse.

El único agente con LLM del proyecto trabaja *después*, sobre la bitácora. Cuando el veto escala un caso (por ejemplo, un puntaje inválido), el agente propone una hipótesis sobre la causa para que una persona la verifique. Nunca aprueba, bloquea ni cambia nada. Antes de llegar a un humano, su respuesta pasa tres filtros: validación de esquema, verificación contra los datos reales y, en los casos que lo requieren, un auditor que la revisa.

---

## Cómo leer los resultados

Lo validé en tres problemas de dos sectores, finanzas e industria. No los elegí al azar: cada uno pone a prueba algo que el anterior no probó.

1. **Fraude global:** ¿el enfoque aguanta frente al conjunto de datos de referencia del sector?
2. **Fraude por cuenta:** ¿sigue siendo rápido cuando tiene que recordar el comportamiento de cada cliente?
3. **Anomalías industriales:** ¿el patrón funciona fuera de las finanzas, sin adversario y sin ejemplos etiquetados?

Dos reglas de método valen para los tres. Sin ellas las cifras no significan nada:

- **Partición temporal:** se entrena con el pasado y se evalúa con el futuro, nunca con una mezcla aleatoria. Así el resultado se parece a lo que pasaría en producción, donde el futuro no se conoce.
- **Latencia medida:** cada tiempo es una medición real con `time.perf_counter()`, en Python y en un solo hilo, no una estimación. En un lenguaje compilado debería ser menor, pero no lo he medido.

Y un mini glosario para las métricas:

- **Precisión:** de las alertas que da el sistema, qué fracción era real.
- **Recall:** de los casos reales, qué fracción detectó.
- **F1:** combina precisión y recall en un solo número de 0 a 1.
- **AUC-PR y AUC-ROC:** resumen la calidad del modelo para todos los umbrales posibles, no solo el elegido.

### 1. Fraude con tarjeta: patrones globales

**Qué pone a prueba:** si el enfoque es competitivo en el problema más estudiado del sector, con el conjunto de datos que todos usan como referencia.

**El problema:** aprobar o rechazar cada transacción en tiempo real. Si el sistema es lento, el cliente lo nota; si es permisivo, hay pérdidas; si es agresivo, se bloquean compras legítimas.

**Datos:** ULB Credit Card Fraud, con 284.807 transacciones de las que solo 492 son fraude (el 0,17 %). Con tan pocos fraudes, un modelo que lo aprobara todo acertaría el 99,8 % de las veces y no serviría de nada. Por eso se miden precisión y recall, no el acierto total.

**Cómo lo resuelve SYNAPSE:** regresión logística calibrada fuera de línea. El Ejecutor la aplica sobre las variables de la transacción más dos variables de historia reciente: el promedio móvil de montos y el número de transacciones en una ventana de tiempo. Se calculan igual en entrenamiento y en producción, para que el modelo vea en producción lo mismo que vio al aprender.

**Resultados:** F1 0,78 (precisión 0,93, recall 0,67) y AUC-ROC 0,978. **8,33 µs por decisión.** En la práctica, 93 de cada 100 alertas son fraude real.

**Lo que no hace:** no supera el estado del arte en detección. Con recall 0,67, uno de cada tres fraudes pasa. La ventaja está en otra parte: la latencia, y que cada decisión es reproducible y auditable.

### 2. Fraude por cuenta: comportamiento individual

**Qué pone a prueba:** si el Ejecutor sigue decidiendo en microsegundos cuando, además, tiene que recordar cómo se comporta cada cliente.

**El problema:** los patrones globales no detectan ataques hechos a la medida de una cuenta. Una compra de 500 dólares es normal para un cliente y rara para otro.

**Datos:** Sparkov, un generador de datos sintéticos (1.170.945 transacciones, 99 cuentas, 907 fraudes). Lo elegí porque aporta lo que le faltaba al primer conjunto: un identificador de cuenta. Hay pocos conjuntos públicos de fraude con esa información.

**Cómo lo resuelve SYNAPSE:** gradient boosting (un conjunto de árboles de decisión) con 6 variables. La más importante es una "huella" de las categorías en las que suele gastar cada cuenta. El Ejecutor guarda el historial de cada cuenta y evalúa el modelo como una tabla precalculada a partir de los umbrales de los árboles, así que sigue siendo aritmética.

**Resultados:** F1 0,85 y AUC-PR 0,90 en el periodo de prueba. **9,84 µs por decisión**, solo 1,5 µs más que el dominio 1 pese a recordar cada cuenta.

**Lo que no hace:** probé un ataque simple, disfrazar el fraude con la categoría típica de la cuenta, y el recall cae del 82 % al 29,5 %. Intenté una mitigación que solo lo subió al 33,5 %, y la descarté. Es una vulnerabilidad conocida y documentada. Además, los datos son sintéticos: con la proporción de fraude de un mercado real, la precisión sería menor. Esta cifra demuestra que el diseño funciona con historial por cuenta, no cuánto fraude detectaría en un banco.

### 3. Anomalías industriales (IoT)

**Qué pone a prueba:** si el patrón sirve fuera de las finanzas, en un problema físico, sin adversario y sin ejemplos etiquetados de fallos.

**El problema:** detectar fallos de equipos antes de que ocurran. Una válvula que falla no se anuncia: muestra cambios sutiles en vibración, presión y caudal.

**Datos:** SKAB (Skoltech Anomaly Benchmark), un conjunto público de referencia con tabla oficial de resultados. Tiene 34 archivos con anomalías y 8 sensores leídos una vez por segundo. Son lecturas continuas en vez de transacciones, y fallos mecánicos en vez de un atacante.

**Cómo lo resuelve SYNAPSE:** el Calibrador calcula la media y la desviación de cada sensor en operación normal. El Ejecutor marca una anomalía si algún sensor se aleja más de 3 desviaciones estándar. Es un detector **no supervisado**: nunca ve ejemplos de fallos, solo aprende cómo es lo normal. Eso importa porque en una planta real casi nunca hay historial etiquetado de fallos.

**Resultados:** F1 0,76 con la metodología oficial de SKAB. **Tercer lugar entre 9 métodos comparables** (8 publicados más SYNAPSE), a 2 puntos del mejor, una red neuronal (Conv-AE). Otros 2 métodos de la tabla oficial no se podían alinear con los datos y los excluí en vez de forzar la comparación. **13,05 µs por decisión.**

**Lo que no hace:** no gana. Probé dos mejoras (distancia de Mahalanobis y suavizado EWMA) y ninguna subió el F1. La lección fue otra: el detector más simple posible empató o superó a 6 de los 8 métodos publicados. Este dominio es exploratorio: el veto físico que diseñé (un caudal negativo es imposible) nunca se activó en los datos de prueba, así que aquí el veto sigue sin probarse.

---

## Lo que muestran los tres juntos

La respuesta a la pregunta inicial es que no hace falta elegir entre velocidad y criterio, siempre que aceptes una condición: el criterio se construye antes, no en el momento de decidir.

| Dominio | Qué pone a prueba | Datos | Calidad de detección | Latencia por decisión |
| --- | --- | --- | --- | --- |
| Fraude global | Conjunto de referencia del sector | ULB, 284.807 transacciones | F1 0,78 | 8,33 µs |
| Fraude por cuenta | Historial por cliente | Sparkov, 1.170.945 transacciones (sintético) | F1 0,85 · AUC-PR 0,90 | 9,84 µs |
| Anomalías industriales | Otro sector, sin etiquetas | SKAB, 34 archivos | F1 0,76 (3.º de 9) | 13,05 µs |

**El diseño se trasladó sin cambios.** Calibrador → Artefacto → Ejecutor → Veto funcionó igual en los tres. Solo cambió el modelo dentro del Calibrador.

**La latencia sale del diseño, no del problema.** Los tres quedan entre 8 y 13 µs porque el Ejecutor solo hace aritmética sobre parámetros ya calibrados, sea cual sea el modelo.

**El veto no tiene la misma madurez en todos.** En los dos dominios de fraude está implementado y cubierto por pruebas. En el industrial es un diseño exploratorio.

**Documentar lo que falla es parte del resultado.** Cada dominio registra lo que no funcionó junto a lo que sí. Sin eso no podrías decidir si estas cifras aplican a tu caso.

---

## ¿Aplica a tu caso?

Esta sección es para que evalúes tu propio sistema. El patrón encaja si respondes "sí" a estas cuatro preguntas:

1. ¿Tomas una decisión por cada evento (transacción, lectura, solicitud)?
2. ¿Tu presupuesto de latencia deja poco margen para ejecutar un modelo?
3. ¿Necesitas un respaldo que siga funcionando aunque el modelo falle?
4. ¿Tienes que poder explicar cualquier decisión después, ante un auditor o un cliente?

Con esas condiciones, estos sectores son candidatos naturales:

- **Seguros y salud:** puntuar reclamos antes de que entren al flujo de pago, con trazabilidad para el regulador. Para dar escala: la Coalition Against Insurance Fraud estima que el fraude de seguros cuesta 308.600 millones de dólares al año en Estados Unidos ([estudio de 2022](https://insurancefraud.org/wp-content/uploads/The-Impact-of-Insurance-Fraud-on-the-U.S.-Economy-Report-2022-8.26.2022.pdf)). Y en junio de 2025 el Departamento de Justicia de ese país acusó a 324 personas por esquemas de fraude en salud con más de 14.600 millones de dólares en pérdidas pretendidas ([comunicado oficial](https://www.justice.gov/opa/pr/national-health-care-fraud-takedown-results-324-defendants-charged-connection-over-146)).
- **Comercio electrónico y pagos:** fraude sin tarjeta presente, robo de cuentas y bots durante el pago.
- **Ciberseguridad:** detección de intrusiones; cuanto antes se detectan, menor el daño.
- **Telecomunicaciones:** duplicación fraudulenta de SIM (SIM swap) y fraude de roaming.
- **Logística y energía:** anomalías en rutas o en la cadena de frío, y fraude en medidores inteligentes.

Ninguno está probado todavía: son hipótesis. Se confirman igual que los tres dominios anteriores, con un conjunto de datos real, una partición temporal y latencia medida. Si trabajas en uno de ellos y tienes datos, me interesa hablar.

---

## El proyecto y una pregunta para ti

SYNAPSE es código abierto (licencia MIT), y todo lo que describo aquí corresponde a la versión 1.0.1. Según lo que busques, este es el punto de entrada:

- **Verlo funcionar:** una demo que corre en unos 5 segundos sin descargar datos.
- **Entender el porqué del diseño:** tres registros de decisiones de arquitectura (ADR) y la arquitectura en formato C4.
- **Revisar si las cifras se sostienen:** 187 pruebas automatizadas (2 se omiten si no está descargado el conjunto Sparkov) y notas de investigación por dominio, con todo lo que no funcionó.

GitHub: https://github.com/stiven0210/synapse

Y la pregunta que te prometí al principio.

Si diseñas u operas sistemas que deciden en tiempo real: **¿qué hace tu sistema cuando el modelo devuelve algo que no tiene sentido?** ¿Lo detecta, lo escala o lo deja pasar? Te leo en los comentarios.

---

*Genes Stivens Benavides Angel — Arquitecto de Soluciones. Proyecto personal de investigación.*
