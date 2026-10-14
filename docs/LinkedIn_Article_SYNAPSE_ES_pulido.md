# Tu modelo es demasiado lento para decidir. Tu camino rápido no tiene criterio. ¿Y si los separas?

Soy arquitecto de soluciones, y durante las últimas semanas he estado trabajando en un proyecto personal de investigación alrededor de una pregunta que aparece en casi todo sistema que decide en tiempo real: **¿hay que elegir entre velocidad y criterio?**

Este artículo cuenta lo que construí para responderla, los números que obtuve en tres problemas reales y, con el mismo detalle, lo que no funcionó.

---

## El dilema

Un banco tiene que aprobar o rechazar cada transacción de tarjeta dentro de la ventana de autorización: unos pocos milisegundos, que además se reparten entre red, validaciones y otros controles. Un sensor en una planta industrial lee vibración y presión varias veces por segundo, y una válvula que empieza a fallar no avisa. Un sistema de ciberseguridad tiene que detectar una intrusión antes de que el atacante se mueva lateralmente.

En los tres casos, el modelo que *debería* decidir (entrenado con historia y calibrado contra resultados reales) compite por un presupuesto de tiempo que no es suyo. Cada milisegundo que consume la inferencia es un milisegundo que no tiene el resto del sistema.

La salida habitual es un compromiso incómodo: el camino rápido corre con reglas fijas o umbrales estáticos, y el modelo bueno se usa "después, en batch". Lo que se gana en velocidad se pierde en criterio.

---

## La idea: separar aprender de decidir

Aprender y decidir tienen restricciones de tiempo muy distintas. Aprender puede tardar horas: procesa historia, ajusta parámetros y valida contra datos reservados. Decidir tiene que tardar microsegundos: recibe un evento y responde antes de que llegue el siguiente.

Si se fuerzan ambas cosas en el mismo componente, una pierde. SYNAPSE las separa en dos capas que **nunca comparten presupuesto de tiempo**:

- **Capa lenta (Calibrador).** Aprende offline, sin presión de tiempo, con el algoritmo que haga falta (regresión logística, gradient boosting, lo que el problema pida). Nunca toca el camino de la solicitud. Su única salida es un archivo JSON pequeño y versionado: el **Artefacto de Política**, con los parámetros ya aprendidos.
- **Capa rápida (Ejecutor).** Aplica ese artefacto con aritmética pura: multiplicar, sumar, comparar. Sin inferencia de modelo, sin llamadas de red, sin reentrenar. Decide en el orden de los microsegundos.
- **Puente.** Reemplaza el artefacto de forma atómica, así el Ejecutor nunca lee un archivo a medio escribir.

### Lo que casi ningún sistema tiene: un respaldo que no dependa del modelo

Un modelo rápido y bien calibrado sigue siendo un modelo, y puede equivocarse con total confianza: una mala calibración, un cambio en los datos, un bug silencioso.

Por eso SYNAPSE agrega una **Capa de Veto**: reglas duras, independientes del modelo, que se evalúan después de él y siempre pueden anular su decisión. Si el score no es un número válido (por ejemplo, un NaN), no se trata como "no sospechoso": se escala a revisión humana. Si el monto supera un límite absoluto, se marca sin importar lo que opine el modelo. Si el artefacto está corrupto, no se publica.

Completan el diseño dos piezas más. Un **Detector de Deriva** (PSI y Kolmogorov-Smirnov) decide *cuándo* recalibrar: no por calendario, sino cuando los datos indican que el modelo quedó desactualizado. Y una **Bitácora de Decisiones** registra cada decisión con la versión del artefacto que la produjo, para poder reconstruirla después ante un auditor o un regulador.

### ¿Y la IA generativa?

Es la primera pregunta que me hacen, así que la respondo de entrada: **ningún LLM participa en la decisión.** El único agente con LLM del proyecto trabaja *después*, sobre la bitácora, y solo propone hipótesis sobre fallas del sistema (por ejemplo, por qué un score salió inválido) para que una persona las verifique. Nunca aprueba, bloquea ni cambia nada, y sus respuestas pasan por validación de esquema, verificación contra los datos reales y un auditor selectivo antes de llegar a un humano.

---

## ¿Funciona? Tres problemas, medidos con honestidad

Lo validé en tres problemas de dos sectores (finanzas e industria), con datos de forma distinta y criterios de evaluación distintos. En los tres usé particiones temporales (entrenar con el pasado, evaluar con el futuro), nunca particiones aleatorias, y medí la latencia de punta a punta con `time.perf_counter()`, no la estimé.

### 1. Fraude con tarjeta: patrones globales

**El problema:** aprobar o rechazar cada transacción en tiempo real. Si el sistema es lento, la experiencia del cliente se resiente; si es permisivo, hay pérdidas; si es agresivo, se bloquean compras legítimas.

**Datos:** ULB Credit Card Fraud, el benchmark público de referencia (284.807 transacciones, 492 fraudes).

**Cómo lo resuelve SYNAPSE:** regresión logística calibrada offline. El Ejecutor la aplica sobre las variables del evento más dos variables recursivas (promedio móvil de montos y conteo en ventana de tiempo), calculadas igual en entrenamiento y en producción para no introducir diferencias entre ambos.

**Resultados:** F1 0,78 (precisión 0,93, recall 0,67), AUC-ROC 0,978. **8,33 µs por decisión.**

**Lo que no hace:** no supera el estado del arte en detección. Con recall 0,67, uno de cada tres fraudes pasa. La ventaja está en otra parte: latencia, y que cada decisión es reproducible y auditable.

### 2. Fraude por cuenta: comportamiento individual

**El problema:** los patrones globales no ven ataques personalizados. Una compra de 500 dólares es normal para un cliente y rara para otro. Hay que aprender el comportamiento de cada cuenta y seguir decidiendo en microsegundos.

**Datos:** Sparkov, un generador sintético (1.170.945 transacciones, 99 cuentas, 907 fraudes). Aporta lo que le faltaba al primer dataset: un identificador de cuenta.

**Cómo lo resuelve SYNAPSE:** gradient boosting con 6 variables, la más importante una "huella" de en qué categorías suele gastar cada cuenta. El Ejecutor mantiene estado por cuenta y evalúa el modelo como una tabla precalculada a partir de los umbrales reales de los árboles: sigue siendo aritmética.

**Resultados:** F1 0,85, AUC-PR 0,90 en test. **9,84 µs por decisión.**

**Lo que no hace:** probé un ataque simple, disfrazar el fraude con la categoría típica de la cuenta, y el recall cae de 82 % a 29,5 %. Intenté una mitigación: subió solo a 33,5 % y la descarté. Es una vulnerabilidad conocida y documentada. Además, los datos son sintéticos: con la prevalencia de fraude de un mercado real, la precisión sería menor.

### 3. Anomalías industriales (IoT)

**El problema:** detectar fallas de equipos antes de que ocurran. Una válvula que falla no se anuncia; muestra cambios sutiles en vibración, presión y caudal.

**Datos:** SKAB (Skoltech Anomaly Benchmark): 34 archivos con anomalías, 8 sensores a 1 Hz. Físicamente es otro mundo: lecturas continuas en vez de transacciones, fallas mecánicas en vez de un adversario.

**Cómo lo resuelve SYNAPSE:** el Calibrador calcula media y desviación de la operación normal; el Ejecutor marca una anomalía si algún sensor se aleja más de 3 desviaciones estándar. Es un detector **no supervisado**: nunca ve etiquetas de anomalía.

**Resultados:** F1 0,76 con la metodología oficial de SKAB. **Tercer lugar entre 9 métodos comparables** (8 publicados más SYNAPSE), a 2 puntos del mejor, una red neuronal (Conv-AE). Otros 2 métodos del leaderboard no pudieron alinearse con los datos y los excluí en vez de forzar la comparación. **13,05 µs por decisión.**

**Lo que no hace:** no gana. Probé dos mejoras (distancia de Mahalanobis y suavizado EWMA) y ninguna subió el F1. La lección fue otra: el detector más simple posible empató o superó a 6 de los 8 métodos publicados. Este dominio es una validación exploratoria: el veto físico que diseñé (un caudal negativo es imposible) nunca llegó a dispararse en los datos de prueba.

---

## Lo que muestran los tres juntos

**El patrón se trasladó sin cambiar el diseño.** Calibrador → Artefacto → Ejecutor → Veto funcionó igual para fraude global, fraude por cuenta y sensores industriales. Lo que cambió fue el modelo dentro del Calibrador, no la arquitectura.

**La latencia sale del diseño, no del problema.** Entre 8 y 13 µs por decisión en los tres casos, en Python y en un solo hilo. Es la consecuencia directa de que el Ejecutor solo haga aritmética sobre parámetros ya calibrados.

**El veto no tiene la misma madurez en todos.** En los dos dominios de fraude está implementado y cubierto por pruebas. En el industrial es un diseño exploratorio.

**Documentar lo que falla es parte del resultado.** Cada dominio registra lo que no funcionó junto a lo que sí. Sin eso, los números de arriba no servirían para tomar decisiones.

---

## Dónde más podría aplicar (sin validar todavía)

El patrón encaja donde se cumplen cuatro condiciones: una decisión por evento, un presupuesto de latencia que deja poco margen para inferencia, la necesidad de un respaldo que no dependa del modelo y la obligación de explicar cualquier decisión después.

- **Seguros y salud:** puntuar reclamos antes de que entren al flujo de pago, con trazabilidad para cumplimiento regulatorio.
- **E-commerce y pagos:** fraude sin tarjeta presente, toma de cuentas y bots durante el checkout.
- **Ciberseguridad:** detección de intrusiones y movimiento lateral; cuanto antes se detecta, menor el daño.
- **Telecomunicaciones:** SIM swap y fraude de roaming.
- **Logística y energía:** anomalías en rutas o en la cadena de frío, y fraude de medidores inteligentes.

Ninguno de estos está probado. Son hipótesis, y la forma de confirmarlas es la misma que en los tres dominios anteriores: un dataset real, una partición temporal y latencia medida.

---

## El proyecto

SYNAPSE es open source (licencia MIT). Tiene 187 pruebas (2 se omiten si no está el dataset local de Sparkov) y una demo que corre en segundos sin descargar nada. El repositorio incluye la arquitectura en formato C4, tres ADRs que explican el *porqué* de las decisiones de diseño y notas de investigación por dominio, incluido todo lo que no funcionó.

GitHub: https://github.com/stiven0210/synapse

Si diseñas o operas sistemas que deciden en tiempo real, me interesa tu experiencia concreta: **¿qué hace tu sistema cuando el modelo devuelve algo que no tiene sentido?** ¿Lo detecta, lo escala o lo deja pasar? Te leo en los comentarios.

---

*Genes Stivens Benavides Angel — Arquitecto de Soluciones. Proyecto personal de investigación.*
