# Referencia: cómo llegarían los datos en un despliegue online real

**Estado: documentación de referencia, no un plan activo.** SYNAPSE sigue en
fase de pruebas/investigación (ver `docs/PLAN_DE_TRABAJO.md` y
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`) — nada de esto está
implementado ni se está construyendo ahora. Esta nota existe para no perder
el razonamiento de una conversación puntual sobre cómo se vería la ingesta
de datos el día que se decida desplegar algo real, apoyado en lo que el
código de `src/` (Dominio 1 y Dominio 2) ya exige por diseño, no en
suposiciones.

## 1. Primero, una decisión que cambia todo: ¿bloquea la transacción o solo la observa?

- **Si SYNAPSE tiene que aprobar/rechazar ANTES de que la transacción se
  complete** (autorización real, como funciona una red de tarjetas) → no es
  streaming asíncrono, es una **llamada síncrona directa** (gRPC o REST): el
  procesador de pagos llama y espera la respuesta antes de seguir. Un
  manejador de eventos en el medio solo agregaría latencia que no se puede
  pagar en ese flujo.
- **Si SYNAPSE solo observa transacciones ya completadas** (para alertar,
  generar un caso de revisión manual, o alimentar `deriva.py`/el disparador
  de recalibración) → ahí sí tiene sentido un manejador de eventos
  asíncrono (sección 3).

## 2. Requisitos reales de llegada de datos, según lo que el código ya exige

No es teoría — son restricciones que `src/ejecutor.py`, `src/ciclo.py` y
`src/features_recursivas*.py` ya imponen hoy:

1. **Un evento a la vez, en orden temporal — nunca por lotes.** El Ejecutor
   procesa transacción por transacción. Un evento que llega "del pasado"
   después de uno más nuevo se **rechaza explícitamente**
   (`TiempoFueraDeOrden`) en vez de corromper el estado en silencio. Para
   Dominio 1 el orden que importa es global; para Dominio 2, por cuenta
   (`EstadoRecursivoPorCuenta`).
2. **Una sola línea de procesamiento por entidad.** El estado no es seguro
   para múltiples hilos a la vez, por diseño (ver docstring de
   `ejecutor.py`). Con varias instancias corriendo en paralelo, cada cuenta
   tiene que ir siempre al mismo "carril" para que su historial no se
   divida entre instancias distintas.
3. **El artefacto de política ya tiene que estar cargado antes de la primera
   transacción real** — se lee una vez al arrancar (vía el Puente) y se
   recarga cuando hay versión nueva, nunca a mitad de una decisión.
4. **Formato del evento**: solo los campos crudos que el modelo necesita
   (para Dominio 2: cuenta, monto, marca de tiempo, categoría) — el resto
   (EWMA, huella, etc.) lo calcula el Ejecutor mismo, no hace falta
   mandarlo precalculado.

En resumen: el patrón real es una **cola/stream de eventos, uno por
transacción, con orden garantizado al menos por entidad** — no un archivo
ni un lote procesado cada tanto.

## 3. Manejadores de eventos, comparados en lo que le importa a SYNAPSE

Eje de comparación: ¿garantiza orden por key? ¿permite un consumidor de
larga duración que mantenga estado en memoria (para no perder la ventaja de
microsegundos)?

| Opción | Orden por key | Mejor para | Nota |
|---|---|---|---|
| **Apache Kafka** (o gestionado: Confluent Cloud, Amazon MSK, Azure Event Hubs API Kafka) | Sí, por partición | El estándar real de la industria bancaria — muchos sistemas de fraude reales se construyen sobre esto | Partición = misma lógica de shard que Kinesis; un consumidor de larga duración por partición mantiene el estado en memoria |
| **Amazon Kinesis Data Streams** | Sí, por shard | Ecosistema AWS | Mismo patrón que Kafka (partition key → shard) |
| **Google Cloud Pub/Sub** (con ordering key) | Sí, con matices | Ecosistema GCP | Si un mensaje falla, los siguientes con la misma key se traban esperando — hay que manejar ese caso |
| **Azure Event Hubs** | Sí, por partición | Ecosistema Azure | Compatible con protocolo Kafka, mismo patrón |
| **Redis Streams** | Sí, por stream | Setups chicos/simples | Puede ser broker de eventos Y el lugar donde vive el estado compartido a la vez — un solo sistema en vez de dos |
| **NATS JetStream** | Sí, por subject | Latencia muy baja, sistemas livianos | Menos común en banca, técnicamente muy rápido |
| **RabbitMQ** | Parcial (requiere configuración extra) | Colas de trabajo tradicionales | El que peor encaja — no está pensado para "un log ordenado por entidad" |

**Trampa a evitar en cualquiera de estas opciones:** no usar un consumidor
sin estado (ej. AWS Lambda por defecto) si se quiere preservar la ventaja
de microsegundos — perdería el estado en memoria (`EstadoRecursivoPorCuenta`)
entre invocaciones. Alternativas: (a) un proceso de larga duración
(ECS/Fargate, EC2, Cloud Run con `min-instances`) manteniendo el estado en
memoria, o (b) externalizar el estado a Redis/DynamoDB/Memorystore, a costa
de pasar de microsegundos a milisegundos bajos por la ida y vuelta de red
(ver también el punto de Memorystore en la comparación de arquitectura GCP
de esta misma conversación).

## 4. Recomendación (solo como referencia futura)

- **Parte que decide/bloquea:** llamada directa (gRPC/REST), sin
  intermediario.
- **Parte que observa/alimenta bitácora y deriva:** Kafka (o su equivalente
  gestionado según el proveedor) — patrón más probado, partición por
  entidad, worker de larga duración con estado en memoria.
- **Si el volumen es chico** (demo/POC): Redis Streams, para no operar dos
  sistemas separados (cola + estado).

## 5. Costos (referencia GCP, revisar equivalentes en cada proveedor)

Para una demo/POC a bajo volumen, la mayoría de las piezas necesarias caben
en el *Always Free tier* de GCP sin costo real: Cloud Run (2M requests/mes),
BigQuery (1TB de queries/mes), Cloud Storage (5GB-mes), Cloud Scheduler (3
jobs/mes), Secret Manager. Las dos piezas que **siempre cobran algo**,
aunque el volumen sea bajo: Memorystore/Redis administrado (sin free tier,
~$35-50/mes mínimo, cobra por hora aunque esté inactivo) y Vertex AI
Pipelines (sin free tier, pero barato para corridas diarias — unos pocos
dólares al mes). Para quedarse en $0 real: saltar Memorystore (seguir con
estado en memoria de proceso) y correr el Calibrador como un job simple
disparado por scheduler en vez de un pipeline gestionado.
