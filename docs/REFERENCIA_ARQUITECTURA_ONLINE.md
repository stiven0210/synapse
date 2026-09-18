# Reference: how data would arrive in a real online deployment

**Status: reference documentation, not an active plan.** SYNAPSE is still in
its testing/research phase (see `docs/PLAN_DE_TRABAJO.md` and
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`) — none of this is
implemented or being built right now. This note exists so the reasoning
from a specific conversation about what data ingestion would look like the
day a real deployment is decided doesn't get lost — grounded in what the
code in `src/` (Domain 1 and Domain 2) already requires by design, not in
assumptions.

## 1. First, a decision that changes everything: does it block the transaction, or just observe it?

- **If SYNAPSE has to approve/reject BEFORE the transaction completes**
  (real authorization, the way a card network works) → this isn't async
  streaming, it's a **direct synchronous call** (gRPC or REST): the payment
  processor calls and waits for the response before continuing. An event
  handler in the middle would only add latency that flow can't afford.
- **If SYNAPSE only observes already-completed transactions** (to alert,
  generate a manual-review case, or feed `deriva.py`/the recalibration
  trigger) → then an async event handler makes sense (section 3).

## 2. Real data-arrival requirements, based on what the code already demands

This isn't theory — these are constraints `src/ejecutor.py`, `src/ciclo.py`,
and `src/features_recursivas*.py` already impose today:

1. **One event at a time, in temporal order — never in batches.** The
   Executor processes transaction by transaction. An event that arrives
   "from the past" after a newer one is **explicitly rejected**
   (`TiempoFueraDeOrden`) instead of silently corrupting state. For
   Domain 1 the order that matters is global; for Domain 2, per account
   (`EstadoRecursivoPorCuenta`).
2. **A single processing lane per entity.** State is not safe for
   multiple threads at once, by design (see `ejecutor.py`'s docstring).
   With several instances running in parallel, each account always has to
   land on the same "lane" so its history doesn't get split across
   different instances.
3. **The policy artifact must already be loaded before the first real
   transaction** — it's read once at startup (via the Bridge) and reloaded
   when a new version exists, never mid-decision.
4. **Event format**: only the raw fields the model needs (for Domain 2:
   account, amount, timestamp, category) — everything else (EWMA,
   fingerprint, etc.) is computed by the Executor itself; there's no need
   to send it pre-computed.

In short: the real pattern is an **event queue/stream, one per
transaction, with order guaranteed at least per entity** — not a file or a
batch processed every so often.

## 3. Event handlers, compared on what actually matters to SYNAPSE

Comparison axis: does it guarantee order per key? Does it allow a
long-lived consumer that keeps state in memory (so the microsecond
advantage isn't lost)?

| Option | Order per key | Best for | Note |
|---|---|---|---|
| **Apache Kafka** (or managed: Confluent Cloud, Amazon MSK, Azure Event Hubs Kafka API) | Yes, per partition | The real standard in the banking industry — many real fraud systems are built on this | Partition = same sharding logic as Kinesis; one long-lived consumer per partition keeps state in memory |
| **Amazon Kinesis Data Streams** | Yes, per shard | AWS ecosystem | Same pattern as Kafka (partition key → shard) |
| **Google Cloud Pub/Sub** (with ordering key) | Yes, with caveats | GCP ecosystem | If one message fails, the following ones with the same key get stuck waiting — has to be handled |
| **Azure Event Hubs** | Yes, per partition | Azure ecosystem | Kafka-protocol compatible, same pattern |
| **Redis Streams** | Yes, per stream | Small/simple setups | Can be the event broker AND where the shared state lives, at once — one system instead of two |
| **NATS JetStream** | Yes, per subject | Very low latency, lightweight systems | Less common in banking, technically very fast |
| **RabbitMQ** | Partial (needs extra config) | Traditional work queues | The worst fit — not designed for "a log ordered per entity" |

**Trap to avoid with any of these:** don't use a stateless consumer (e.g.
AWS Lambda by default) if you want to keep the microsecond advantage — it
would lose the in-memory state (`EstadoRecursivoPorCuenta`) between
invocations. Alternatives: (a) a long-lived process (ECS/Fargate, EC2,
Cloud Run with `min-instances`) keeping state in memory, or (b)
externalize state to Redis/DynamoDB/Memorystore, at the cost of going from
microseconds to low milliseconds because of the network round trip (see
also the Memorystore point in this same conversation's GCP architecture
comparison).

## 4. Recommendation (reference only, for the future)

- **The part that decides/blocks:** a direct call (gRPC/REST), with no
  intermediary.
- **The part that observes/feeds the log and drift detection:** Kafka (or
  its managed equivalent depending on the provider) — the most proven
  pattern, partitioned by entity, a long-lived worker with in-memory
  state.
- **If volume is small** (demo/POC): Redis Streams, to avoid running two
  separate systems (queue + state).

## 5. Costs (GCP reference, check the equivalents for each provider)

For a low-volume demo/POC, most of the necessary pieces fit in GCP's
*Always Free* tier at no real cost: Cloud Run (2M requests/month),
BigQuery (1TB of queries/month), Cloud Storage (5GB-month), Cloud
Scheduler (3 jobs/month), Secret Manager. The two pieces that **always
cost something**, even at low volume: managed Memorystore/Redis (no free
tier, ~$35-50/month minimum, billed hourly even while idle) and Vertex AI
Pipelines (no free tier, but cheap for daily runs — a few dollars a
month). To stay at real $0: skip Memorystore (keep in-process memory
state) and run the Calibrator as a simple scheduler-triggered job instead
of a managed pipeline.
