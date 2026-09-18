# SYNAPSE — Claude Code Context

## Qué es este proyecto
Framework de decisión de dos velocidades: una capa lenta (Calibrador) que
aprende/calibra sin límite de tiempo, y una capa rápida (Ejecutor) que
decide en microsegundos aplicando lo ya aprendido, sin volver a pensar.
Ver `docs/PLAN_DE_TRABAJO.md` para el estado por fase.

Proyecto **independiente** — no comparte código con ningún proyecto anterior.

## Principios
- El Ejecutor nunca hace inferencia de modelo pesado, ni llamadas de red, ni
  reentrena nada — solo evalúa un artefacto de política ya calibrado contra
  un estado compacto.
- El Calibrador nunca decide en tiempo real — su única salida es un
  artefacto de política versionado (JSON).
- El Puente actualiza el artefacto de forma atómica (archivo temporal +
  reemplazo atómico) — el Ejecutor nunca debe poder leer un artefacto a
  medio escribir.
- La capa de Veto es independiente del modelo — sus invariantes se cumplen
  pase lo que pase en Calibrador/Ejecutor.
- Sin capa de interfaces/contrato genérico todavía: se construye concreto
  para el Dominio 1 (detección de fraude), y se generaliza solo después de
  un segundo dominio real (ver plan de Fase 6).
- Walk-forward / split temporal siempre — nunca split aleatorio sobre datos
  con orden temporal real (mismo principio de honestidad estadística que en
  proyectos anteriores).
- Cada fórmula/actualizador probado contra un caso numérico conocido de
  antemano, no solo "corre sin error".
- La velocidad se mide, nunca se asume — todo benchmark de latencia reporta
  un número real (microsegundos), no una estimación.

## Stack
Python 3.14. `scikit-learn` para el Calibrador (Fase 1). El dataset del
Dominio 1 es público, sin credenciales. Única excepción: `src/agente_triage.py`
(triage de escalamientos de Veto) usa `anthropic` + `ANTHROPIC_API_KEY` en
el entorno — nunca hardcoded, y solo en el cliente real
(`crear_cliente_claude()`); toda la lógica de auditoría se testea con un
cliente inyectable, sin llamada real (ver `docs/adr/0003-triage-agent.md`).

## Principios (agentes LLM)
- **Ningún LLM en el camino caliente.** `agente_triage.py` nunca se invoca
  desde `CicloDecision.decidir()` — corre después, sobre entradas ya
  escritas en `bitacora_decisiones.py`.
- **Ningún LLM decide ni bloquea nada.** Su output es una hipótesis para
  que un humano la verifique, nunca cambia estado del sistema.
- **Misma disciplina de 3 capas usada en un proyecto anterior propio**
  (schema determinista, grounding determinista, auditor selectivo —
  desacuerdo nunca se resuelve por mayoría), replicada como código nuevo:
  SYNAPSE no comparte código con otros proyectos.

## Estructura
Ver `README.md` para el árbol completo.

## Fase actual
Fase 0 — Fundamentos (dataset descargado y validado, ADRs en progreso).
Ver `docs/PLAN_DE_TRABAJO.md` para el estado detallado.
