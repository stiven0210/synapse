# ADR 003 — Agente de triage sobre escalamientos operativos de Veto

**Estado:** Decidido, con alcance deliberadamente acotado.

**Contexto:** Se evaluó si vale la pena agregar componentes agénticos (LLM)
a SYNAPSE. De 5 roles candidatos discutidos (narrar reportes de deriva,
auditar artefactos antes de publicar, triar escalamientos de Veto, explicar
transacciones marcadas individualmente, proponer parámetros no calibrados),
solo uno tenía valor real no cubierto ya por lógica determinista: **triar
los escalamientos de Veto que representan una falla de *sistema*, no un
juicio sobre una transacción.**

De los 4 tipos de `DecisionFinal` (ver `src/bitacora_decisiones.py`), solo
2 califican: `SCORE_INVALIDO` y `ERROR_EJECUTOR`. `MODELO` no necesita
triage (es el camino normal) y `MONTO_EXCEDE_LIMITE` es autoexplicativo en
el propio dato (el monto ya lo dice todo) — meter un LLM ahí sería
wrapping cosmético de algo ya determinista.

## Decisión

1. **El agente nunca participa en el camino caliente.** No se invoca desde
   `CicloDecision.decidir()` ni desde `Ejecutor`/`Veto`. Corre como paso
   posterior, sobre entradas ya escritas en la bitácora — mismo principio
   que `disparador_recalibracion.py` (capa lenta, nunca capa rápida).
2. **Nunca decide ni bloquea nada.** El output es una hipótesis para que un
   humano la verifique — no aprueba, no rechaza, no cambia ningún estado.
3. **Misma disciplina de 3 capas que `an earlier project`/an earlier project
   (`an ADR from an earlier project` de an earlier project)**, replicada como
   código nuevo e independiente (SYNAPSE no comparte código con otros
   proyectos, `CLAUDE.md`):
   - Capa 1 (determinista, 100% de las respuestas): schema estricto —
     JSON bien formado, campos presentes, enums válidos, confianza en
     [0,1]. Costo cero, sin LLM.
   - Capa 2 (determinista, barata, 100% de las salidas válidas de Capa 1):
     cada afirmación en `evidencia_citada` debe compartir vocabulario real
     con los datos que de verdad se le dieron al agente (`entrada` +
     `contexto`) — no NLP sofisticado a propósito, detecta el caso obvio
     de alucinación total desconectada de los datos.
   - Capa 3 (selectiva — solo si falla Capa 2 o por muestreo aleatorio):
     un segundo LLM audita la hipótesis contra los mismos datos.
     Desacuerdo proponente/auditor **nunca se resuelve por mayoría** — se
     descarta el ciclo completo.
4. **Rate limiting** (`src/limitador_llamadas.py`, mismo patrón que
   `an earlier project/src/agents/rate_limiter.py`): presupuesto diario, agotarlo
   lanza una excepción en vez de seguir gastando.
5. **Circuit breaker** (`CircuitoTriage`): si la tasa de descarte (Capa
   1/Capa 3 fallida) de los últimos N triages supera un umbral, el agente
   se apaga solo (lanza `CircuitoAbierto` antes de gastar en una llamada
   más) — el llamador cae al reporte plano determinista (la entrada de
   bitácora tal cual, sin narrativa), nunca se bloquea el flujo de
   escalamiento real.

## Plan de pruebas — 3 niveles (discutido en conversación, no solo tests unitarios)

1. **Determinista, 100% testeable con cliente LLM inyectable (falso), sin
   llamadas reales** — igual que los tests del agente en an earlier project:
   schema inválido se descarta en Capa 1, grounding fallido dispara Capa 3,
   desacuerdo descarta sin resolver por mayoría, rate limiter agota
   presupuesto, circuit breaker se abre con tasa de descarte simulada.
2. **Casos de causa real conocida** (ver `tests/test_bitacora_decisiones.py`,
   los 2 escenarios de causa real: feature en NaN → `SCORE_INVALIDO`,
   feature faltante → `ERROR_EJECUTOR`) — se verifica que, con un cliente
   LLM falso que devuelve una hipótesis correcta, el agente la deja pasar;
   y que una hipótesis que inventa una causa no presente en los datos
   reales queda atrapada por Capa 2/Capa 3.
3. **Lo que no se puede validar offline**: si esto reduce en la práctica el
   esfuerzo de un humano investigando un incidente real solo se sabe con
   uso real — no hay forma honesta de simularlo con un dataset histórico
   sin incidentes reales. Se documenta como pendiente de validar en uso,
   no se finge que el test automatizado ya probó valor humano.
