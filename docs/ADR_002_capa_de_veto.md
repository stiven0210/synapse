# ADR 002 — Capa de veto (invariantes duros)

**Estado:** Decidido para esta iteración del Dominio 1 — con una limitación
de datos documentada explícitamente abajo.

**Contexto:** La capa de veto existe para proteger contra dos escenarios que
el Calibrador/Ejecutor no pueden cubrir por diseño: (a) el modelo está mal
calibrado o el artefacto llegó corrupto, y (b) hay reglas de negocio que
deben cumplirse siempre, independientes de qué tan bueno sea el modelo.

## Decisión — invariantes para esta iteración

1. **Artefacto ausente o corrupto → veto automático a "revisar manualmente".**
   Si `puente.leer_vigente()` falla o el artefacto no valida contra
   `ADR_001` (features y coeficientes de distinta longitud, score fuera de
   [0,1]), el Ejecutor nunca debe "seguir operando con lo último que
   recuerde" — eso es exactamente el tipo de estado implícito no auditable
   que causó el bug del kill switch en un proyecto anterior. Sin artefacto
   válido, no hay decisión automática.
2. **Monto absoluto extremo → veto a revisión, sin importar el score del
   modelo.** Umbral fijo, configurado explícitamente (no un valor por
   defecto silencioso) — ver `config` en Fase 4.
3. **Score del modelo fuera de rango numérico válido → veto.** Un score
   NaN o fuera de [0,1] no se trata como "0" ni como "1" — se trata como
   "no sé", que en un sistema de fraude significa escalar, no aprobar.

## Limitación de datos, documentada explícitamente (Fase 0)

El dataset de esta iteración (`docs/PLAN_DE_TRABAJO.md`, Credit Card Fraud
Detection) **no tiene identificador de tarjeta/cuenta** — cada fila es una
transacción anonimizada independiente, sin forma de agrupar por entidad.
Esto significa que, para esta iteración:

- El Ejecutor (Fase 2) mantiene **estadísticos recursivos globales**
  (EWMA de monto, conteo en ventana, sobre toda la población de
  transacciones), no por tarjeta — la arquitectura (actualización O(1) por
  evento nuevo) es la misma, pero pierde la dimensión de personalización
  por cuenta que un sistema de fraude real necesita.
- Invariantes de veto que dependerían de historial por cuenta (ej. "monto
  muy superior al promedio de *esa* tarjeta") **no se implementan en esta
  iteración** — quedan explícitamente pendientes para cuando se use un
  dataset con identificador de entidad real.

Esto no invalida el objetivo de Fase 5 (medir precisión/recall/latencia del
patrón de dos capas) — sí limita qué tan realista es la personalización por
cuenta hasta que se resuelva la fuente de datos.
