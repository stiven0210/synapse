"""Ejecutor (capa rápida) — Fase 2. Recibe una transacción, actualiza el
estado compacto (`features_recursivas.py`) y aplica el artefacto de
política vigente para decidir. Nunca reentrena, nunca llama a red, nunca
carga scikit-learn — solo aritmética sobre el estado y los coeficientes ya
calibrados.

**`decidir()` es de bajo nivel — no es la API pública recomendada.** Puede
devolver un `score` inválido (NaN) sin que nada lo detenga aquí; en Python,
`nan >= umbral` es `False`, así que usar `resultado["es_sospechosa"]`
directo trataría un score inválido como "no sospechosa" — exactamente lo
que `ADR_002` prohíbe. La forma correcta de usar el Ejecutor es a través de
`ciclo.py::CicloDecision`, que fuerza el paso por `veto.py` (que sí
distingue NaN) y por el Puente. Este módulo se deja de bajo nivel a
propósito, para que Calibrador/Ejecutor sigan siendo independientes y
testeables por separado — la garantía de seguridad vive en `ciclo.py`, no
aquí.

**No es thread-safe, por diseño, no por descuido.** `EstadoRecursivoGlobal`
no tiene ninguna sincronización, y `TiempoFueraDeOrden` asume que las
transacciones llegan en un único stream secuencial, en orden de `Time` —
llamar `decidir()` desde múltiples hilos sobre la misma instancia no está
soportado (investigado con carga real concurrente, ver
`docs/PLAN_DE_TRABAJO.md`). No hace falta locking: el punto entero de
decisiones O(1) en microsegundos es que un solo hilo ya cubre volúmenes
reales de fraude con margen enorme. Si algún día hace falta paralelismo
real, la forma correcta es particionar por cuenta/entidad (cuando exista
ese identificador, ver limitación de Fase 0 en `ADR_002`), nunca compartir
una instancia de `Ejecutor` entre hilos.
"""
import copy
import math
from dataclasses import dataclass, field

from src.artefacto import ArtefactoInvalido, validar_artefacto
from src.features_recursivas import EstadoRecursivoGlobal

__all__ = ["ArtefactoInvalido", "Ejecutor", "validar_artefacto"]


def _sigmoide(z: float) -> float:
    """Forma numéricamente estable (evita overflow de exp para z muy negativo)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class Ejecutor:
    artefacto: dict
    estado: EstadoRecursivoGlobal = field(default_factory=EstadoRecursivoGlobal)

    def __post_init__(self) -> None:
        validar_artefacto(self.artefacto)
        # Copia defensiva: sin esto, si algo externo muta el mismo dict después de
        # construir el Ejecutor (logging, una capa de monitoreo, re-serialización) —
        # incluso poniendo umbral_decision fuera de [0,1] — decidir() seguiría usando
        # los valores mutados sin volver a validar. Hallazgo de la auditoría.
        self.artefacto = copy.deepcopy(self.artefacto)

    def decidir(self, transaccion: dict) -> dict:
        """`transaccion` trae al menos `Amount`, `Time`, y las columnas
        V1..V28 usadas por el artefacto vigente. O(1) — ninguna operación
        aquí depende del tamaño del historial. Ver advertencia del módulo:
        el resultado de esta función NO debe usarse directo, debe pasar
        por `veto.evaluar()`."""
        monto = transaccion["Amount"]
        tiempo = transaccion["Time"]
        monto_ewma, conteo_ventana = self.estado.leer_features(monto, tiempo)

        valores = dict(transaccion)
        valores["monto_ewma_global"] = monto_ewma
        valores["conteo_ventana_global"] = conteo_ventana

        z = self.artefacto["intercepto"]
        for nombre, coef in zip(self.artefacto["features"], self.artefacto["coeficientes"]):
            z += coef * valores[nombre]
        score = _sigmoide(z)

        # El estado se actualiza DESPUÉS de leer/decidir -- la transacción actual nunca
        # se ve a sí misma en su propio contexto reciente (misma semántica causal que
        # el modo batch usado para calibrar, ver features_recursivas.py).
        self.estado.actualizar(monto, tiempo)

        return {"score": score, "es_sospechosa": score >= self.artefacto["umbral_decision"]}
