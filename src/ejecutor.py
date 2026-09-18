"""Executor (fast layer) — Phase 2. Receives a transaction, updates the
compact state (`features_recursivas.py`), and applies the current policy
artifact to decide. Never retrains, never makes a network call, never
loads scikit-learn — just arithmetic over the state and the
already-calibrated coefficients.

**`decidir()` is low-level — it is not the recommended public API.** It
can return an invalid `score` (NaN) with nothing stopping it here; in
Python, `nan >= umbral` is `False`, so using `resultado["es_sospechosa"]`
directly would treat an invalid score as "not suspicious" — exactly what
`ADR_002` prohibits. The correct way to use the Executor is through
`ciclo.py::CicloDecision`, which forces the call through `veto.py` (which
does distinguish NaN) and through the Bridge. This module is deliberately
left low-level, so the Calibrator/Executor stay independent and
separately testable — the safety guarantee lives in `ciclo.py`, not here.

**Not thread-safe, by design, not by oversight.** `EstadoRecursivoGlobal`
has no synchronization at all, and `TiempoFueraDeOrden` assumes
transactions arrive on a single sequential stream, in `Time` order —
calling `decidir()` from multiple threads on the same instance is not
supported (investigated under real concurrent load, see
`docs/PLAN_DE_TRABAJO.md`). No locking is needed: the entire point of O(1)
microsecond decisions is that a single thread already covers real fraud
volumes with enormous headroom. If real parallelism is ever needed, the
right approach is to partition by account/entity (once that identifier
exists, see the Phase 0 limitation in `ADR_002`), never to share a single
`Ejecutor` instance across threads.
"""
import copy
import math
from dataclasses import dataclass, field

from src.artefacto import ArtefactoInvalido, validar_artefacto
from src.features_recursivas import EstadoRecursivoGlobal

__all__ = ["ArtefactoInvalido", "Ejecutor", "validar_artefacto"]


def _sigmoide(z: float) -> float:
    """Numerically stable form (avoids exp overflow for very negative z)."""
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
        # Defensive copy: without this, if something external mutates the same dict after
        # the Executor is built (logging, a monitoring layer, re-serialization) —
        # even setting umbral_decision outside [0,1] — decidir() would keep using
        # the mutated values without re-validating. An audit finding.
        self.artefacto = copy.deepcopy(self.artefacto)

    def decidir(self, transaccion: dict) -> dict:
        """`transaccion` carries at least `Amount`, `Time`, and the
        V1..V28 columns the current artifact uses. O(1) — no operation
        here depends on history size. See the module warning: this
        function's result must NOT be used directly, it must go through
        `veto.evaluar()`."""
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

        # State is updated AFTER reading/deciding -- the current transaction never
        # sees itself in its own recent context (same causal semantics as
        # the batch mode used for calibration, see features_recursivas.py).
        self.estado.actualizar(monto, tiempo)

        return {"score": score, "es_sospechosa": score >= self.artefacto["umbral_decision"]}
