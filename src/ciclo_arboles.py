"""CicloDecisionArboles — the sanctioned entry point for Domain 2, parallel
to `CicloDecision` (Domain 1, unmodified). Same 3 guarantees as its
counterpart (see `ciclo.py`): the artifact always goes through the Bridge,
the Executor's result always goes through `veto.py::evaluar` (reused
without modification -- it's model-independent by design), and any
Executor exception is turned into an escalation for manual review instead
of propagating.

**Why a parallel Cycle exists instead of a generic one**:
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` itself (section 6) points to
this as the right call -- generalize the interface only after a second
real domain, not before; forcing a common contract today between a
dot-product Executor and a lookup-table one would be abstracting without
proven need yet.
"""
from dataclasses import dataclass, field
from pathlib import Path

from src.artefacto_arboles import ArtefactoArbolesInvalido
from src.ejecutor_arboles import EjecutorArboles
from src.puente_arboles import leer_vigente
from src.veto import DecisionFinal, MONTO_MAXIMO_ABSOLUTO, evaluar as evaluar_veto


@dataclass
class CicloDecisionArboles:
    ruta_artefacto: Path
    monto_maximo: float = MONTO_MAXIMO_ABSOLUTO
    _ejecutor: EjecutorArboles = field(init=False, repr=False)

    def __post_init__(self) -> None:
        artefacto = leer_vigente(self.ruta_artefacto)  # can raise -- first load has no fallback
        self._ejecutor = EjecutorArboles(artefacto=artefacto)

    def recargar_artefacto(self) -> bool:
        """Same as `CicloDecision.recargar_artefacto()`: preserves the
        accumulated recursive state (global and per-account) and doesn't
        replace the current Executor if the new artifact is
        missing/corrupt/invalid."""
        try:
            artefacto_nuevo = leer_vigente(self.ruta_artefacto)
        except (FileNotFoundError, ArtefactoArbolesInvalido, ValueError):
            return False
        self._ejecutor = EjecutorArboles(
            artefacto=artefacto_nuevo,
            estado_global=self._ejecutor.estado_global,
            estado_cuenta=self._ejecutor.estado_cuenta,
        )
        return True

    def decidir(self, transaccion: dict) -> DecisionFinal:
        """The one method for making a real decision in Domain 2. `veto.py`
        expects the `Amount` key (Domain 1's contract) -- an `amt` alias is
        added here without modifying `veto.py`, which is intentionally
        independent of any domain's column schema."""
        try:
            resultado = self._ejecutor.decidir(transaccion)
        except Exception as e:
            return DecisionFinal(es_sospechosa=True, razon=f"error en el ejecutor: {e} — escalar a revisión manual", score=None)
        transaccion_para_veto = {**transaccion, "Amount": transaccion["amt"]}
        return evaluar_veto(resultado, transaccion_para_veto, monto_maximo=self.monto_maximo)
