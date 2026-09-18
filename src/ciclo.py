"""CicloDecision — the one sanctioned entry point for making a real
decision. Closes 3 critical audit findings (see
`docs/PLAN_DE_TRABAJO.md`):

1. **Nothing stopped an `Ejecutor` from being built with an artifact that
   never went through `puente.leer_vigente()`** — a direct
   `Ejecutor(artefacto=calibrar(...))` was perfectly possible, skipping
   the Bridge's validation and atomic write. `CicloDecision` can only be
   built from a file path — the artifact ALWAYS goes through the Bridge.
2. **Nothing forced `Ejecutor.decidir()`'s result through
   `veto.evaluar()`** — and since `nan >= umbral` is `False` in Python, an
   invalid score was interpreted as "not suspicious" if someone used
   `Ejecutor.decidir()` directly. `CicloDecision.decidir()` always applies
   the Veto — it exposes no way to skip it.
3. **"Escalate for manual review" on a missing/corrupt artifact (ADR_002,
   invariant 1) didn't exist as code** — only as an uncaught exception
   that crashed the process. Here it's actually caught and turned into a
   real `DecisionFinal`, with graceful degradation: if the reload fails,
   it keeps operating on the last known valid artifact instead of
   crashing.
"""
from dataclasses import dataclass, field
from pathlib import Path

from src.artefacto import ArtefactoInvalido
from src.ejecutor import Ejecutor
from src.puente import leer_vigente
from src.veto import DecisionFinal, MONTO_MAXIMO_ABSOLUTO, evaluar as evaluar_veto


@dataclass
class CicloDecision:
    ruta_artefacto: Path
    monto_maximo: float = MONTO_MAXIMO_ABSOLUTO
    _ejecutor: Ejecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        artefacto = leer_vigente(self.ruta_artefacto)  # can raise -- first load has no fallback
        self._ejecutor = Ejecutor(artefacto=artefacto)

    def recargar_artefacto(self) -> bool:
        """Tries to reload the current artifact from the Bridge, preserving
        the accumulated recursive state (no history is lost from
        recalibrating). If it fails (missing file, corrupt, or invalid),
        it **does not replace** the current Executor — it keeps operating
        on the last known valid artifact. Returns whether the reload
        succeeded."""
        try:
            artefacto_nuevo = leer_vigente(self.ruta_artefacto)
        except (FileNotFoundError, ArtefactoInvalido, ValueError):
            return False
        self._ejecutor = Ejecutor(artefacto=artefacto_nuevo, estado=self._ejecutor.estado)
        return True

    def decidir(self, transaccion: dict) -> DecisionFinal:
        """The one method for making a real decision. Any Executor failure
        (invalid score, unexpected exception) is caught here and turned
        into an escalation for manual review — it's never propagated
        uncaught to the caller."""
        try:
            resultado = self._ejecutor.decidir(transaccion)
        except Exception as e:
            return DecisionFinal(es_sospechosa=True, razon=f"error en el ejecutor: {e} — escalar a revisión manual", score=None)
        return evaluar_veto(resultado, transaccion, monto_maximo=self.monto_maximo)
