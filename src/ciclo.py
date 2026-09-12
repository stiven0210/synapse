"""CicloDecision — único punto de entrada sancionado para tomar una
decisión real. Cierra 3 hallazgos críticos de la auditoría (ver
`docs/PLAN_DE_TRABAJO.md`):

1. **Nada impedía construir un `Ejecutor` con un artefacto que nunca pasó
   por `puente.leer_vigente()`** — `Ejecutor(artefacto=calibrar(...))`
   directo era perfectamente posible, sin la validación ni la escritura
   atómica del Puente. `CicloDecision` solo se construye desde una ruta de
   archivo — el artefacto SIEMPRE pasa por el Puente.
2. **Nada obligaba a que el resultado de `Ejecutor.decidir()` pasara por
   `veto.evaluar()`** — y como `nan >= umbral` es `False` en Python, un
   score inválido se interpretaba como "no sospechosa" si alguien usaba
   `Ejecutor.decidir()` directo. `CicloDecision.decidir()` siempre aplica
   el Veto — no expone ninguna forma de saltárselo.
3. **"Revisar manualmente" por artefacto ausente/corrupto (ADR_002,
   invariante 1) no existía como código** — solo como una excepción sin
   capturar que tumbaba el proceso. Aquí sí se captura y se convierte en
   una `DecisionFinal` real, con degradación con gracia: si la recarga
   falla, se seguye operando con el último artefacto válido conocido en
   vez de caerse.
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
        artefacto = leer_vigente(self.ruta_artefacto)  # puede lanzar -- primera carga sin fallback
        self._ejecutor = Ejecutor(artefacto=artefacto)

    def recargar_artefacto(self) -> bool:
        """Intenta recargar el artefacto vigente desde el Puente,
        conservando el estado recursivo acumulado (no se pierde historial
        por recalibrar). Si falla (archivo ausente, corrupto, o inválido),
        **no reemplaza** el Ejecutor actual — sigue operando con el último
        artefacto válido conocido. Devuelve si la recarga tuvo éxito."""
        try:
            artefacto_nuevo = leer_vigente(self.ruta_artefacto)
        except (FileNotFoundError, ArtefactoInvalido, ValueError):
            return False
        self._ejecutor = Ejecutor(artefacto=artefacto_nuevo, estado=self._ejecutor.estado)
        return True

    def decidir(self, transaccion: dict) -> DecisionFinal:
        """Único método para tomar una decisión real. Cualquier fallo del
        Ejecutor (score inválido, excepción inesperada) se captura aquí y
        se convierte en escalar a revisión manual — nunca se propaga sin
        capturar hacia el llamador."""
        try:
            resultado = self._ejecutor.decidir(transaccion)
        except Exception as e:
            return DecisionFinal(es_sospechosa=True, razon=f"error en el ejecutor: {e} — escalar a revisión manual", score=None)
        return evaluar_veto(resultado, transaccion, monto_maximo=self.monto_maximo)
