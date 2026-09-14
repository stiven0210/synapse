"""CicloDecisionArboles — punto de entrada sancionado para Dominio 2, paralelo
a `CicloDecision` (Dominio 1, no modificado). Mismas 3 garantías que su
contraparte (ver `ciclo.py`): el artefacto siempre pasa por el Puente, el
resultado del Ejecutor siempre pasa por `veto.py::evaluar` (reusado sin
modificar -- es independiente del modelo por diseño), y cualquier excepción
del Ejecutor se convierte en escalar a revisión manual en vez de propagarse.

**Por qué existe un Ciclo paralelo y no uno genérico**: el propio
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` (sección 6) señala esto como
la opción correcta -- generalizar la interfaz recién after un segundo
dominio real, no antes; forzar hoy un contrato común entre un Ejecutor de
producto punto y uno de tabla de búsqueda sería abstraer sin necesidad
comprobada todavía.
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
        artefacto = leer_vigente(self.ruta_artefacto)  # puede lanzar -- primera carga sin fallback
        self._ejecutor = EjecutorArboles(artefacto=artefacto)

    def recargar_artefacto(self) -> bool:
        """Igual que `CicloDecision.recargar_artefacto()`: conserva el
        estado recursivo acumulado (global y por cuenta) y no reemplaza el
        Ejecutor vigente si el artefacto nuevo está ausente/corrupto/inválido."""
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
        """Único método para tomar una decisión real en Dominio 2. `veto.py`
        espera la clave `Amount` (contrato de Dominio 1) -- se agrega aquí un
        alias de `amt` sin modificar `veto.py`, que es intencionalmente
        independiente del esquema de columnas de cualquier dominio."""
        try:
            resultado = self._ejecutor.decidir(transaccion)
        except Exception as e:
            return DecisionFinal(es_sospechosa=True, razon=f"error en el ejecutor: {e} — escalar a revisión manual", score=None)
        transaccion_para_veto = {**transaccion, "Amount": transaccion["amt"]}
        return evaluar_veto(resultado, transaccion_para_veto, monto_maximo=self.monto_maximo)
