"""Rate limiting y control de costos para llamadas a un LLM. Mismo patrón
que `an earlier project/src/agents/rate_limiter.py`, replicado como código nuevo e
independiente (SYNAPSE no comparte código con otros proyectos, `CLAUDE.md`).

Envuelve cualquier cliente LLM (`Callable[[str], str]`) y cuenta llamadas
por día calendario. Al agotar el presupuesto, **lanza una excepción** en vez
de seguir llamando — el objetivo es no exceder el gasto, no llevar un
contador informativo que alguien revisa después.
"""
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable


class PresupuestoAgotado(Exception):
    pass


@dataclass
class LimitadorLlamadasDiarias:
    cliente_llm: Callable[[str], str]
    max_llamadas_por_dia: int = 50
    reloj: Callable[[], date] = field(default=date.today)
    ruta_estado: Path | None = None
    _fecha_actual: date | None = field(default=None, init=False, repr=False)
    _contador: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Hallazgo real de auditoría: sin `ruta_estado`, el contador vive
        solo en memoria del proceso -- pero los dos puntos de entrada reales
        (`scripts/runner_diario.py`, `scripts/triage_veto.py`) son procesos
        de un solo uso (por diseño, para Task Scheduler/cron: "una
        invocación = un día"). Cada corrida creaba un `LimitadorLlamadasDiarias`
        nuevo con presupuesto fresco, así que el límite de "N llamadas/día"
        nunca sobrevivía entre corridas -- un reintento manual el mismo día
        obtenía presupuesto completo de nuevo, exactamente lo que el
        objetivo declarado ("no exceder el gasto") prohíbe. Con `ruta_estado`,
        el contador persiste entre procesos que corren el mismo día
        calendario."""
        if self.ruta_estado is not None and self.ruta_estado.exists():
            data = json.loads(self.ruta_estado.read_text(encoding="utf-8"))
            fecha_guardada = date.fromisoformat(data["fecha"])
            if fecha_guardada == self.reloj():
                self._fecha_actual = fecha_guardada
                self._contador = data["contador"]

    def _guardar_estado(self) -> None:
        if self.ruta_estado is None:
            return
        data = {"fecha": self._fecha_actual.isoformat(), "contador": self._contador}
        self.ruta_estado.parent.mkdir(parents=True, exist_ok=True)
        ruta_temporal = self.ruta_estado.with_name(self.ruta_estado.name + ".tmp")
        ruta_temporal.write_text(json.dumps(data), encoding="utf-8")
        ruta_temporal.replace(self.ruta_estado)

    def __call__(self, prompt: str) -> str:
        hoy = self.reloj()
        if hoy != self._fecha_actual:
            self._fecha_actual = hoy
            self._contador = 0
        if self._contador >= self.max_llamadas_por_dia:
            raise PresupuestoAgotado(f"límite de {self.max_llamadas_por_dia} llamadas/día alcanzado ({hoy})")
        self._contador += 1
        self._guardar_estado()
        return self.cliente_llm(prompt)

    @property
    def llamadas_restantes_hoy(self) -> int:
        hoy = self.reloj()
        if hoy != self._fecha_actual:
            return self.max_llamadas_por_dia
        return max(0, self.max_llamadas_por_dia - self._contador)
