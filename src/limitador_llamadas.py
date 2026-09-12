"""Rate limiting y control de costos para llamadas a un LLM. Mismo patrón
que `an earlier project/src/agents/rate_limiter.py`, replicado como código nuevo e
independiente (SYNAPSE no comparte código con otros proyectos, `CLAUDE.md`).

Envuelve cualquier cliente LLM (`Callable[[str], str]`) y cuenta llamadas
por día calendario. Al agotar el presupuesto, **lanza una excepción** en vez
de seguir llamando — el objetivo es no exceder el gasto, no llevar un
contador informativo que alguien revisa después.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Callable


class PresupuestoAgotado(Exception):
    pass


@dataclass
class LimitadorLlamadasDiarias:
    cliente_llm: Callable[[str], str]
    max_llamadas_por_dia: int = 50
    reloj: Callable[[], date] = field(default=date.today)
    _fecha_actual: date | None = field(default=None, init=False, repr=False)
    _contador: int = field(default=0, init=False, repr=False)

    def __call__(self, prompt: str) -> str:
        hoy = self.reloj()
        if hoy != self._fecha_actual:
            self._fecha_actual = hoy
            self._contador = 0
        if self._contador >= self.max_llamadas_por_dia:
            raise PresupuestoAgotado(f"límite de {self.max_llamadas_por_dia} llamadas/día alcanzado ({hoy})")
        self._contador += 1
        return self.cliente_llm(prompt)

    @property
    def llamadas_restantes_hoy(self) -> int:
        hoy = self.reloj()
        if hoy != self._fecha_actual:
            return self.max_llamadas_por_dia
        return max(0, self.max_llamadas_por_dia - self._contador)
