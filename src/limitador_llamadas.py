"""Rate limiting and cost control for LLM calls. Same pattern used in an
earlier project of ours, replicated here as new, independent code
(SYNAPSE shares no code with other projects, `CLAUDE.md`).

Wraps any LLM client (`Callable[[str], str]`) and counts calls per
calendar day. Once the budget is exhausted, it **raises an exception**
instead of continuing to call — the goal is to never exceed the spend, not
to keep an informational counter someone checks later.
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
        """Real audit finding: without `ruta_estado`, the counter lives only
        in the process's memory -- but the two real entry points
        (`scripts/runner_diario.py`, `scripts/triage_veto.py`) are
        single-use processes (by design, for Task Scheduler/cron: "one
        invocation = one day"). Each run used to create a fresh
        `LimitadorLlamadasDiarias` with a fresh budget, so the "N
        calls/day" limit never survived across runs -- a manual retry on
        the same day got a full budget again, exactly what the stated goal
        ("never exceed the spend") prohibits. With `ruta_estado`, the
        counter persists across processes that run on the same calendar
        day."""
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
