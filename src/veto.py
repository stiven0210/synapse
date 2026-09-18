"""Veto — Phase 4. Hard, model-independent invariants
(`docs/adr/0002-veto-layer.md`). Evaluated AFTER the Executor and can
override its decision — never the other way around. Protects the system
even if the Calibrator or the Executor are miscalibrated.
"""
import math
from dataclasses import dataclass

MONTO_MAXIMO_ABSOLUTO = 10_000.0  # explicitly configured (ADR_002) — not a silent default


@dataclass(frozen=True)
class DecisionFinal:
    es_sospechosa: bool
    razon: str
    score: float | None = None


def _score_valido(score) -> bool:
    return isinstance(score, (int, float)) and not isinstance(score, bool) and not math.isnan(score) and 0.0 <= score <= 1.0


def evaluar(resultado_ejecutor: dict, transaccion: dict, monto_maximo: float = MONTO_MAXIMO_ABSOLUTO) -> DecisionFinal:
    score = resultado_ejecutor.get("score")

    if not _score_valido(score):
        # ADR_002, invariant 3: an invalid score is never treated as "0" or "1" —
        # it's treated as "I don't know", which in a fraud system means escalate.
        return DecisionFinal(es_sospechosa=True, razon="score fuera de rango numérico válido — escalar a revisión manual", score=score)

    if abs(transaccion["Amount"]) > monto_maximo:
        # ADR_002, invariant 2: an extreme ABSOLUTE amount vetoes regardless of the model's score.
        # Bug fixed (audit): this used to compare Amount > monto_maximo without abs() -- a
        # very negative amount (e.g. a fraudulent refund/chargeback) never triggered the
        # veto despite ADR_002 explicitly calling it "absolute amount".
        return DecisionFinal(
            es_sospechosa=True,
            razon=f"|monto| {abs(transaccion['Amount'])} excede el límite absoluto {monto_maximo}",
            score=score,
        )

    return DecisionFinal(es_sospechosa=resultado_ejecutor["es_sospechosa"], razon="modelo", score=score)
