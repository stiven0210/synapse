"""Veto — Fase 4. Invariantes duros e independientes del modelo
(`docs/adr/0002-veto-layer.md`). Se evalúan DESPUÉS del Ejecutor y pueden
sobreescribir su decisión — nunca al revés. Protege incluso si el
Calibrador o el Ejecutor están mal calibrados.
"""
import math
from dataclasses import dataclass

MONTO_MAXIMO_ABSOLUTO = 10_000.0  # configurado explícitamente (ADR_002) — no un default silencioso


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
        # ADR_002, invariante 3: un score inválido nunca se trata como "0" ni como "1" —
        # se trata como "no sé", que en un sistema de fraude significa escalar.
        return DecisionFinal(es_sospechosa=True, razon="score fuera de rango numérico válido — escalar a revisión manual", score=score)

    if abs(transaccion["Amount"]) > monto_maximo:
        # ADR_002, invariante 2: monto ABSOLUTO extremo veta sin importar el score del modelo.
        # Bug corregido (auditoría): antes comparaba Amount > monto_maximo sin abs() -- un
        # monto muy negativo (ej. un reembolso/chargeback fraudulento) nunca disparaba el
        # veto pese a que ADR_002 lo llama explícitamente "monto absoluto".
        return DecisionFinal(
            es_sospechosa=True,
            razon=f"|monto| {abs(transaccion['Amount'])} excede el límite absoluto {monto_maximo}",
            score=score,
        )

    return DecisionFinal(es_sospechosa=resultado_ejecutor["es_sospechosa"], razon="modelo", score=score)
