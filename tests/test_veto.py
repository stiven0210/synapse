import math

import pytest

from src.veto import DecisionFinal, evaluar


def test_deja_pasar_decision_del_modelo_cuando_todo_es_normal():
    resultado = evaluar({"score": 0.9, "es_sospechosa": True}, {"Amount": 50.0})
    assert resultado == DecisionFinal(es_sospechosa=True, razon="modelo", score=0.9)


def test_deja_pasar_no_sospechosa_cuando_todo_es_normal():
    resultado = evaluar({"score": 0.1, "es_sospechosa": False}, {"Amount": 50.0})
    assert resultado.es_sospechosa is False
    assert resultado.razon == "modelo"


def test_veta_por_score_nan():
    resultado = evaluar({"score": float("nan"), "es_sospechosa": False}, {"Amount": 50.0})
    assert resultado.es_sospechosa is True
    assert "rango" in resultado.razon


def test_veta_por_score_fuera_de_0_1():
    resultado = evaluar({"score": 1.5, "es_sospechosa": False}, {"Amount": 50.0})
    assert resultado.es_sospechosa is True
    assert "rango" in resultado.razon


def test_veta_por_score_none():
    resultado = evaluar({"score": None, "es_sospechosa": False}, {"Amount": 50.0})
    assert resultado.es_sospechosa is True


def test_veta_por_monto_absoluto_incluso_si_el_modelo_dice_que_no_es_sospechosa():
    # El caso que de verdad importa: el veto SOBREESCRIBE al modelo, no al revés.
    resultado = evaluar({"score": 0.01, "es_sospechosa": False}, {"Amount": 50_000.0})
    assert resultado.es_sospechosa is True
    assert "excede el límite absoluto" in resultado.razon


def test_monto_justo_en_el_limite_no_veta():
    resultado = evaluar({"score": 0.1, "es_sospechosa": False}, {"Amount": 10_000.0}, monto_maximo=10_000.0)
    assert resultado.es_sospechosa is False  # estrictamente mayor que el límite, no >=


def test_veta_por_monto_negativo_extremo():
    # Bug corregido: antes solo Amount > monto_maximo, sin abs() -- un reembolso/chargeback
    # fraudulento con monto muy negativo nunca disparaba el veto.
    resultado = evaluar({"score": 0.01, "es_sospechosa": False}, {"Amount": -50_000.0})
    assert resultado.es_sospechosa is True
    assert "excede el límite absoluto" in resultado.razon
