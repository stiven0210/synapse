import numpy as np
import pytest

from src.calibrador import _mejor_umbral_por_f1
from src.costo_decision import costo_esperado, mejor_umbral_por_costo


def test_costo_esperado_caso_numerico_a_mano():
    y_true = np.array([0, 0, 1, 1])
    montos = np.array([10.0, 10.0, 500.0, 800.0])
    scores = np.array([0.2, 0.6, 0.3, 0.9])

    # umbral=0.5 -> pred=[0,1,0,1]: idx1 es FP (true=0), idx2 es FN (true=1, monto=500)
    costo = costo_esperado(y_true, montos, scores, umbral=0.5, costo_falso_positivo=50.0)

    assert costo == pytest.approx(1 * 50.0 + 500.0)  # 1 FP * costo_fp + monto del FN


def test_costo_esperado_sin_errores_es_cero():
    y_true = np.array([0, 1])
    montos = np.array([10.0, 500.0])
    scores = np.array([0.1, 0.9])

    assert costo_esperado(y_true, montos, scores, umbral=0.5, costo_falso_positivo=999.0) == 0.0


def test_mejor_umbral_por_costo_prefiere_capturar_un_fraude_caro_sobre_evitar_un_fp_barato():
    # Fraude de monto grande (1000) con score intermedio (0.5), entre dos
    # transacciones legítimas de monto chico (5) con scores 0.4 y 0.6.
    # Con costo_falso_positivo=20 (barato frente a 1000), el óptimo real
    # debe bajar el umbral lo suficiente para atrapar el fraude, aceptando
    # como mucho 1 falso positivo -- nunca dejar pasar el fraude de 1000
    # por evitar un FP de 20.
    y_true = np.array([0, 0, 1])
    montos = np.array([5.0, 5.0, 1000.0])
    scores = np.array([0.4, 0.6, 0.5])

    resultado = mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=20.0)

    # umbral=0.5 -> 1 FP (idx1) + fraude capturado -> costo = 20
    # umbral=0.6 -> 1 FP (idx1) + fraude NO capturado -> costo = 20 + 1000 = 1020
    # umbral=0.4 -> 2 FP -> costo = 40
    assert resultado["umbral_optimo"] == pytest.approx(0.5)
    assert resultado["costo_esperado_minimo"] == pytest.approx(20.0)
    assert resultado["costo_falso_positivo_asumido"] == 20.0
    assert len(resultado["curva_costo_por_umbral"]) == 3  # un candidato por score único


def test_mejor_umbral_por_costo_rechaza_costo_no_positivo():
    y_true = np.array([0, 1])
    montos = np.array([10.0, 500.0])
    scores = np.array([0.1, 0.9])

    with pytest.raises(ValueError, match="debe ser > 0"):
        mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=0.0)

    with pytest.raises(ValueError, match="debe ser > 0"):
        mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=-5.0)


def test_umbral_por_costo_diverge_del_umbral_por_f1_cuando_hay_un_fraude_caro_y_uno_barato():
    # F1 cuenta cada falso negativo igual (un fraude es un fraude) -- entre
    # capturar los dos fraudes con 2 FP (F1=0.667) o solo el caro con 1 FP
    # (F1=0.5), F1 prefiere capturar los dos. El costo esperado no piensa
    # en conteos: con un FP barato (50) frente a un fraude de $1000, más
    # vale aceptar 1 FP menos y dejar pasar el fraude barato ($10) que
    # pagar el FP extra -- verificado a mano:
    #   umbral=0.3 (F1-óptimo): 2 FP, 0 FN -> costo = 2*50 = 100
    #   umbral=0.7 (costo-óptimo): 1 FP, 1 FN ($10) -> costo = 50 + 10 = 60
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.9, 0.3, 0.7])
    montos = np.array([5.0, 5.0, 10.0, 1000.0])  # fraude barato en score 0.3, fraude caro en score 0.7

    umbral_f1 = _mejor_umbral_por_f1(y_true, scores)
    resultado_costo = mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=50.0)

    assert umbral_f1 == pytest.approx(0.3)
    assert resultado_costo["umbral_optimo"] == pytest.approx(0.7)
    assert resultado_costo["costo_esperado_minimo"] == pytest.approx(60.0)
    assert umbral_f1 != pytest.approx(resultado_costo["umbral_optimo"])
