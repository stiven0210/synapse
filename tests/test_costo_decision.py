import numpy as np
import pytest

from src.calibrador import _mejor_umbral_por_f1
from src.costo_decision import costo_esperado, mejor_umbral_por_costo


def test_costo_esperado_caso_numerico_a_mano():
    y_true = np.array([0, 0, 1, 1])
    montos = np.array([10.0, 10.0, 500.0, 800.0])
    scores = np.array([0.2, 0.6, 0.3, 0.9])

    # threshold=0.5 -> pred=[0,1,0,1]: idx1 is FP (true=0), idx2 is FN (true=1, amount=500)
    costo = costo_esperado(y_true, montos, scores, umbral=0.5, costo_falso_positivo=50.0)

    assert costo == pytest.approx(1 * 50.0 + 500.0)  # 1 FP * fp_cost + FN's amount


def test_costo_esperado_sin_errores_es_cero():
    y_true = np.array([0, 1])
    montos = np.array([10.0, 500.0])
    scores = np.array([0.1, 0.9])

    assert costo_esperado(y_true, montos, scores, umbral=0.5, costo_falso_positivo=999.0) == 0.0


def test_mejor_umbral_por_costo_prefiere_capturar_un_fraude_caro_sobre_evitar_un_fp_barato():
    # Large-amount fraud (1000) with an intermediate score (0.5), between two
    # legitimate small-amount transactions (5) with scores 0.4 and 0.6.
    # With costo_falso_positivo=20 (cheap next to 1000), the real optimum
    # must lower the threshold enough to catch the fraud, accepting at
    # most 1 false positive -- never let the 1000 fraud slip through to
    # avoid a 20-cost FP.
    y_true = np.array([0, 0, 1])
    montos = np.array([5.0, 5.0, 1000.0])
    scores = np.array([0.4, 0.6, 0.5])

    resultado = mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=20.0)

    # threshold=0.5 -> 1 FP (idx1) + fraud captured -> cost = 20
    # threshold=0.6 -> 1 FP (idx1) + fraud NOT captured -> cost = 20 + 1000 = 1020
    # threshold=0.4 -> 2 FP -> cost = 40
    assert resultado["umbral_optimo"] == pytest.approx(0.5)
    assert resultado["costo_esperado_minimo"] == pytest.approx(20.0)
    assert resultado["costo_falso_positivo_asumido"] == 20.0
    assert len(resultado["curva_costo_por_umbral"]) == 3  # one candidate per unique score


def test_mejor_umbral_por_costo_rechaza_costo_no_positivo():
    y_true = np.array([0, 1])
    montos = np.array([10.0, 500.0])
    scores = np.array([0.1, 0.9])

    with pytest.raises(ValueError, match="debe ser > 0"):
        mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=0.0)

    with pytest.raises(ValueError, match="debe ser > 0"):
        mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=-5.0)


def test_umbral_por_costo_diverge_del_umbral_por_f1_cuando_hay_un_fraude_caro_y_uno_barato():
    # F1 counts every false negative the same (a fraud is a fraud) -- between
    # capturing both frauds with 2 FP (F1=0.667) or only the expensive one with 1 FP
    # (F1=0.5), F1 prefers capturing both. Expected cost doesn't think in
    # counts: with a cheap FP (50) against a $1000 fraud, it's worth
    # accepting one fewer FP and letting the cheap fraud ($10) through than
    # paying the extra FP -- verified by hand:
    #   threshold=0.3 (F1-optimal): 2 FP, 0 FN -> cost = 2*50 = 100
    #   threshold=0.7 (cost-optimal): 1 FP, 1 FN ($10) -> cost = 50 + 10 = 60
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.9, 0.3, 0.7])
    montos = np.array([5.0, 5.0, 10.0, 1000.0])  # cheap fraud at score 0.3, expensive fraud at score 0.7

    umbral_f1 = _mejor_umbral_por_f1(y_true, scores)
    resultado_costo = mejor_umbral_por_costo(y_true, montos, scores, costo_falso_positivo=50.0)

    assert umbral_f1 == pytest.approx(0.3)
    assert resultado_costo["umbral_optimo"] == pytest.approx(0.7)
    assert resultado_costo["costo_esperado_minimo"] == pytest.approx(60.0)
    assert umbral_f1 != pytest.approx(resultado_costo["umbral_optimo"])
