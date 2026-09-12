from datetime import date

import pytest

from src.limitador_llamadas import LimitadorLlamadasDiarias, PresupuestoAgotado


def test_permite_llamadas_dentro_del_limite():
    limitador = LimitadorLlamadasDiarias(cliente_llm=lambda p: f"respuesta a {p}", max_llamadas_por_dia=3)
    assert limitador("a") == "respuesta a a"
    assert limitador("b") == "respuesta a b"
    assert limitador("c") == "respuesta a c"


def test_lanza_presupuesto_agotado_al_exceder_limite():
    limitador = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=2)
    limitador("a")
    limitador("b")
    with pytest.raises(PresupuestoAgotado):
        limitador("c")


def test_contador_se_resetea_en_un_dia_nuevo():
    dias = iter([date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 2)])
    limitador = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=1, reloj=lambda: next(dias))
    assert limitador("a") == "ok"  # 2026-01-01, consume el único cupo del día
    with pytest.raises(PresupuestoAgotado):
        limitador("b")  # todavía 2026-01-01, cupo agotado
    assert limitador("c") == "ok"  # 2026-01-02: día nuevo, cupo reseteado


def test_llamadas_restantes_hoy():
    limitador = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=5)
    assert limitador.llamadas_restantes_hoy == 5
    limitador("a")
    limitador("b")
    assert limitador.llamadas_restantes_hoy == 3


def test_no_cuenta_llamadas_que_lanzan_presupuesto_agotado():
    limitador = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=1)
    limitador("a")
    with pytest.raises(PresupuestoAgotado):
        limitador("b")
    assert limitador.llamadas_restantes_hoy == 0
