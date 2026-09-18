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
    assert limitador("a") == "ok"  # 2026-01-01, consumes the day's only slot
    with pytest.raises(PresupuestoAgotado):
        limitador("b")  # still 2026-01-01, budget exhausted
    assert limitador("c") == "ok"  # 2026-01-02: new day, budget reset


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


def test_sin_ruta_estado_se_comporta_solo_en_memoria(tmp_path):
    # Regression: behavior without persistence must not change.
    limitador_1 = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=2)
    limitador_1("a")
    limitador_1("b")

    limitador_2 = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=2)
    assert limitador_2.llamadas_restantes_hoy == 2  # new instance, no ruta_estado -- fresh budget


def test_persistencia_sobrevive_a_una_instancia_nueva_el_mismo_dia(tmp_path):
    # Real audit finding: the 2 real entry points
    # (runner_diario.py, triage_veto.py) build a new limiter on
    # every invocation -- without persisting to disk, a same-day retry
    # got a fresh budget, breaking "never exceed the spend".
    ruta_estado = tmp_path / "estado_limitador.json"
    dia_fijo = date(2026, 1, 1)

    limitador_1 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=3, reloj=lambda: dia_fijo, ruta_estado=ruta_estado,
    )
    limitador_1("a")
    limitador_1("b")  # 2 of 3 used, persisted to disk

    # New instance -- simulates a new process (another run of the script) on the same day.
    limitador_2 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=3, reloj=lambda: dia_fijo, ruta_estado=ruta_estado,
    )
    assert limitador_2.llamadas_restantes_hoy == 1  # inherits the 2 already used, not a fresh budget
    limitador_2("c")
    with pytest.raises(PresupuestoAgotado):
        limitador_2("d")


def test_persistencia_resetea_en_un_dia_calendario_nuevo(tmp_path):
    ruta_estado = tmp_path / "estado_limitador.json"

    limitador_1 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=1, reloj=lambda: date(2026, 1, 1), ruta_estado=ruta_estado,
    )
    limitador_1("a")  # exhausts 2026-01-01's slot, persisted

    limitador_2 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=1, reloj=lambda: date(2026, 1, 2), ruta_estado=ruta_estado,
    )
    assert limitador_2.llamadas_restantes_hoy == 1  # different calendar day -- fresh slot, doesn't inherit the exhausted one
