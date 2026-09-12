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


def test_sin_ruta_estado_se_comporta_solo_en_memoria(tmp_path):
    # Regresión: el comportamiento sin persistencia no debe cambiar.
    limitador_1 = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=2)
    limitador_1("a")
    limitador_1("b")

    limitador_2 = LimitadorLlamadasDiarias(cliente_llm=lambda p: "ok", max_llamadas_por_dia=2)
    assert limitador_2.llamadas_restantes_hoy == 2  # instancia nueva, sin ruta_estado -- presupuesto fresco


def test_persistencia_sobrevive_a_una_instancia_nueva_el_mismo_dia(tmp_path):
    # Hallazgo real de auditoría: los 2 puntos de entrada reales
    # (runner_diario.py, triage_veto.py) construyen un limitador nuevo en
    # cada invocación -- sin persistir a disco, un reintento el mismo día
    # obtenía presupuesto fresco, incumpliendo "no exceder el gasto".
    ruta_estado = tmp_path / "estado_limitador.json"
    dia_fijo = date(2026, 1, 1)

    limitador_1 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=3, reloj=lambda: dia_fijo, ruta_estado=ruta_estado,
    )
    limitador_1("a")
    limitador_1("b")  # 2 de 3 usadas, persistidas a disco

    # Nueva instancia -- simula un proceso nuevo (otra corrida del script) el mismo día.
    limitador_2 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=3, reloj=lambda: dia_fijo, ruta_estado=ruta_estado,
    )
    assert limitador_2.llamadas_restantes_hoy == 1  # hereda las 2 ya usadas, no un presupuesto fresco
    limitador_2("c")
    with pytest.raises(PresupuestoAgotado):
        limitador_2("d")


def test_persistencia_resetea_en_un_dia_calendario_nuevo(tmp_path):
    ruta_estado = tmp_path / "estado_limitador.json"

    limitador_1 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=1, reloj=lambda: date(2026, 1, 1), ruta_estado=ruta_estado,
    )
    limitador_1("a")  # agota el cupo del 2026-01-01, persistido

    limitador_2 = LimitadorLlamadasDiarias(
        cliente_llm=lambda p: "ok", max_llamadas_por_dia=1, reloj=lambda: date(2026, 1, 2), ruta_estado=ruta_estado,
    )
    assert limitador_2.llamadas_restantes_hoy == 1  # día calendario distinto -- cupo fresco, no hereda el agotado
