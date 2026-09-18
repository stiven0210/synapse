import pytest

from src.ciclo_arboles import CicloDecisionArboles
from src.puente_arboles import publicar

ARTEFACTO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "gradient_boosting",
    "features": ["amt", "hora", "conteo_ventana_global", "monto_ewma_cuenta", "huella_categoria_cuenta"],
    "umbrales_por_feature": {
        "amt": [100.0],
        "hora": [],
        "conteo_ventana_global": [],
        "monto_ewma_cuenta": [],
        "huella_categoria_cuenta": [],
    },
    "tabla_busqueda_forma": [2, 1, 1, 1, 1],
    "tabla_busqueda_plana": [0.1, 0.9],
    "calibracion_isotonica": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "umbral_decision": 0.5,
    "metricas_validacion": {},
    "frecuencia_poblacional_categoria": {"x": 1.0},
}

TRANSACCION = {"cc_num": "A", "amt": 20.0, "unix_time": 1704122400, "category": "x"}


def test_ciclo_decision_solo_se_construye_desde_una_ruta_valida(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecisionArboles(ruta_artefacto=ruta)
    decision = ciclo.decidir(TRANSACCION)
    assert decision.razon == "modelo"


def test_artefacto_ausente_al_construir_lanza_error_sin_fallback_silencioso(tmp_path):
    with pytest.raises(FileNotFoundError):
        CicloDecisionArboles(ruta_artefacto=tmp_path / "no_existe.json")


def test_veto_siempre_se_aplica_score_nan_no_pasa_como_no_sospechosa(tmp_path, monkeypatch):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecisionArboles(ruta_artefacto=ruta)

    def decidir_con_score_invalido(self, transaccion):
        return {"score": float("nan"), "es_sospechosa": False}

    monkeypatch.setattr("src.ejecutor_arboles.EjecutorArboles.decidir", decidir_con_score_invalido)

    decision = ciclo.decidir(TRANSACCION)

    assert decision.es_sospechosa is True
    assert "rango" in decision.razon


def test_excepcion_del_ejecutor_escala_a_revision_manual_en_vez_de_propagarse(tmp_path, monkeypatch):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecisionArboles(ruta_artefacto=ruta)

    def decidir_que_revienta(self, transaccion):
        raise KeyError("amt")

    monkeypatch.setattr("src.ejecutor_arboles.EjecutorArboles.decidir", decidir_que_revienta)

    decision = ciclo.decidir(TRANSACCION)

    assert decision.es_sospechosa is True
    assert "escalar a revisión manual" in decision.razon


def test_monto_maximo_absoluto_veta_incluso_con_score_bajo(tmp_path):
    # amt > monto_maximo -- the table's score (bin amt<=100 -> 0.1, not suspicious) doesn't matter.
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecisionArboles(ruta_artefacto=ruta, monto_maximo=15.0)

    decision = ciclo.decidir(TRANSACCION)  # amt=20.0 > monto_maximo=15.0 (max amount)

    assert decision.es_sospechosa is True
    assert "excede el límite absoluto" in decision.razon


def test_recargar_artefacto_exitoso_actualiza_el_umbral(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecisionArboles(ruta_artefacto=ruta)

    nuevo = {**ARTEFACTO, "umbral_decision": 0.99}
    publicar(nuevo, ruta)

    assert ciclo.recargar_artefacto() is True
    assert ciclo._ejecutor.artefacto["umbral_decision"] == 0.99


def test_recargar_artefacto_conserva_el_estado_recursivo(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecisionArboles(ruta_artefacto=ruta)

    ciclo.decidir(TRANSACCION)  # feeds both states (global and per-account)
    estado_global_antes = ciclo._ejecutor.estado_global
    estado_cuenta_antes = ciclo._ejecutor.estado_cuenta

    publicar({**ARTEFACTO, "umbral_decision": 0.8}, ruta)
    ciclo.recargar_artefacto()

    assert ciclo._ejecutor.estado_global is estado_global_antes
    assert ciclo._ejecutor.estado_cuenta is estado_cuenta_antes


def test_recargar_artefacto_fallido_no_reemplaza_el_ejecutor_vigente(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecisionArboles(ruta_artefacto=ruta)

    ruta.write_text("esto no es json valido {{{", encoding="utf-8")

    exito = ciclo.recargar_artefacto()

    assert exito is False
    decision = ciclo.decidir({**TRANSACCION, "unix_time": TRANSACCION["unix_time"] + 1})
    assert decision.razon == "modelo"
