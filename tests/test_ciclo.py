from src.ciclo import CicloDecision
from src.puente import publicar

ARTEFACTO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "regresion_logistica",
    "features": ["Amount"],
    "coeficientes": [0.1],
    "intercepto": -1.0,
    "umbral_decision": 0.5,
    "metricas_validacion": {},
}


def test_ciclo_decision_solo_se_construye_desde_una_ruta_valida(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)  # the only way to build it -- forces the Bridge
    decision = ciclo.decidir({"Amount": 20.0, "Time": 0.0})
    assert decision.razon == "modelo"  # reached the Veto normally


def test_artefacto_ausente_al_construir_lanza_error_sin_fallback_silencioso(tmp_path):
    import pytest
    from src.puente import leer_vigente  # re-exports FileNotFoundError from puente

    with pytest.raises(FileNotFoundError):
        CicloDecision(ruta_artefacto=tmp_path / "no_existe.json")


def test_veto_siempre_se_aplica_score_nan_no_pasa_como_no_sospechosa(tmp_path, monkeypatch):
    # Critical finding #2: without CicloDecision, nan >= threshold is False in Python -- an
    # invalid score would look like "not suspicious". With CicloDecision, the Veto ALWAYS runs.
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    def decidir_con_score_invalido(self, transaccion):
        return {"score": float("nan"), "es_sospechosa": False}

    monkeypatch.setattr("src.ejecutor.Ejecutor.decidir", decidir_con_score_invalido)

    decision = ciclo.decidir({"Amount": 20.0, "Time": 0.0})

    assert decision.es_sospechosa is True  # the Veto catches it, it doesn't stay "not suspicious"
    assert "rango" in decision.razon


def test_excepcion_del_ejecutor_escala_a_revision_manual_en_vez_de_propagarse(tmp_path, monkeypatch):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    def decidir_que_revienta(self, transaccion):
        raise KeyError("V1")  # e.g. a missing feature in the real transaction

    monkeypatch.setattr("src.ejecutor.Ejecutor.decidir", decidir_que_revienta)

    decision = ciclo.decidir({"Amount": 20.0, "Time": 0.0})  # must not propagate the KeyError

    assert decision.es_sospechosa is True
    assert "escalar a revisión manual" in decision.razon


def test_recargar_artefacto_exitoso_actualiza_el_umbral(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    nuevo = {**ARTEFACTO, "umbral_decision": 0.99}
    publicar(nuevo, ruta)

    assert ciclo.recargar_artefacto() is True
    assert ciclo._ejecutor.artefacto["umbral_decision"] == 0.99


def test_recargar_artefacto_conserva_el_estado_recursivo(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    ciclo.decidir({"Amount": 100.0, "Time": 0.0})  # feeds the state (EWMA, window)
    estado_antes = ciclo._ejecutor.estado

    publicar({**ARTEFACTO, "umbral_decision": 0.8}, ruta)
    ciclo.recargar_artefacto()

    assert ciclo._ejecutor.estado is estado_antes  # the same state object, not a new empty one


def test_recargar_artefacto_fallido_no_reemplaza_el_ejecutor_vigente(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    ruta.write_text("esto no es json valido {{{", encoding="utf-8")  # corrupts the file

    exito = ciclo.recargar_artefacto()

    assert exito is False
    # keeps operating on the original artifact -- graceful degradation, not a crash
    decision = ciclo.decidir({"Amount": 20.0, "Time": 1.0})
    assert decision.razon == "modelo"
