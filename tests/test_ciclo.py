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
    ciclo = CicloDecision(ruta_artefacto=ruta)  # única forma de construirlo -- fuerza el Puente
    decision = ciclo.decidir({"Amount": 20.0, "Time": 0.0})
    assert decision.razon == "modelo"  # llegó normal hasta el Veto


def test_artefacto_ausente_al_construir_lanza_error_sin_fallback_silencioso(tmp_path):
    import pytest
    from src.puente import leer_vigente  # reexporta FileNotFoundError de puente

    with pytest.raises(FileNotFoundError):
        CicloDecision(ruta_artefacto=tmp_path / "no_existe.json")


def test_veto_siempre_se_aplica_score_nan_no_pasa_como_no_sospechosa(tmp_path, monkeypatch):
    # El hallazgo crítico #2: sin CicloDecision, nan >= umbral es False en Python -- un score
    # inválido se vería como "no sospechosa". Con CicloDecision, el Veto SIEMPRE corre.
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    def decidir_con_score_invalido(self, transaccion):
        return {"score": float("nan"), "es_sospechosa": False}

    monkeypatch.setattr("src.ejecutor.Ejecutor.decidir", decidir_con_score_invalido)

    decision = ciclo.decidir({"Amount": 20.0, "Time": 0.0})

    assert decision.es_sospechosa is True  # el Veto lo atrapa, no queda como "no sospechosa"
    assert "rango" in decision.razon


def test_excepcion_del_ejecutor_escala_a_revision_manual_en_vez_de_propagarse(tmp_path, monkeypatch):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    def decidir_que_revienta(self, transaccion):
        raise KeyError("V1")  # ej. un feature faltante en la transacción real

    monkeypatch.setattr("src.ejecutor.Ejecutor.decidir", decidir_que_revienta)

    decision = ciclo.decidir({"Amount": 20.0, "Time": 0.0})  # no debe propagar el KeyError

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

    ciclo.decidir({"Amount": 100.0, "Time": 0.0})  # alimenta el estado (EWMA, ventana)
    estado_antes = ciclo._ejecutor.estado

    publicar({**ARTEFACTO, "umbral_decision": 0.8}, ruta)
    ciclo.recargar_artefacto()

    assert ciclo._ejecutor.estado is estado_antes  # el mismo objeto de estado, no uno nuevo vacío


def test_recargar_artefacto_fallido_no_reemplaza_el_ejecutor_vigente(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    ciclo = CicloDecision(ruta_artefacto=ruta)

    ruta.write_text("esto no es json valido {{{", encoding="utf-8")  # corrompe el archivo

    exito = ciclo.recargar_artefacto()

    assert exito is False
    # sigue operando con el artefacto original -- degradación con gracia, no una caída
    decision = ciclo.decidir({"Amount": 20.0, "Time": 1.0})
    assert decision.razon == "modelo"
