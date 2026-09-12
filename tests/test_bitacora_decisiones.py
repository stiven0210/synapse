from src.bitacora_decisiones import (
    TipoEscalamiento,
    clasificar_razon,
    filtrar_escalamientos_operativos,
    leer_bitacora,
    registrar_decision,
)
from src.ciclo import CicloDecision
from src.puente import publicar
from src.veto import DecisionFinal, evaluar

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


# --- Nivel 1: unidad, determinista ------------------------------------------------

def test_registrar_y_leer_round_trip(tmp_path):
    ruta = tmp_path / "bitacora.jsonl"
    decision = DecisionFinal(es_sospechosa=True, razon="modelo", score=0.9)
    registrar_decision(decision, {"Amount": 50.0, "Time": 0.0}, ruta, timestamp="2026-01-01T00:00:00+00:00")

    entradas = leer_bitacora(ruta)

    assert len(entradas) == 1
    assert entradas[0]["tipo"] == "modelo"
    assert entradas[0]["es_sospechosa"] is True
    assert entradas[0]["score"] == 0.9
    assert entradas[0]["transaccion"] == {"Amount": 50.0, "Time": 0.0}
    assert entradas[0]["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_registrar_multiples_veces_no_sobreescribe_entradas_previas(tmp_path):
    ruta = tmp_path / "bitacora.jsonl"
    for monto in (10.0, 20.0, 30.0):
        registrar_decision(DecisionFinal(es_sospechosa=False, razon="modelo", score=0.1), {"Amount": monto, "Time": 0.0}, ruta)

    entradas = leer_bitacora(ruta)

    assert len(entradas) == 3
    assert [e["transaccion"]["Amount"] for e in entradas] == [10.0, 20.0, 30.0]


def test_leer_bitacora_archivo_inexistente_devuelve_lista_vacia(tmp_path):
    assert leer_bitacora(tmp_path / "no_existe.jsonl") == []


def test_clasificar_razon_mapea_los_4_tipos_conocidos():
    assert clasificar_razon("modelo") == TipoEscalamiento.MODELO
    assert clasificar_razon("|monto| 50000.0 excede el límite absoluto 10000.0") == TipoEscalamiento.MONTO_EXCEDE_LIMITE
    assert clasificar_razon("score fuera de rango numérico válido — escalar a revisión manual") == TipoEscalamiento.SCORE_INVALIDO
    assert clasificar_razon("error en el ejecutor: 'Amount' — escalar a revisión manual") == TipoEscalamiento.ERROR_EJECUTOR


def test_clasificar_razon_cubre_los_textos_reales_de_veto_y_ciclo():
    # No copia los strings a mano -- llama a veto.evaluar() de verdad, para que un
    # cambio de texto en veto.py rompa este test en vez de misclasificar en silencio.
    normal = evaluar({"score": 0.9, "es_sospechosa": True}, {"Amount": 50.0})
    assert clasificar_razon(normal.razon) == TipoEscalamiento.MODELO

    por_monto = evaluar({"score": 0.01, "es_sospechosa": False}, {"Amount": 50_000.0})
    assert clasificar_razon(por_monto.razon) == TipoEscalamiento.MONTO_EXCEDE_LIMITE

    por_score = evaluar({"score": float("nan"), "es_sospechosa": False}, {"Amount": 50.0})
    assert clasificar_razon(por_score.razon) == TipoEscalamiento.SCORE_INVALIDO


def test_filtrar_escalamientos_operativos_excluye_modelo_y_monto(tmp_path):
    entradas = [
        {"tipo": "modelo"},
        {"tipo": "monto_excede_limite"},
        {"tipo": "score_invalido"},
        {"tipo": "error_ejecutor"},
    ]
    operativos = filtrar_escalamientos_operativos(entradas)
    assert {e["tipo"] for e in operativos} == {"score_invalido", "error_ejecutor"}


# --- Nivel 2: escenarios de causa real conocida, de punta a punta -----------------
# A diferencia de tests/test_ciclo.py (que monkeypatchea Ejecutor.decidir directo
# para aislar el Veto), estos alimentan una condición de datos real y verifican que
# TODO el camino -- Ejecutor -> Veto -> CicloDecision -> bitácora -- termina
# clasificado correctamente. Es la base para poder verificar después, con un
# agente de triage, si su hipótesis señala la causa correcta.

def test_escenario_feature_corrupta_nan_se_clasifica_como_score_invalido(tmp_path):
    # Causa real: un pipeline de datos aguas arriba entrega Amount=NaN para una
    # transacción -- el Ejecutor no lanza excepción, pero el score resultante es NaN.
    ruta_artefacto = tmp_path / "artefacto.json"
    ruta_bitacora = tmp_path / "bitacora.jsonl"
    publicar(ARTEFACTO, ruta_artefacto)
    ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)

    transaccion = {"Amount": float("nan"), "Time": 0.0}
    decision = ciclo.decidir(transaccion)
    registrar_decision(decision, transaccion, ruta_bitacora)

    entradas = leer_bitacora(ruta_bitacora)
    assert entradas[0]["tipo"] == "score_invalido"
    assert len(filtrar_escalamientos_operativos(entradas)) == 1


def test_escenario_feature_faltante_se_clasifica_como_error_ejecutor(tmp_path):
    # Causa real: la fuente de transacciones no manda "Amount" -- desajuste de
    # esquema entre esa fuente y lo que el artefacto vigente espera.
    ruta_artefacto = tmp_path / "artefacto.json"
    ruta_bitacora = tmp_path / "bitacora.jsonl"
    publicar(ARTEFACTO, ruta_artefacto)
    ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)

    transaccion = {"Time": 0.0}  # falta "Amount"
    decision = ciclo.decidir(transaccion)
    registrar_decision(decision, transaccion, ruta_bitacora)

    entradas = leer_bitacora(ruta_bitacora)
    assert entradas[0]["tipo"] == "error_ejecutor"
    assert len(filtrar_escalamientos_operativos(entradas)) == 1


def test_escenario_decision_normal_no_aparece_como_escalamiento_operativo(tmp_path):
    ruta_artefacto = tmp_path / "artefacto.json"
    ruta_bitacora = tmp_path / "bitacora.jsonl"
    publicar(ARTEFACTO, ruta_artefacto)
    ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)

    transaccion = {"Amount": 20.0, "Time": 0.0}
    registrar_decision(ciclo.decidir(transaccion), transaccion, ruta_bitacora)

    entradas = leer_bitacora(ruta_bitacora)
    assert entradas[0]["tipo"] == "modelo"
    assert filtrar_escalamientos_operativos(entradas) == []
