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


# --- Level 1: unit, deterministic ------------------------------------------------

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


def test_registrar_decision_guarda_indice_fila_cuando_se_da(tmp_path):
    ruta = tmp_path / "bitacora.jsonl"
    registrar_decision(DecisionFinal(es_sospechosa=False, razon="modelo", score=0.1), {"Amount": 10.0}, ruta, indice_fila=42)

    assert leer_bitacora(ruta)[0]["indice_fila"] == 42


def test_registrar_decision_sin_indice_fila_queda_none(tmp_path):
    ruta = tmp_path / "bitacora.jsonl"
    registrar_decision(DecisionFinal(es_sospechosa=False, razon="modelo", score=0.1), {"Amount": 10.0}, ruta)

    assert leer_bitacora(ruta)[0]["indice_fila"] is None


def test_leer_bitacora_tolera_una_linea_final_truncada(tmp_path):
    # Real audit finding: a write interrupted mid-way through the last
    # line (e.g. a power cut, a process killed mid-write) must not
    # invalidate the earlier entries, which is exactly what this
    # module claims to guarantee.
    ruta = tmp_path / "bitacora.jsonl"
    registrar_decision(DecisionFinal(es_sospechosa=False, razon="modelo", score=0.1), {"Amount": 1.0}, ruta)
    registrar_decision(DecisionFinal(es_sospechosa=True, razon="modelo", score=0.9), {"Amount": 2.0}, ruta)
    with ruta.open("a", encoding="utf-8") as f:
        f.write('{"timestamp": "2026-01-01T00:00:00", "tipo": "modelo", "es_sosp')  # truncated line, unclosed

    entradas = leer_bitacora(ruta)

    assert len(entradas) == 2  # the 2 complete ones are still read, the truncated one is skipped
    assert [e["transaccion"]["Amount"] for e in entradas] == [1.0, 2.0]


def test_clasificar_razon_mapea_los_4_tipos_conocidos():
    assert clasificar_razon("modelo") == TipoEscalamiento.MODELO
    assert clasificar_razon("|monto| 50000.0 excede el límite absoluto 10000.0") == TipoEscalamiento.MONTO_EXCEDE_LIMITE
    assert clasificar_razon("score fuera de rango numérico válido — escalar a revisión manual") == TipoEscalamiento.SCORE_INVALIDO
    assert clasificar_razon("error en el ejecutor: 'Amount' — escalar a revisión manual") == TipoEscalamiento.ERROR_EJECUTOR


def test_clasificar_razon_cubre_los_textos_reales_de_veto_y_ciclo():
    # Doesn't copy the strings by hand -- calls the real veto.evaluar(), so that a
    # text change in veto.py breaks this test instead of silently misclassifying.
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


# --- Level 2: end-to-end scenarios with a known real cause -----------------
# Unlike tests/test_ciclo.py (which monkeypatches Ejecutor.decidir directly
# to isolate the Veto), these feed a real data condition and verify that
# the WHOLE path -- Executor -> Veto -> CicloDecision -> log -- ends up
# classified correctly. This is the foundation for later verifying, with
# a triage agent, whether its hypothesis points to the right cause.

def test_escenario_feature_corrupta_nan_se_clasifica_como_score_invalido(tmp_path):
    # Real cause: an upstream data pipeline delivers Amount=NaN for a
    # transaction -- the Executor doesn't raise, but the resulting score is NaN.
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
    # Real cause: the transaction source doesn't send "Amount" -- a schema
    # mismatch between that source and what the current artifact expects.
    ruta_artefacto = tmp_path / "artefacto.json"
    ruta_bitacora = tmp_path / "bitacora.jsonl"
    publicar(ARTEFACTO, ruta_artefacto)
    ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)

    transaccion = {"Time": 0.0}  # missing "Amount"
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
