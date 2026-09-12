import json
import sys
from unittest.mock import MagicMock

import pytest

from src.agente_triage import AgenteTriage, CircuitoAbierto, CircuitoTriage, crear_cliente_claude


class _RngFijo:
    """Doble de random.Random con .random() fijo, para controlar el
    muestreo de la Capa 3 sin depender de una semilla real."""

    def __init__(self, valor: float):
        self.valor = valor

    def random(self) -> float:
        return self.valor


def _respuesta_valida(**overrides) -> str:
    base = {
        "hipotesis": "La transacción tiene Amount en NaN, probablemente un dato corrupto aguas arriba",
        "severidad": "media",
        "accion_sugerida": "investigar_pipeline_datos",
        "evidencia_citada": ["Amount aparece como NaN en la transacción"],
        "confianza": 0.7,
    }
    base.update(overrides)
    return json.dumps(base)


ENTRADA = {
    "tipo": "score_invalido",
    "razon": "score fuera de rango numérico válido — escalar a revisión manual",
    "transaccion": {"Amount": float("nan"), "Time": 0.0},
}
CONTEXTO = {"reporte_deriva_reciente": None}


def _cliente_fake(respuesta_triage: str, respuesta_auditor: str | None = None, llamadas: list | None = None):
    def cliente(prompt: str) -> str:
        if llamadas is not None:
            llamadas.append(prompt)
        if prompt.startswith("Eres un auditor"):
            return respuesta_auditor
        return respuesta_triage

    return cliente


# --- Nivel 1: guardrails deterministas, cliente LLM falso -------------------------

def test_respuesta_valida_sin_muestreo_capa3_pasa_directo():
    llamadas = []
    cliente = _cliente_fake(_respuesta_valida(), llamadas=llamadas)
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99))  # 0.99 >= tasa 0.2 -> sin muestreo

    resultado = agente.triar(ENTRADA, CONTEXTO)

    assert resultado.descartado is False
    assert resultado.severidad == "media"
    assert resultado.accion_sugerida == "investigar_pipeline_datos"
    assert resultado.confianza == pytest.approx(0.7)
    assert resultado.detalle["auditado_capa3"] is False
    assert len(llamadas) == 1  # solo la llamada de triage, el auditor no se invocó


def test_json_invalido_se_descarta_en_capa1():
    agente = AgenteTriage(cliente_llm=_cliente_fake("esto no es json"), rng=_RngFijo(0.99))
    resultado = agente.triar(ENTRADA, CONTEXTO)
    assert resultado.descartado is True
    assert resultado.detalle["capa_fallida"] == "capa1"
    assert resultado.hipotesis is None


def test_severidad_invalida_se_descarta_en_capa1():
    cliente = _cliente_fake(_respuesta_valida(severidad="catastrofica"))
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99))
    resultado = agente.triar(ENTRADA, CONTEXTO)
    assert resultado.detalle["capa_fallida"] == "capa1"


def test_accion_sugerida_invalida_se_descarta_en_capa1():
    cliente = _cliente_fake(_respuesta_valida(accion_sugerida="reiniciar_todo"))
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99))
    resultado = agente.triar(ENTRADA, CONTEXTO)
    assert resultado.detalle["capa_fallida"] == "capa1"


def test_confianza_fuera_de_rango_se_descarta_en_capa1():
    cliente = _cliente_fake(_respuesta_valida(confianza=5.0))
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99))
    resultado = agente.triar(ENTRADA, CONTEXTO)
    assert resultado.detalle["capa_fallida"] == "capa1"


def test_campo_faltante_se_descarta_en_capa1():
    respuesta = json.dumps({"hipotesis": "algo"})  # faltan el resto de campos
    agente = AgenteTriage(cliente_llm=_cliente_fake(respuesta), rng=_RngFijo(0.99))
    resultado = agente.triar(ENTRADA, CONTEXTO)
    assert resultado.detalle["capa_fallida"] == "capa1"


def test_grounding_fallido_dispara_capa3_aunque_no_toque_muestreo():
    llamadas = []
    respuesta_desconectada = _respuesta_valida(
        evidencia_citada=["se detectó una intrusión de malware en el servidor remoto"]
    )
    cliente = _cliente_fake(
        respuesta_desconectada,
        respuesta_auditor=json.dumps({"de_acuerdo": False, "razon": "no está respaldado por los datos"}),
        llamadas=llamadas,
    )
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99))  # sin muestreo aleatorio, pero capa2 falla

    resultado = agente.triar(ENTRADA, CONTEXTO)

    assert resultado.detalle["paso_capa2_grounding"] is False
    assert resultado.detalle["auditado_capa3"] is True
    assert resultado.descartado is True
    assert resultado.detalle["capa_fallida"] == "capa3"
    assert len(llamadas) == 2  # triage + auditor


def test_capa3_por_muestreo_aleatorio_con_acuerdo_deja_pasar():
    llamadas = []
    cliente = _cliente_fake(
        _respuesta_valida(),
        respuesta_auditor=json.dumps({"de_acuerdo": True, "razon": "razonable y respaldado"}),
        llamadas=llamadas,
    )
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.01))  # 0.01 < tasa 0.2 -> se muestrea

    resultado = agente.triar(ENTRADA, CONTEXTO)

    assert resultado.detalle["auditado_capa3"] is True
    assert resultado.descartado is False
    assert len(llamadas) == 2


def test_desacuerdo_proponente_auditor_descarta_sin_resolver_por_mayoria():
    cliente = _cliente_fake(
        _respuesta_valida(),
        respuesta_auditor=json.dumps({"de_acuerdo": False, "razon": "sobreestima la causa"}),
    )
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.01))

    resultado = agente.triar(ENTRADA, CONTEXTO)

    assert resultado.descartado is True
    assert resultado.detalle["capa_fallida"] == "capa3"
    assert resultado.hipotesis is None


def test_respuesta_del_auditor_no_json_se_trata_como_desacuerdo():
    cliente = _cliente_fake(_respuesta_valida(), respuesta_auditor="no soy json")
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.01))
    resultado = agente.triar(ENTRADA, CONTEXTO)
    assert resultado.descartado is True
    assert resultado.detalle["capa_fallida"] == "capa3"


def test_respuesta_envuelta_en_bloque_markdown_json_se_parsea_igual():
    # Hallazgo real: pese a "Responde SOLO en JSON", el modelo puede envolver
    # la respuesta en ```json ... ``` -- debe parsearse igual, no descartarse.
    respuesta_con_fence = "```json\n" + _respuesta_valida() + "\n```"
    agente = AgenteTriage(cliente_llm=_cliente_fake(respuesta_con_fence), rng=_RngFijo(0.99))

    resultado = agente.triar(ENTRADA, CONTEXTO)

    assert resultado.descartado is False
    assert resultado.severidad == "media"


def test_respuesta_envuelta_en_bloque_markdown_generico_tambien_se_parsea():
    respuesta_con_fence = "```\n" + _respuesta_valida() + "\n```"
    agente = AgenteTriage(cliente_llm=_cliente_fake(respuesta_con_fence), rng=_RngFijo(0.99))

    resultado = agente.triar(ENTRADA, CONTEXTO)

    assert resultado.descartado is False


def test_sin_cliente_llm_configurado_lanza_error():
    agente = AgenteTriage()
    with pytest.raises(ValueError):
        agente.triar(ENTRADA, CONTEXTO)


def test_circuito_se_abre_tras_tasa_de_descarte_alta_y_no_llama_al_cliente():
    llamadas = []
    cliente = _cliente_fake("esto no es json", llamadas=llamadas)  # siempre falla capa1
    circuito = CircuitoTriage(ventana=4, umbral_tasa_descarte=0.5)
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99), circuito=circuito)

    for _ in range(4):
        agente.triar(ENTRADA, CONTEXTO)  # 4 descartes seguidos -> tasa 100% >= 50%
    assert len(llamadas) == 4

    with pytest.raises(CircuitoAbierto):
        agente.triar(ENTRADA, CONTEXTO)
    assert len(llamadas) == 4  # el circuito abierto NO gasta una llamada más


def test_circuito_no_se_abre_con_menos_datos_que_la_ventana():
    cliente = _cliente_fake("esto no es json")
    circuito = CircuitoTriage(ventana=10, umbral_tasa_descarte=0.5)
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99), circuito=circuito)

    for _ in range(9):  # menos que la ventana de 10 -- no hay señal suficiente todavía
        resultado = agente.triar(ENTRADA, CONTEXTO)
        assert resultado.descartado is True
    assert circuito.abierto is False


# --- Nivel 2: escenarios de causa real conocida (mismos que test_bitacora_decisiones.py) --

def test_hipotesis_correcta_grounded_en_datos_reales_pasa():
    # Escenario real: feature Amount en NaN (ver test_bitacora_decisiones.py) -- una
    # hipótesis que cita el dato real correcto debe pasar Capa 2 sin necesitar Capa 3.
    entrada_real = {
        "tipo": "score_invalido",
        "razon": "score fuera de rango numérico válido — escalar a revisión manual",
        "score": None,
        "transaccion": {"Amount": float("nan"), "Time": 0.0},
    }
    respuesta = _respuesta_valida(
        hipotesis="El feature Amount llegó en NaN, el score se propagó como NaN",
        evidencia_citada=["Amount tiene valor NaN en la transacción escalada"],
    )
    agente = AgenteTriage(cliente_llm=_cliente_fake(respuesta), rng=_RngFijo(0.99))

    resultado = agente.triar(entrada_real, CONTEXTO)

    assert resultado.descartado is False
    assert resultado.detalle["paso_capa2_grounding"] is True


def test_hipotesis_alucinada_sin_respaldo_en_datos_reales_se_descarta():
    # Misma entrada real, pero el LLM inventa una causa que no está en los datos --
    # Capa 2 debe fallar y, con el auditor en desacuerdo, descartarse antes de llegar
    # a un humano como si fuera una conclusión confiable.
    entrada_real = {
        "tipo": "score_invalido",
        "razon": "score fuera de rango numérico válido — escalar a revisión manual",
        "score": None,
        "transaccion": {"Amount": float("nan"), "Time": 0.0},
    }
    respuesta = _respuesta_valida(
        hipotesis="Esto fue causado por un ataque de denegación de servicio",
        evidencia_citada=["se detectó tráfico anómalo desde múltiples regiones geográficas"],
    )
    cliente = _cliente_fake(
        respuesta,
        respuesta_auditor=json.dumps({"de_acuerdo": False, "razon": "no hay ningún dato sobre tráfico o ataques en la entrada"}),
    )
    agente = AgenteTriage(cliente_llm=cliente, rng=_RngFijo(0.99))

    resultado = agente.triar(entrada_real, CONTEXTO)

    assert resultado.detalle["paso_capa2_grounding"] is False
    assert resultado.descartado is True
    assert resultado.hipotesis is None


# --- crear_cliente_claude: hallazgo real probado con ANTHROPIC_API_KEY real -------
# `anthropic.Anthropic()` se simula (sin llamada de red real ni costo) para
# reproducir exactamente la forma de respuesta que causó el bug real: un
# bloque de razonamiento extendido (ThinkingBlock) antes del bloque de texto.

def _mockear_anthropic(monkeypatch, bloques_respuesta):
    respuesta_fake = MagicMock(content=bloques_respuesta)
    cliente_fake = MagicMock()
    cliente_fake.messages.create.return_value = respuesta_fake
    modulo_anthropic_fake = MagicMock()
    modulo_anthropic_fake.Anthropic.return_value = cliente_fake
    monkeypatch.setitem(sys.modules, "anthropic", modulo_anthropic_fake)


def test_crear_cliente_claude_ignora_bloques_de_pensamiento_y_extrae_el_texto(monkeypatch):
    bloque_pensamiento = MagicMock(type="thinking")
    bloque_texto = MagicMock(type="text", text='{"hipotesis": "ok"}')
    _mockear_anthropic(monkeypatch, [bloque_pensamiento, bloque_texto])

    llamar = crear_cliente_claude()

    assert llamar("prompt de prueba") == '{"hipotesis": "ok"}'


def test_crear_cliente_claude_sin_ningun_bloque_de_texto_lanza_error_explicito(monkeypatch):
    bloque_pensamiento = MagicMock(type="thinking")
    _mockear_anthropic(monkeypatch, [bloque_pensamiento])

    llamar = crear_cliente_claude()

    with pytest.raises(ValueError, match="ningún bloque de texto"):
        llamar("prompt de prueba")
