"""Triage agent for operational Veto escalations —
`docs/adr/0003-triage-agent.md`. Implements the same 3-layer discipline
used in a similar agent from an earlier project of ours, replicated here
as new, independent code — SYNAPSE shares no code with other projects
(`CLAUDE.md`).

**Never participates in the hot path**: it is never invoked from
`CicloDecision.decidir()`. It runs afterward, over entries already
written to `src/bitacora_decisiones.py` (only the ones of type
`SCORE_INVALIDO` or `ERROR_EJECUTOR` — the ones representing a system
failure, not a judgment call about a transaction; see
`filtrar_escalamientos_operativos()`).

**Never decides or blocks anything** — its output is a hypothesis for a
human to verify, it never changes any system state.

Injectable LLM client (`cliente_llm: Callable[[str], str]`) so the
auditing logic can be tested without a real API — in production, pass
`crear_cliente_claude()` (requires `ANTHROPIC_API_KEY` in the environment,
never hardcoded), wrapped in `limitador_llamadas.LimitadorLlamadasDiarias`.
"""
import json
import random
from dataclasses import dataclass, field
from typing import Callable

SEVERIDADES_VALIDAS = {"baja", "media", "alta"}
ACCIONES_VALIDAS = {"investigar_pipeline_datos", "verificar_artefacto", "verificar_esquema_transaccion", "sin_accion_clara"}

TASA_MUESTREO_AUDITOR_DEFECTO = 0.2  # Layer 3: 1 in 5, not 100% (same principle used in an earlier project of ours)
VENTANA_CIRCUITO_DEFECTO = 20
UMBRAL_TASA_DESCARTE_DEFECTO = 0.5

PROMPT_TRIAGE = """Eres un asistente de triage para un sistema de detección de fraude.
Te doy UNA entrada de la bitácora de decisiones que fue escalada por una falla de
*sistema* (el modelo no pudo producir una decisión normal) -- no es una alerta de
fraude en sí misma. Tu trabajo es proponer una hipótesis de causa para que un
humano la verifique -- nunca decides nada, nunca bloqueas ni apruebas nada.

Usa SOLO los datos que te doy en ENTRADA y CONTEXTO. No inventes hechos que no
estén ahí. Cada elemento de "evidencia_citada" debe ser una afirmación concreta y
verificable contra ENTRADA/CONTEXTO (ej. un nombre de feature, un valor, una fecha
que sí aparecen ahí) -- no una conjetura sin respaldo.

Responde SOLO en JSON:
{{
  "hipotesis": "explicación breve de la causa más probable",
  "severidad": "baja|media|alta",
  "accion_sugerida": "investigar_pipeline_datos|verificar_artefacto|verificar_esquema_transaccion|sin_accion_clara",
  "evidencia_citada": ["afirmaciones concretas y verificables contra ENTRADA/CONTEXTO"],
  "confianza": 0.0 a 1.0
}}

ENTRADA:
{entrada}

CONTEXTO:
{contexto}
"""


class FalloCapa1(Exception):
    """The LLM output doesn't meet the schema -- discarded, never reaches the human as a hypothesis."""


class CircuitoAbierto(Exception):
    """Recent discard rate too high -- the agent shuts itself off before
    spending on another call that would likely be discarded anyway.
    The caller should fall back to the flat, deterministic report (the log
    entry as-is, with no narrative)."""


@dataclass(frozen=True)
class ResultadoLLM:
    hipotesis: str
    severidad: str
    accion_sugerida: str
    evidencia_citada: list
    confianza: float


@dataclass(frozen=True)
class ResultadoAuditoria:
    aprobado: bool
    detalle: str


@dataclass(frozen=True)
class ResultadoTriage:
    descartado: bool
    hipotesis: str | None
    severidad: str | None
    accion_sugerida: str | None
    confianza: float
    detalle: dict


@dataclass
class CircuitoTriage:
    """Sliding window over the last `ventana` results (discarded or not).
    If there isn't enough history yet, it's assumed closed (don't shut the
    agent off before there's real signal)."""
    ventana: int = VENTANA_CIRCUITO_DEFECTO
    umbral_tasa_descarte: float = UMBRAL_TASA_DESCARTE_DEFECTO
    _historial: list = field(default_factory=list, init=False, repr=False)

    def registrar(self, descartado: bool) -> None:
        self._historial.append(descartado)
        if len(self._historial) > self.ventana:
            self._historial.pop(0)

    @property
    def abierto(self) -> bool:
        if len(self._historial) < self.ventana:
            return False
        return (sum(self._historial) / len(self._historial)) >= self.umbral_tasa_descarte


def _despojar_bloque_markdown(texto: str) -> str:
    """Real finding (tested with a real ANTHROPIC_API_KEY): even though the
    prompt asks for "JSON ONLY", the model can wrap the response in a
    markdown code block (```json ... ```) -- common LLM behavior, not a
    model error. Stripped before parsing, without trying to interpret
    anything beyond that."""
    texto = texto.strip()
    if texto.startswith("```"):
        primer_salto = texto.find("\n")
        if primer_salto != -1:
            texto = texto[primer_salto + 1:]
        texto = texto.removesuffix("```").strip()
    return texto


def _capa1_validar_schema(texto_respuesta: str) -> ResultadoLLM:
    """Deterministic check over 100% of responses: well-formed JSON,
    required fields present, valid enums and ranges. Zero cost, no LLM."""
    try:
        data = json.loads(_despojar_bloque_markdown(texto_respuesta))
    except json.JSONDecodeError as e:
        raise FalloCapa1(f"JSON inválido: {e}") from e

    campos_requeridos = {"hipotesis", "severidad", "accion_sugerida", "evidencia_citada", "confianza"}
    faltantes = campos_requeridos - data.keys()
    if faltantes:
        raise FalloCapa1(f"campos faltantes: {faltantes}")

    confianza = data["confianza"]
    if isinstance(confianza, bool) or not isinstance(confianza, (int, float)) or not (0.0 <= confianza <= 1.0):
        raise FalloCapa1(f"confianza fuera de [0,1] o no numérica: {confianza!r}")
    if data["severidad"] not in SEVERIDADES_VALIDAS:
        raise FalloCapa1(f"severidad inválida: {data['severidad']!r}")
    if data["accion_sugerida"] not in ACCIONES_VALIDAS:
        raise FalloCapa1(f"accion_sugerida inválida: {data['accion_sugerida']!r}")
    if not isinstance(data["evidencia_citada"], list):
        raise FalloCapa1("evidencia_citada debe ser una lista")
    if not isinstance(data["hipotesis"], str) or not data["hipotesis"]:
        raise FalloCapa1("hipotesis debe ser un texto no vacío")

    return ResultadoLLM(
        hipotesis=data["hipotesis"],
        severidad=data["severidad"],
        accion_sugerida=data["accion_sugerida"],
        evidencia_citada=list(data["evidencia_citada"]),
        confianza=float(confianza),
    )


def _capa2_grounding(resultado: ResultadoLLM, entrada: dict, contexto: dict) -> bool:
    """Cheap, deterministic check over 100% of valid outputs: do the claims
    in `evidencia_citada` share real vocabulary with the data actually
    given to the agent? Not sophisticated NLP, by design (same principle
    used in an earlier project of ours) -- it catches the obvious case of
    total hallucination disconnected from the data; real semantic
    evaluation is Layer 3's job, and that one is selective.

    Word-length threshold of 3 (lower than in that earlier project): here
    the verifiable vocabulary is short technical tokens (feature names
    like "V14", "NaN"), not English prose.

    **Fix for a real audit finding**: an empty `evidencia_citada` used to
    return `True` ("nothing to ground") -- but Layer 1 only requires it to
    be a list, not to have content, so a hypothesis citing no evidence at
    all passed Layer 2 for free, without ever needing Layer 3. That
    inverts the incentive Layer 2 exists to create: citing nothing was
    safer for a bad hypothesis than citing something verifiable. With no
    evidence cited, there's nothing to confirm the hypothesis -- it's
    treated as failed grounding, forcing Layer 3."""
    afirmaciones = resultado.evidencia_citada
    if not afirmaciones:
        return False

    datos_reales = json.dumps({"entrada": entrada, "contexto": contexto}, ensure_ascii=False).lower()
    coincidencias = 0
    for afirmacion in afirmaciones:
        palabras_clave = [p for p in str(afirmacion).lower().split() if len(p) >= 3]
        if not palabras_clave or any(p in datos_reales for p in palabras_clave):
            coincidencias += 1
    return (coincidencias / len(afirmaciones)) >= 0.5


def _capa3_auditor(resultado: ResultadoLLM, entrada: dict, contexto: dict, cliente_llm: Callable[[str], str]) -> ResultadoAuditoria:
    """Selective Agent-as-Judge: only invoked if Layer 2 fails or by random
    sampling -- never over 100% of cases."""
    prompt_auditor = (
        "Eres un auditor escéptico de un sistema de triage de incidentes. Te doy los "
        "datos reales (ENTRADA y CONTEXTO) y una hipótesis que otro asistente propuso "
        "sobre esos datos. Evalúa si la hipótesis está respaldada por los datos reales "
        "y si no inventa nada. Responde SOLO JSON: "
        '{"de_acuerdo": true o false, "razon": "..."}.\n\n'
        f"ENTRADA:\n{json.dumps(entrada, ensure_ascii=False)}\n\n"
        f"CONTEXTO:\n{json.dumps(contexto, ensure_ascii=False)}\n\n"
        f"HIPÓTESIS A AUDITAR:\n{json.dumps(resultado.__dict__, ensure_ascii=False)}"
    )
    respuesta = cliente_llm(prompt_auditor)
    try:
        data = json.loads(respuesta)
        de_acuerdo = bool(data["de_acuerdo"])
        razon = str(data.get("razon", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        return ResultadoAuditoria(aprobado=False, detalle="respuesta del auditor no es JSON válido")
    return ResultadoAuditoria(aprobado=de_acuerdo, detalle=razon)


@dataclass
class AgenteTriage:
    cliente_llm: Callable[[str], str] | None = None
    tasa_muestreo_auditor: float = TASA_MUESTREO_AUDITOR_DEFECTO
    rng: random.Random = field(default_factory=random.Random)
    circuito: CircuitoTriage = field(default_factory=CircuitoTriage)

    def triar(self, entrada: dict, contexto: dict) -> ResultadoTriage:
        if self.cliente_llm is None:
            raise ValueError("cliente_llm no configurado — ver crear_cliente_claude()")
        if self.circuito.abierto:
            raise CircuitoAbierto(
                f"tasa de descarte >= {self.circuito.umbral_tasa_descarte} en los últimos "
                f"{self.circuito.ventana} triages — agente apagado, usar el reporte plano"
            )

        respuesta_cruda = self.cliente_llm(
            PROMPT_TRIAGE.format(
                entrada=json.dumps(entrada, ensure_ascii=False),
                contexto=json.dumps(contexto, ensure_ascii=False),
            )
        )

        try:
            resultado = _capa1_validar_schema(respuesta_cruda)
        except FalloCapa1 as e:
            self.circuito.registrar(descartado=True)
            return ResultadoTriage(
                descartado=True, hipotesis=None, severidad=None, accion_sugerida=None, confianza=0.0,
                detalle={"capa_fallida": "capa1", "razon": str(e)},
            )

        paso_capa2 = _capa2_grounding(resultado, entrada, contexto)
        necesita_capa3 = (not paso_capa2) or (self.rng.random() < self.tasa_muestreo_auditor)

        detalle = {"paso_capa2_grounding": paso_capa2, "auditado_capa3": necesita_capa3}

        if necesita_capa3:
            auditoria = _capa3_auditor(resultado, entrada, contexto, self.cliente_llm)
            detalle["auditoria_capa3"] = auditoria.detalle
            if not auditoria.aprobado:
                # proposer and auditor disagree -> not resolved "by majority": the
                # whole cycle is discarded, the human sees the flat report with no hypothesis.
                self.circuito.registrar(descartado=True)
                detalle["capa_fallida"] = "capa3"
                return ResultadoTriage(
                    descartado=True, hipotesis=None, severidad=None, accion_sugerida=None, confianza=0.0, detalle=detalle,
                )

        self.circuito.registrar(descartado=False)
        return ResultadoTriage(
            descartado=False,
            hipotesis=resultado.hipotesis,
            severidad=resultado.severidad,
            accion_sugerida=resultado.accion_sugerida,
            confianza=resultado.confianza,
            detalle=detalle,
        )


def crear_cliente_claude(modelo: str = "claude-sonnet-5") -> Callable[[str], str]:
    """Real client for production. Requires `ANTHROPIC_API_KEY` in the
    environment (never hardcoded) -- `anthropic.Anthropic()` reads it
    automatically. Lazy import of `anthropic` so the rest of the module
    (and its tests) don't depend on the package being installed."""
    import anthropic

    cliente = anthropic.Anthropic()

    def llamar(prompt: str) -> str:
        """Real finding (tested with a real ANTHROPIC_API_KEY, not just the
        fake client used in tests): `respuesta.content[0]` isn't always
        text -- the model can return an extended-thinking block
        (`ThinkingBlock`) first, and `.content[0].text` blows up with
        `AttributeError`. Explicitly looks for the block(s) of type "text"
        instead of assuming the position."""
        respuesta = cliente.messages.create(
            model=modelo, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        bloques_texto = [bloque.text for bloque in respuesta.content if bloque.type == "text"]
        if not bloques_texto:
            raise ValueError(f"la respuesta del modelo no tiene ningún bloque de texto: {respuesta.content!r}")
        return "".join(bloques_texto)

    return llamar
