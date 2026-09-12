"""Agente de triage sobre escalamientos operativos de Veto —
`docs/ADR_003_agente_triage_veto.md`. Implementa la misma disciplina de 3
capas que el agente de an earlier project (`an ADR from an earlier project`
de ese proyecto), replicada aquí como código nuevo e independiente — SYNAPSE
no comparte código con otros proyectos (`CLAUDE.md`).

**Nunca participa en el camino caliente**: no se invoca desde
`CicloDecision.decidir()`. Se corre después, sobre entradas ya escritas en
`src/bitacora_decisiones.py` (solo las de tipo `SCORE_INVALIDO` o
`ERROR_EJECUTOR` — las que representan una falla de sistema, no un juicio
sobre una transacción; ver `filtrar_escalamientos_operativos()`).

**Nunca decide ni bloquea nada** — el output es una hipótesis para que un
humano la verifique, nunca cambia ningún estado del sistema.

Cliente LLM inyectable (`cliente_llm: Callable[[str], str]`) para poder
testear la lógica de auditoría sin API real — en producción, pasar
`crear_cliente_claude()` (requiere `ANTHROPIC_API_KEY` en el entorno, nunca
hardcoded), envuelto en `limitador_llamadas.LimitadorLlamadasDiarias`.
"""
import json
import random
from dataclasses import dataclass, field
from typing import Callable

SEVERIDADES_VALIDAS = {"baja", "media", "alta"}
ACCIONES_VALIDAS = {"investigar_pipeline_datos", "verificar_artefacto", "verificar_esquema_transaccion", "sin_accion_clara"}

TASA_MUESTREO_AUDITOR_DEFECTO = 0.2  # Capa 3: 1 de cada 5, no el 100% (mismo principio que ADR_001 de an earlier project)
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
    """El output del LLM no cumple el schema -- se descarta, nunca llega al humano como hipótesis."""


class CircuitoAbierto(Exception):
    """Tasa de descarte reciente demasiado alta -- el agente se apaga solo
    antes de gastar en una llamada más que probablemente se va a descartar.
    El llamador debe caer al reporte plano determinista (la entrada de
    bitácora tal cual, sin narrativa)."""


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
    """Ventana deslizante de los últimos `ventana` resultados (descartado o
    no). Si no hay suficiente historial todavía, se asume cerrado (no
    apagar el agente antes de tener señal real)."""
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


def _capa1_validar_schema(texto_respuesta: str) -> ResultadoLLM:
    """Chequeo determinista sobre el 100% de las respuestas: JSON bien
    formado, campos presentes, enums y rangos válidos. Costo cero, sin LLM."""
    try:
        data = json.loads(texto_respuesta)
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
    """Chequeo determinista y barato sobre el 100% de las salidas válidas:
    ¿las afirmaciones de `evidencia_citada` comparten vocabulario real con
    los datos que de verdad se le dieron al agente? No es NLP sofisticado a
    propósito (mismo principio que la Capa 2 de an earlier project) -- detecta el caso
    obvio de alucinación total desconectada de los datos; la evaluación
    semántica real es trabajo de la Capa 3, y esa sí es selectiva.

    Umbral de longitud de palabra en 3 (no 4 como en an earlier project): acá el
    vocabulario verificable son tokens técnicos cortos (nombres de feature
    como "V14", "NaN"), no prosa en inglés."""
    afirmaciones = resultado.evidencia_citada
    if not afirmaciones:
        return True  # nada que groundear

    datos_reales = json.dumps({"entrada": entrada, "contexto": contexto}, ensure_ascii=False).lower()
    coincidencias = 0
    for afirmacion in afirmaciones:
        palabras_clave = [p for p in str(afirmacion).lower().split() if len(p) >= 3]
        if not palabras_clave or any(p in datos_reales for p in palabras_clave):
            coincidencias += 1
    return (coincidencias / len(afirmaciones)) >= 0.5


def _capa3_auditor(resultado: ResultadoLLM, entrada: dict, contexto: dict, cliente_llm: Callable[[str], str]) -> ResultadoAuditoria:
    """Agent-as-Judge selectivo: solo se invoca si la Capa 2 falla o por
    muestreo aleatorio -- nunca sobre el 100% de los casos."""
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
                # proponente y auditor en desacuerdo -> no se resuelve "por mayoría": se
                # descarta el ciclo completo, el humano ve el reporte plano sin hipótesis.
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
    """Cliente real para producción. Requiere `ANTHROPIC_API_KEY` en el
    entorno (nunca hardcoded) -- `anthropic.Anthropic()` la lee
    automáticamente. Import perezoso de `anthropic` para que el resto del
    módulo (y sus tests) no dependan del paquete instalado."""
    import anthropic

    cliente = anthropic.Anthropic()

    def llamar(prompt: str) -> str:
        respuesta = cliente.messages.create(
            model=modelo, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        return respuesta.content[0].text

    return llamar
