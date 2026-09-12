import json

import numpy as np
import pandas as pd
import pytest

from src.agente_triage import AgenteTriage
from src.bitacora_decisiones import leer_bitacora
from src.calibrador import cargar_dataset
from src.ejecutor import Ejecutor
from src.features_recursivas import EstadoRecursivoGlobal
from src.runner import cargar_estado_runner, correr_ciclo_diario


def _escribir_dataset_csv(ruta, n: int, seed: int = 0) -> None:
    """Mismo generador sintético que el resto del proyecto, pero escrito a
    CSV crudo (Time, V1..V28, Amount, Class) -- runner.py llama a
    `cargar_dataset()` internamente, igual que en producción, así que el
    test debe alimentarlo con el mismo formato de archivo real."""
    rng = np.random.default_rng(seed)
    es_fraude = rng.random(n) < 0.05
    data = {"Time": np.arange(n, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n) + (0.5 if i == 1 else 0) * es_fraude
    data["Amount"] = np.where(es_fraude, rng.normal(500, 100, n), rng.normal(50, 20, n)).clip(min=0)
    data["Class"] = es_fraude.astype(int)
    pd.DataFrame(data).to_csv(ruta, index=False)


def _rutas(tmp_path):
    return {
        "dataset": tmp_path / "dataset.csv",
        "artefacto": tmp_path / "artefacto.json",
        "bitacora": tmp_path / "bitacora.jsonl",
        "estado_runner": tmp_path / "estado_runner.json",
    }


def test_primera_corrida_hace_bootstrap_y_procesa_el_primer_lote(tmp_path):
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=3000, seed=0)

    resultado = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )

    assert resultado.agotado is False
    assert resultado.filas_procesadas == 1000
    assert resultado.indice_procesado == 900 + 1000  # fin_bootstrap (30% de 3000) + el lote
    # ya desde la primera corrida hay una referencia real con qué comparar: la
    # ventana de bootstrap misma (rows[0:900]) contra el primer lote nuevo.
    assert resultado.resultado_disparador is not None
    assert rutas["artefacto"].exists()
    assert len(leer_bitacora(rutas["bitacora"])) == 1000  # el bootstrap no se registra, solo el lote nuevo


def test_segunda_corrida_retoma_desde_el_indice_correcto_sin_reprocesar(tmp_path):
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=3000, seed=0)

    primera = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )
    segunda = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )

    assert segunda.indice_procesado == primera.indice_procesado + 1000
    assert segunda.resultado_disparador is not None  # ya hay un lote anterior con qué comparar
    entradas = leer_bitacora(rutas["bitacora"])
    assert len(entradas) == 2000  # 1000 de la primera + 1000 de la segunda, ninguna repetida


def test_crash_simulado_entre_registrar_y_guardar_estado_no_duplica(tmp_path):
    # Hallazgo crítico de auditoría, reproducido de punta a punta: simula
    # una corrida anterior que se interrumpió justo después de registrar la
    # decisión de la fila en `indice_procesado` pero ANTES de persistir el
    # nuevo índice -- se escribe esa entrada a mano en la bitácora sin
    # avanzar `estado_runner.json`, exactamente el estado en el que
    # quedaría el sistema real tras ese crash.
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=3000, seed=0)

    correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )  # indice_procesado queda en 1900

    indice_antes_del_crash = cargar_estado_runner(rutas["estado_runner"])["indice_procesado"]
    with rutas["bitacora"].open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": "2026-01-01T00:00:00+00:00", "tipo": "modelo", "es_sospechosa": False,
            "razon": "modelo", "score": 0.1, "transaccion": {}, "indice_fila": indice_antes_del_crash,
        }) + "\n")
    n_entradas_antes = len(leer_bitacora(rutas["bitacora"]))

    correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )

    entradas = leer_bitacora(rutas["bitacora"])
    indices_de_esa_fila = [e for e in entradas if e.get("indice_fila") == indice_antes_del_crash]
    assert len(indices_de_esa_fila) == 1  # nunca 2 -- no se duplicó
    assert len(entradas) == n_entradas_antes + 999  # el resto del lote sí avanza normal (1000 filas - la ya registrada)


def test_dataset_agotado_no_reprocesa_ni_falla(tmp_path):
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=1200, seed=1)

    # Con frac_bootstrap=0.3 y n=1200, el bootstrap consume 360 filas -- un
    # lote de 1000 agota las 840 restantes en una sola corrida.
    primera = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )
    assert primera.agotado is False
    assert primera.indice_procesado == 1200

    segunda = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )
    assert segunda.agotado is True
    assert segunda.filas_procesadas == 0
    assert len(leer_bitacora(rutas["bitacora"])) == primera.filas_procesadas  # nada nuevo se agregó


def test_estado_recursivo_se_guarda_y_restaura_exactamente(tmp_path):
    # No compara scores de punta a punta entre corridas fragmentadas vs. una
    # sola (el disparador puede recalibrar en el punto de corte intermedio,
    # y ESO sí cambiaría legítimamente el artefacto -- no sería un bug).
    # Lo que sí debe coincidir siempre es el estado recursivo persistido: se
    # reconstruye a mano, fuera de CicloDecision, procesando las mismas
    # filas en el mismo orden causal, y se compara contra lo que el runner
    # guardó.
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=3000, seed=0)

    correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )

    estado_guardado = cargar_estado_runner(rutas["estado_runner"])["estado_ejecutor"]

    df = cargar_dataset(rutas["dataset"])
    estado_esperado = EstadoRecursivoGlobal()
    for _, fila in df.iloc[:1900].iterrows():  # 900 de bootstrap + 1000 del primer lote
        estado_esperado.leer_features(fila["Amount"], fila["Time"])
        estado_esperado.actualizar(fila["Amount"], fila["Time"])

    assert estado_guardado["monto_ewma"] == pytest.approx(estado_esperado.monto_ewma, abs=1e-9)
    assert estado_guardado["tiempos_ventana"] == list(estado_esperado.tiempos_ventana)
    assert estado_guardado["ultimo_tiempo_visto"] == pytest.approx(estado_esperado.ultimo_tiempo_visto)


def test_error_de_red_del_llm_durante_triage_no_tumba_la_corrida_completa(tmp_path, monkeypatch):
    # Hallazgo de auditoría: solo se capturaban CircuitoAbierto/PresupuestoAgotado
    # alrededor del triage -- un error real de cliente (timeout, 5xx, lo que
    # sea) revienta sin capturar y tumba TODO el job, incluso después de que
    # ya se guardaron decisiones y estado reales. Se fuerza UNA sola falla
    # real de Ejecutor (KeyError en una fila puntual, vía monkeypatch --
    # igual que tests/test_ciclo.py) para producir un escalamiento operativo
    # genuino sin tocar los datos que alimentan una posible recalibración.
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=3000, seed=0)

    decidir_original = Ejecutor.decidir

    def decidir_con_una_falla(self, transaccion):
        if transaccion["Time"] == 950.0:
            raise KeyError("V1")
        return decidir_original(self, transaccion)

    monkeypatch.setattr(Ejecutor, "decidir", decidir_con_una_falla)

    def cliente_que_revienta(prompt):
        raise ConnectionError("simulando un timeout real del cliente Anthropic")

    agente = AgenteTriage(cliente_llm=cliente_que_revienta)

    resultado = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3, agente_triage=agente,
    )  # no debe propagar el ConnectionError

    assert resultado.n_triages_intentados == 1
    assert resultado.n_triages_exitosos == 0
    assert resultado.n_triages_fallidos == 1
    assert resultado.agotado is False  # el resto del job (decisiones, estado) sí se completó


def test_triage_se_intenta_sobre_escalamientos_operativos_nuevos_del_lote(tmp_path):
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=3000, seed=0)

    respuesta_valida = json.dumps({
        "hipotesis": "causa de prueba", "severidad": "baja",
        "accion_sugerida": "sin_accion_clara", "evidencia_citada": [], "confianza": 0.5,
    })
    llamadas = []

    def cliente_fake(prompt):
        llamadas.append(prompt)
        return respuesta_valida

    agente = AgenteTriage(cliente_llm=cliente_fake)

    # Primera corrida: normalmente no hay escalamientos operativos (dataset limpio) --
    # solo confirma que no revienta con el agente configurado y sin nada que triar.
    resultado = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3, agente_triage=agente,
    )

    assert resultado.n_triages_intentados == 0
    assert llamadas == []
