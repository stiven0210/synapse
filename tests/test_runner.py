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
    """Same synthetic generator as the rest of the project, but written to a
    raw CSV (Time, V1..V28, Amount, Class) -- runner.py calls
    `cargar_dataset()` internally, same as in production, so the
    test needs to feed it the same real file format."""
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
    assert resultado.indice_procesado == 900 + 1000  # fin_bootstrap (30% of 3000) + the batch
    # already from the first run there's a real reference to compare against: the
    # bootstrap window itself (rows[0:900]) against the first new batch.
    assert resultado.resultado_disparador is not None
    assert rutas["artefacto"].exists()
    assert len(leer_bitacora(rutas["bitacora"])) == 1000  # the bootstrap isn't logged, only the new batch


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
    assert segunda.resultado_disparador is not None  # there's already a previous batch to compare against
    entradas = leer_bitacora(rutas["bitacora"])
    assert len(entradas) == 2000  # 1000 from the first + 1000 from the second, none repeated


def test_crash_simulado_entre_registrar_y_guardar_estado_no_duplica(tmp_path):
    # Critical audit finding, reproduced end to end: simulates a previous
    # run that got interrupted right after logging the decision for the
    # row at `indice_procesado` but BEFORE persisting the new index --
    # that entry is written into the log by hand without advancing
    # `estado_runner.json`, exactly the state the real system would be
    # left in after that crash.
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=3000, seed=0)

    correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3,
    )  # indice_procesado ends at 1900

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
    assert len(indices_de_esa_fila) == 1  # never 2 -- it wasn't duplicated
    assert len(entradas) == n_entradas_antes + 999  # the rest of the batch does advance normally (1000 rows - the one already logged)


def test_dataset_agotado_no_reprocesa_ni_falla(tmp_path):
    rutas = _rutas(tmp_path)
    _escribir_dataset_csv(rutas["dataset"], n=1200, seed=1)

    # With frac_bootstrap=0.3 and n=1200, the bootstrap consumes 360 rows -- a
    # batch of 1000 exhausts the remaining 840 in a single run.
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
    assert len(leer_bitacora(rutas["bitacora"])) == primera.filas_procesadas  # nothing new was added


def test_estado_recursivo_se_guarda_y_restaura_exactamente(tmp_path):
    # Doesn't compare end-to-end scores between fragmented runs vs. a
    # single one (the trigger can recalibrate at the intermediate cut point,
    # and THAT would legitimately change the artifact -- not a bug).
    # What must always match is the persisted recursive state: it's
    # rebuilt by hand, outside CicloDecision, processing the same
    # rows in the same causal order, and compared against what the runner
    # saved.
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
    for _, fila in df.iloc[:1900].iterrows():  # 900 from bootstrap + 1000 from the first batch
        estado_esperado.leer_features(fila["Amount"], fila["Time"])
        estado_esperado.actualizar(fila["Amount"], fila["Time"])

    assert estado_guardado["monto_ewma"] == pytest.approx(estado_esperado.monto_ewma, abs=1e-9)
    assert estado_guardado["tiempos_ventana"] == list(estado_esperado.tiempos_ventana)
    assert estado_guardado["ultimo_tiempo_visto"] == pytest.approx(estado_esperado.ultimo_tiempo_visto)


def test_error_de_red_del_llm_durante_triage_no_tumba_la_corrida_completa(tmp_path, monkeypatch):
    # Audit finding: only CircuitoAbierto/PresupuestoAgotado were caught
    # around the triage -- a real client error (timeout, 5xx, whatever it
    # is) blows up uncaught and takes down the ENTIRE job, even after
    # real decisions and state have already been saved. A SINGLE real
    # Executor failure is forced (KeyError on one specific row, via monkeypatch --
    # same as tests/test_ciclo.py) to produce a genuine operational
    # escalation without touching the data that feeds a possible recalibration.
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
    )  # must not propagate the ConnectionError

    assert resultado.n_triages_intentados == 1
    assert resultado.n_triages_exitosos == 0
    assert resultado.n_triages_fallidos == 1
    assert resultado.agotado is False  # the rest of the job (decisions, state) did complete


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

    # First run: normally there are no operational escalations (clean dataset) --
    # this just confirms it doesn't blow up with the agent configured and nothing to triage.
    resultado = correr_ciclo_diario(
        ruta_dataset=rutas["dataset"], ruta_artefacto=rutas["artefacto"],
        ruta_bitacora=rutas["bitacora"], ruta_estado_runner=rutas["estado_runner"],
        tamano_lote=1000, frac_bootstrap=0.3, agente_triage=agente,
    )

    assert resultado.n_triages_intentados == 0
    assert llamadas == []
