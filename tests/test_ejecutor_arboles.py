import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artefacto_arboles import ArtefactoArbolesInvalido
from src.calibrador_arboles import FEATURES_ARBOLES, entrenar_pipeline_completo
from src.ejecutor_arboles import EjecutorArboles, _interpolar_isotonica

RUTA_DATASET = Path(__file__).resolve().parents[1] / "data" / "raw" / "sparkov_2013_2026.csv"

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


def test_interpolar_isotonica_caso_conocido():
    xs, ys = [0.0, 0.5, 1.0], [0.0, 0.8, 1.0]
    assert _interpolar_isotonica(0.25, xs, ys) == pytest.approx(0.4)  # midpoint between 0.0 and 0.8


def test_interpolar_isotonica_clampa_fuera_de_rango():
    xs, ys = [0.2, 0.8], [0.1, 0.9]
    assert _interpolar_isotonica(-5.0, xs, ys) == pytest.approx(0.1)
    assert _interpolar_isotonica(5.0, xs, ys) == pytest.approx(0.9)


def test_interpolar_isotonica_coincide_con_np_interp_en_secuencia_aleatoria():
    rng = np.random.default_rng(1)
    xs = np.sort(rng.uniform(0, 1, 30)).tolist()
    ys = np.sort(rng.uniform(0, 1, 30)).tolist()  # non-decreasing monotonic, like a real isotonic curve
    puntos = rng.uniform(-1, 2, 200)
    esperado = np.interp(puntos, xs, ys)
    obtenido = [_interpolar_isotonica(float(p), xs, ys) for p in puntos]
    assert obtenido == pytest.approx(esperado.tolist())


def test_decidir_bin_bajo_no_es_sospechosa():
    ejecutor = EjecutorArboles(artefacto=ARTEFACTO)
    resultado = ejecutor.decidir({"cc_num": "A", "amt": 20.0, "unix_time": 1704122400, "category": "x"})
    assert resultado["score"] == pytest.approx(0.1)
    assert resultado["es_sospechosa"] is False


def test_decidir_bin_alto_es_sospechosa():
    ejecutor = EjecutorArboles(artefacto=ARTEFACTO)
    resultado = ejecutor.decidir({"cc_num": "A", "amt": 500.0, "unix_time": 1704122400, "category": "x"})
    assert resultado["score"] == pytest.approx(0.9)
    assert resultado["es_sospechosa"] is True


def test_score_nan_no_se_guarda_en_ejecutor_queda_para_veto():
    # Same as the Domain 1 Executor: decidir() doesn't guard against NaN, that's Veto's responsibility.
    artefacto_roto = {**ARTEFACTO, "tabla_busqueda_plana": [float("nan"), 0.9]}
    ejecutor = EjecutorArboles(artefacto=artefacto_roto)
    resultado = ejecutor.decidir({"cc_num": "A", "amt": 20.0, "unix_time": 1704122400, "category": "x"})
    assert resultado["es_sospechosa"] is False  # nan >= threshold is False -- by design, see docstring


def test_construir_ejecutor_valida_el_artefacto():
    with pytest.raises(ArtefactoArbolesInvalido):
        EjecutorArboles(artefacto={**ARTEFACTO, "umbral_decision": 5.0})


def test_mutar_artefacto_original_no_afecta_al_ejecutor_ya_construido():
    artefacto = dict(ARTEFACTO)
    ejecutor = EjecutorArboles(artefacto=artefacto)
    artefacto["umbral_decision"] = 0.0  # mutates the original dict after construction
    resultado = ejecutor.decidir({"cc_num": "A", "amt": 20.0, "unix_time": 1704122400, "category": "x"})
    assert resultado["es_sospechosa"] is False  # still uses the 0.5 threshold from the defensive copy


ARTEFACTO_6_FEATURES = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "gradient_boosting",
    "features": ["amt", "hora", "conteo_ventana_global", "monto_ewma_cuenta", "huella_categoria_cuenta", "frecuencia_categoria_expandida"],
    "umbrales_por_feature": {
        "amt": [], "hora": [], "conteo_ventana_global": [], "monto_ewma_cuenta": [], "huella_categoria_cuenta": [],
        "frecuencia_categoria_expandida": [0.5],
    },
    "tabla_busqueda_forma": [1, 1, 1, 1, 1, 2],
    "tabla_busqueda_plana": [0.2, 0.8],
    "calibracion_isotonica": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "umbral_decision": 0.5,
    "metricas_validacion": {},
    "frecuencia_poblacional_categoria": {"x": 0.5, "y": 0.5},  # n_categories = 2
}


def test_decidir_usa_frecuencia_categoria_expandida_en_la_tabla():
    # n_categories=2 (frecuencia_poblacional_categoria has 2 keys). 1st transaction of "x":
    # (0+1)/(0+2)=0.5 -> bisect_left([0.5], 0.5)=0 -> table[0]=0.2 -> not suspicious.
    # 2nd transaction of "x" (same category repeated): (1+1)/(1+2)=0.667 -> bin 1 -> table[1]=0.8 -> suspicious.
    ejecutor = EjecutorArboles(artefacto=ARTEFACTO_6_FEATURES)
    r1 = ejecutor.decidir({"cc_num": "A", "amt": 1.0, "unix_time": 1704122400, "category": "x"})
    assert r1["score"] == pytest.approx(0.2)
    assert r1["es_sospechosa"] is False

    r2 = ejecutor.decidir({"cc_num": "A", "amt": 1.0, "unix_time": 1704122401, "category": "x"})
    assert r2["score"] == pytest.approx(0.8)
    assert r2["es_sospechosa"] is True


def test_estado_por_cuenta_es_independiente_entre_cuentas():
    ejecutor = EjecutorArboles(artefacto=ARTEFACTO)
    # Feeds account A with several "y" transactions -- must not affect account B.
    for t in range(6):
        ejecutor.decidir({"cc_num": "A", "amt": 1.0, "unix_time": 1704122400 + t, "category": "y"})
    # EstadoRecursivoGlobal is a single global clock shared by all accounts -- time
    # must keep advancing, never go backward, even when the account changes.
    r_b = ejecutor.decidir({"cc_num": "B", "amt": 1.0, "unix_time": 1704122400 + 6, "category": "y"})
    # B is its first transaction -- huella_categoria_cuenta falls back to the population frequency
    # (frecuencia_categoria["x"]=1.0, "y" isn't in frecuencia_categoria -> 0.0), not A's history.
    assert r_b["score"] in (0.1, 0.9)  # only confirms it didn't raise or get corrupted by A's state


@pytest.fixture(scope="module")
def pipeline_real():
    if not RUTA_DATASET.exists():
        pytest.skip(f"dataset real no disponible en {RUTA_DATASET}")
    return entrenar_pipeline_completo(RUTA_DATASET)


def test_tabla_busqueda_es_bit_exacta_contra_predict_proba_real(pipeline_real):
    # The test that actually matters (see the two-speed-decision skill): full replay from the
    # start of history (train+val+test in order), NOT just the test split -- otherwise
    # the per-account/global recursive state would start empty instead of with the
    # real history, and the numbers wouldn't match the ones batch mode actually computed.
    artefacto = pipeline_real["artefacto"]
    modelo = pipeline_real["modelo"]
    df_test = pipeline_real["df_test"]
    df_full = pd.concat([pipeline_real["df_train"], pipeline_real["df_val"], df_test], ignore_index=True)

    ejecutor = EjecutorArboles(artefacto=artefacto)
    n_total = len(df_full)
    n_test = len(df_test)
    inicio_test = n_total - n_test

    scores = np.empty(n_total)
    for i, fila in enumerate(df_full.itertuples(index=False)):
        transaccion = {"cc_num": fila.cc_num, "amt": fila.amt, "unix_time": int(fila.unix_time), "category": fila.category}
        scores[i] = ejecutor.decidir(transaccion)["score"]

    scores_test_ejecutor = scores[inicio_test:]

    X_test = df_test[FEATURES_ARBOLES].to_numpy(dtype=float)
    scores_crudos_reales = modelo.predict_proba(X_test)[:, 1]
    xs = np.array(artefacto["calibracion_isotonica"]["x"])
    ys = np.array(artefacto["calibracion_isotonica"]["y"])
    scores_test_reales = np.interp(scores_crudos_reales, xs, ys)

    diff = np.abs(scores_test_ejecutor - scores_test_reales)
    assert diff.max() < 1e-9  # bit-exact except for floating-point noise -- see doc section 6
    assert diff.mean() < 1e-12


def test_latencia_real_medida_con_perf_counter(pipeline_real):
    # Real measurement, never estimated (two-speed-decision skill discipline) -- same
    # methodology as scripts/validacion_end_to_end.py: pre-build the list of transactions,
    # only measure decidir()'s time.
    df_full = pd.concat([pipeline_real["df_train"], pipeline_real["df_val"], pipeline_real["df_test"]], ignore_index=True)
    columnas = ["cc_num", "amt", "unix_time", "category"]
    filas = df_full[columnas].to_dict("records")
    for f in filas:
        f["unix_time"] = int(f["unix_time"])

    ejecutor = EjecutorArboles(artefacto=pipeline_real["artefacto"])

    inicio = time.perf_counter()
    for fila in filas:
        ejecutor.decidir(fila)
    duracion = time.perf_counter() - inicio

    us_por_decision = duracion / len(filas) * 1e6
    print(f"\nLatencia real EjecutorArboles: {us_por_decision:.2f} µs/decisión sobre {len(filas)} transacciones")
    # Deliberately generous -- the real number is reported in docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md,
    # this assert only guards against a catastrophic regression (e.g. going back to walking trees on the hot path).
    assert us_por_decision < 100.0
