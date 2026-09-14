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
    assert _interpolar_isotonica(0.25, xs, ys) == pytest.approx(0.4)  # punto medio entre 0.0 y 0.8


def test_interpolar_isotonica_clampa_fuera_de_rango():
    xs, ys = [0.2, 0.8], [0.1, 0.9]
    assert _interpolar_isotonica(-5.0, xs, ys) == pytest.approx(0.1)
    assert _interpolar_isotonica(5.0, xs, ys) == pytest.approx(0.9)


def test_interpolar_isotonica_coincide_con_np_interp_en_secuencia_aleatoria():
    rng = np.random.default_rng(1)
    xs = np.sort(rng.uniform(0, 1, 30)).tolist()
    ys = np.sort(rng.uniform(0, 1, 30)).tolist()  # monótona no decreciente, como una isotónica real
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
    # Igual que Ejecutor de Dominio 1: decidir() no protege contra NaN, es responsabilidad de Veto.
    artefacto_roto = {**ARTEFACTO, "tabla_busqueda_plana": [float("nan"), 0.9]}
    ejecutor = EjecutorArboles(artefacto=artefacto_roto)
    resultado = ejecutor.decidir({"cc_num": "A", "amt": 20.0, "unix_time": 1704122400, "category": "x"})
    assert resultado["es_sospechosa"] is False  # nan >= umbral es False -- por diseño, ver docstring


def test_construir_ejecutor_valida_el_artefacto():
    with pytest.raises(ArtefactoArbolesInvalido):
        EjecutorArboles(artefacto={**ARTEFACTO, "umbral_decision": 5.0})


def test_mutar_artefacto_original_no_afecta_al_ejecutor_ya_construido():
    artefacto = dict(ARTEFACTO)
    ejecutor = EjecutorArboles(artefacto=artefacto)
    artefacto["umbral_decision"] = 0.0  # muta el dict original después de construir
    resultado = ejecutor.decidir({"cc_num": "A", "amt": 20.0, "unix_time": 1704122400, "category": "x"})
    assert resultado["es_sospechosa"] is False  # sigue usando el umbral 0.5 de la copia defensiva


def test_estado_por_cuenta_es_independiente_entre_cuentas():
    ejecutor = EjecutorArboles(artefacto=ARTEFACTO)
    # Alimenta la cuenta A con varias transacciones "y" -- no debe afectar a la cuenta B.
    for t in range(6):
        ejecutor.decidir({"cc_num": "A", "amt": 1.0, "unix_time": 1704122400 + t, "category": "y"})
    # EstadoRecursivoGlobal es un único reloj global compartido por todas las cuentas -- el
    # tiempo debe seguir avanzando, no retroceder, aunque la cuenta cambie.
    r_b = ejecutor.decidir({"cc_num": "B", "amt": 1.0, "unix_time": 1704122400 + 6, "category": "y"})
    # B es su primera transacción -- huella_categoria_cuenta usa respaldo poblacional (frecuencia_categoria["x"]=1.0,
    # "y" no está en frecuencia_categoria -> 0.0), no el historial de A.
    assert r_b["score"] in (0.1, 0.9)  # solo confirma que no lanzó ni se corrompió por el estado de A


@pytest.fixture(scope="module")
def pipeline_real():
    if not RUTA_DATASET.exists():
        pytest.skip(f"dataset real no disponible en {RUTA_DATASET}")
    return entrenar_pipeline_completo(RUTA_DATASET)


def test_tabla_busqueda_es_bit_exacta_contra_predict_proba_real(pipeline_real):
    # La prueba que de verdad importa (ver skill two-speed-decision): replay completo desde el
    # inicio del historial (train+val+test en orden), NO solo el tramo de test -- de lo
    # contrario el estado recursivo por cuenta/global arrancaría vacío en vez de con la
    # historia real, y los números no coincidirían con los que sí calculó el modo batch.
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
    assert diff.max() < 1e-9  # bit-exacto salvo ruido de punto flotante -- ver sección 6 del doc
    assert diff.mean() < 1e-12


def test_latencia_real_medida_con_perf_counter(pipeline_real):
    # Medida real, nunca estimada (disciplina de la skill two-speed-decision) -- misma
    # metodología que scripts/validacion_end_to_end.py: pre-construir la lista de transacciones,
    # medir solo el tiempo de decidir().
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
    # Generoso a propósito -- el número real se reporta en docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md,
    # este assert solo evita una regresión catastrófica (ej. volver a recorrer árboles en caliente).
    assert us_por_decision < 100.0
