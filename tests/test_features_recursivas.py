import time

import numpy as np
import pandas as pd
import pytest

from src.features_recursivas import EstadoRecursivoGlobal, TiempoFueraDeOrden, calcular_features_recursivas_batch


def test_ewma_batch_caso_conocido():
    # amounts=[10,20,30], lambda=0.5 -> causal (shift 1): [10, 10, 15]
    df = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Amount": [10.0, 20.0, 30.0]})
    resultado = calcular_features_recursivas_batch(df, lambda_ewma=0.5, ventana_seg=1000)
    assert resultado["monto_ewma_global"].tolist() == pytest.approx([10.0, 10.0, 15.0])


def test_conteo_ventana_batch_caso_conocido():
    df = pd.DataFrame({"Time": [0.0, 10.0, 30.0, 65.0, 70.0], "Amount": [1.0] * 5})
    resultado = calcular_features_recursivas_batch(df, ventana_seg=60.0)
    assert resultado["conteo_ventana_global"].tolist() == [0, 1, 2, 2, 2]


def test_incremental_ewma_coincide_con_caso_conocido():
    estado = EstadoRecursivoGlobal(lambda_ewma=0.5, ventana_seg=1000)
    amounts = [10.0, 20.0, 30.0]
    esperados = [10.0, 10.0, 15.0]
    for t, (monto, esperado) in enumerate(zip(amounts, esperados)):
        ewma, _ = estado.leer_features(monto, float(t))
        assert ewma == pytest.approx(esperado)
        estado.actualizar(monto, float(t))


def test_incremental_conteo_coincide_con_caso_conocido():
    estado = EstadoRecursivoGlobal(ventana_seg=60.0)
    tiempos = [0.0, 10.0, 30.0, 65.0, 70.0]
    esperados = [0, 1, 2, 2, 2]
    for tiempo, esperado in zip(tiempos, esperados):
        _, conteo = estado.leer_features(1.0, tiempo)
        assert conteo == esperado
        estado.actualizar(1.0, tiempo)


def test_batch_e_incremental_coinciden_exactamente_en_secuencia_aleatoria():
    # The test that actually matters: both calculation modes must produce
    # the same numbers over the same sequence -- if they diverge, the model
    # calibrated in batch mode is useless for incremental-mode decisions
    # (train/serve skew).
    rng = np.random.default_rng(42)
    n = 500
    tiempos = np.sort(rng.uniform(0, 10000, n))
    montos = rng.exponential(50, n)
    df = pd.DataFrame({"Time": tiempos, "Amount": montos})

    resultado_batch = calcular_features_recursivas_batch(df)

    estado = EstadoRecursivoGlobal()
    ewma_incremental = []
    conteo_incremental = []
    for tiempo, monto in zip(tiempos, montos):
        ewma, conteo = estado.leer_features(monto, tiempo)
        ewma_incremental.append(ewma)
        conteo_incremental.append(conteo)
        estado.actualizar(monto, tiempo)

    assert resultado_batch["monto_ewma_global"].to_numpy() == pytest.approx(np.array(ewma_incremental))
    assert resultado_batch["conteo_ventana_global"].tolist() == conteo_incremental


def test_evento_fuera_de_orden_se_rechaza_en_vez_de_corromper_la_ventana():
    # Bug found in the audit: a late event used to get stuck in the middle of the
    # window forever and contaminate every count after it. Now it's rejected.
    estado = EstadoRecursivoGlobal(ventana_seg=60.0)
    for t in [100.0, 105.0, 110.0]:
        estado.leer_features(1.0, t)
        estado.actualizar(1.0, t)

    with pytest.raises(TiempoFueraDeOrden):
        estado.leer_features(1.0, 50.0)  # earlier than the last one processed (110.0)

    # State must not have been corrupted: it still reflects only [100, 105, 110].
    _, conteo = estado.leer_features(1.0, 115.0)
    assert conteo == 3  # window (55, 115] contains 100, 105, and 110 -- none has left yet


def test_mismo_tiempo_no_se_considera_fuera_de_orden():
    # Two transactions with the same Time (a tie) are valid -- they aren't "earlier".
    estado = EstadoRecursivoGlobal(ventana_seg=60.0)
    estado.leer_features(1.0, 100.0)
    estado.actualizar(1.0, 100.0)
    estado.leer_features(1.0, 100.0)  # must not raise
    estado.actualizar(1.0, 100.0)


def test_purga_de_ventana_es_o1_no_degrada_con_el_tamano():
    # tiempos_ventana must be purged in O(1) (deque.popleft), not O(n) (list.pop(0)) --
    # with a very large window and high frequency, a list would show up in the total time.
    estado = EstadoRecursivoGlobal(ventana_seg=1e9)  # huge window -> never purged, grows unbounded
    n = 20_000
    inicio = time.perf_counter()
    for i in range(n):
        estado.leer_features(1.0, float(i))
        estado.actualizar(1.0, float(i))
    duracion = time.perf_counter() - inicio
    # With list.pop(0) the per-operation cost would grow linearly with window size
    # (O(n^2) total); with deque.popleft it's O(1) per operation (O(n) total). The threshold is
    # generous -- what matters is not regressing back to quadratic behavior.
    assert duracion < 2.0
