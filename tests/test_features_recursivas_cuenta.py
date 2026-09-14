import numpy as np
import pandas as pd
import pytest

from src.features_recursivas import TiempoFueraDeOrden
from src.features_recursivas_cuenta import (
    EstadoFrecuenciaCategoriaGlobal,
    EstadoRecursivoPorCuenta,
    calcular_features_recursivas_cuenta_batch,
    calcular_frecuencia_categoria_expandida_batch,
    calcular_frecuencia_poblacional_categoria,
    hora_utc,
)


def test_hora_utc_caso_conocido():
    # 2024-01-01 15:30:00 UTC
    assert hora_utc(1704122400) == 15


def test_frecuencia_poblacional_categoria_caso_conocido():
    df = pd.DataFrame({"category": ["grocery", "grocery", "gas", "gas", "gas", "gas"]})
    freq = calcular_frecuencia_poblacional_categoria(df)
    assert freq["gas"] == pytest.approx(4 / 6)
    assert freq["grocery"] == pytest.approx(2 / 6)


def test_ewma_cuenta_batch_caso_conocido_dos_cuentas_intercaladas():
    # cc=A: montos [10,20,30] lambda=0.5 -> causal [10,10,15]; cc=B intercalada no debe afectar A.
    df = pd.DataFrame({
        "cc_num": ["A", "B", "A", "A", "B"],
        "amt": [10.0, 100.0, 20.0, 30.0, 200.0],
        "category": ["x"] * 5,
        "unix_time": [0, 0, 1, 2, 1],
    })
    resultado = calcular_features_recursivas_cuenta_batch(df, frecuencia_categoria={"x": 1.0}, lambda_ewma=0.5)
    ewma_a = resultado[resultado["cc_num"] == "A"]["monto_ewma_cuenta"].tolist()
    assert ewma_a == pytest.approx([10.0, 10.0, 15.0])


def test_huella_categoria_cuenta_usa_respaldo_poblacional_bajo_min_historial():
    # Con min_historial=5 y solo 2 transacciones previas, se usa la frecuencia poblacional,
    # no una fracción calculada sobre las 2 disponibles.
    df = pd.DataFrame({
        "cc_num": ["A"] * 3,
        "amt": [1.0, 1.0, 1.0],
        "category": ["grocery", "grocery", "gas"],
        "unix_time": [0, 1, 2],
    })
    frecuencia = {"grocery": 0.3, "gas": 0.7}
    resultado = calcular_features_recursivas_cuenta_batch(df, frecuencia_categoria=frecuencia)
    assert resultado["huella_categoria_cuenta"].tolist() == pytest.approx([0.3, 0.3, 0.7])


def test_huella_categoria_cuenta_caso_conocido_con_historial_suficiente():
    # Cuenta A: 5 transacciones previas en "grocery","grocery","gas","grocery","grocery" (4/5
    # grocery), la 6ta transacción es "grocery" -> huella = 4/5 = 0.8.
    categorias_previas = ["grocery", "grocery", "gas", "grocery", "grocery"]
    df = pd.DataFrame({
        "cc_num": ["A"] * 6,
        "amt": [1.0] * 6,
        "category": categorias_previas + ["grocery"],
        "unix_time": list(range(6)),
    })
    resultado = calcular_features_recursivas_cuenta_batch(df, frecuencia_categoria={"grocery": 0.5, "gas": 0.5}, min_historial=5)
    assert resultado["huella_categoria_cuenta"].iloc[5] == pytest.approx(0.8)


def test_huella_categoria_cuenta_ventana_desliza_con_maxlen_k():
    # Con k_huella=2, solo las últimas 2 transacciones cuentan, no todo el historial.
    df = pd.DataFrame({
        "cc_num": ["A"] * 4,
        "amt": [1.0] * 4,
        "category": ["gas", "gas", "grocery", "grocery"],
        "unix_time": list(range(4)),
    })
    resultado = calcular_features_recursivas_cuenta_batch(
        df, frecuencia_categoria={"gas": 0.5, "grocery": 0.5}, k_huella=2, min_historial=2
    )
    # fila 3 (índice 3, categoria "grocery"): ventana de las 2 previas = ["gas","grocery"] -> 1/2 coincide
    assert resultado["huella_categoria_cuenta"].iloc[3] == pytest.approx(0.5)


def test_evento_fuera_de_orden_por_cuenta_se_rechaza():
    df = pd.DataFrame({
        "cc_num": ["A", "A"],
        "amt": [1.0, 1.0],
        "category": ["x", "x"],
        "unix_time": [10, 5],  # fuera de orden para la cuenta A
    })
    with pytest.raises(TiempoFueraDeOrden):
        calcular_features_recursivas_cuenta_batch(df, frecuencia_categoria={"x": 1.0})


def test_incremental_estado_por_cuenta_rechaza_fuera_de_orden():
    estado = EstadoRecursivoPorCuenta(frecuencia_categoria={"x": 1.0})
    estado.leer_features("A", 1.0, "x", 10.0)
    estado.actualizar("A", 1.0, "x", 10.0)
    with pytest.raises(TiempoFueraDeOrden):
        estado.leer_features("A", 1.0, "x", 5.0)


def test_batch_e_incremental_coinciden_exactamente_en_secuencia_aleatoria_multi_cuenta():
    # La prueba que de verdad importa (paridad train/serve): mismos números en batch e
    # incremental sobre la misma secuencia multi-cuenta con categorías mezcladas.
    rng = np.random.default_rng(7)
    n = 800
    cuentas = rng.choice(["A", "B", "C", "D"], size=n)
    categorias_posibles = ["grocery", "gas", "entertainment", "misc"]
    categorias = rng.choice(categorias_posibles, size=n)
    montos = rng.exponential(50, n)

    # unix_time creciente globalmente garantiza no-decreciente por cuenta también.
    tiempos = np.arange(n, dtype=float)

    df = pd.DataFrame({"cc_num": cuentas, "amt": montos, "category": categorias, "unix_time": tiempos})
    frecuencia = calcular_frecuencia_poblacional_categoria(df)

    resultado_batch = calcular_features_recursivas_cuenta_batch(df, frecuencia_categoria=frecuencia)

    estado = EstadoRecursivoPorCuenta(frecuencia_categoria=frecuencia)
    ewma_incremental = []
    huella_incremental = []
    for cc, monto, categoria, tiempo in zip(cuentas, montos, categorias, tiempos):
        ewma, huella = estado.leer_features(cc, monto, categoria, tiempo)
        ewma_incremental.append(ewma)
        huella_incremental.append(huella)
        estado.actualizar(cc, monto, categoria, tiempo)

    assert resultado_batch["monto_ewma_cuenta"].to_numpy() == pytest.approx(np.array(ewma_incremental))
    assert resultado_batch["huella_categoria_cuenta"].to_numpy() == pytest.approx(np.array(huella_incremental))


def test_frecuencia_categoria_expandida_caso_conocido_a_mano():
    # n_categorias=2 ("grocery","gas"). Secuencia: grocery,grocery,gas,grocery.
    # fila 0 (grocery): (0+1)/(0+2) = 0.5
    # fila 1 (grocery): (1+1)/(1+2) = 2/3
    # fila 2 (gas):     (0+1)/(2+2) = 0.25
    # fila 3 (grocery): (2+1)/(3+2) = 0.6
    df = pd.DataFrame({"category": ["grocery", "grocery", "gas", "grocery"]})
    resultado = calcular_frecuencia_categoria_expandida_batch(df, n_categorias=2)
    assert resultado["frecuencia_categoria_expandida"].tolist() == pytest.approx([0.5, 2 / 3, 0.25, 0.6])


def test_estado_frecuencia_categoria_global_incremental_caso_conocido():
    # Misma secuencia y mismos valores esperados que el test batch de arriba -- verifica
    # el estado incremental de forma directa e independiente, no solo contra el batch.
    estado = EstadoFrecuenciaCategoriaGlobal(n_categorias=2)
    valores = []
    for categoria in ["grocery", "grocery", "gas", "grocery"]:
        valores.append(estado.leer(categoria))
        estado.actualizar(categoria)
    assert valores == pytest.approx([0.5, 2 / 3, 0.25, 0.6])


def test_frecuencia_categoria_expandida_batch_e_incremental_coinciden_exactamente():
    # Paridad train/serve para la feature nueva (sección 15.1/16 del doc) -- misma disciplina
    # que el resto del módulo: batch e incremental deben coincidir exacto, no aproximado.
    rng = np.random.default_rng(11)
    n = 1000
    categorias_posibles = ["grocery", "gas", "entertainment", "misc", "travel"]
    categorias = rng.choice(categorias_posibles, size=n)
    df = pd.DataFrame({"category": categorias})

    resultado_batch = calcular_frecuencia_categoria_expandida_batch(df, n_categorias=len(categorias_posibles))

    estado = EstadoFrecuenciaCategoriaGlobal(n_categorias=len(categorias_posibles))
    valores_incrementales = []
    for categoria in categorias:
        valores_incrementales.append(estado.leer(categoria))
        estado.actualizar(categoria)

    assert resultado_batch["frecuencia_categoria_expandida"].to_numpy() == pytest.approx(np.array(valores_incrementales))
