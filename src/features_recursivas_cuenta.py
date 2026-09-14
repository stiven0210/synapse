"""Features recursivas POR CUENTA (Dominio 2 — Sparkov) — mismo principio de
`features_recursivas.py` (Dominio 1: fuente única de verdad compartida entre
Calibrador y Ejecutor, para evitar train/serve skew), pero indexadas por
`cc_num` en vez de globales. No se toca `features_recursivas.py`: sus
features (`monto_ewma_global`, `conteo_ventana_global`) se siguen calculando
tal cual (su API ya es genérica en nombres de argumento, `leer_features(monto,
tiempo)`/`actualizar(monto, tiempo)`), este módulo solo agrega las dos
features nuevas de personalización por cuenta validadas en
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` (sección 3).

**Por qué el batch reutiliza literalmente la misma clase de estado que el
modo incremental** (`_EstadoCuenta`, usada tanto dentro de
`calcular_features_recursivas_cuenta_batch` como dentro de
`EstadoRecursivoPorCuenta`): la huella de categoría no es vectorizable de
forma simple (depende de qué categoría específica trae CADA fila, no un
promedio uniforme), así que en vez de escribir dos implementaciones
paralelas que podrían divergir, se usa una sola máquina de estado en ambos
modos — el batch simplemente la recorre en orden sobre todo el historial.
Aun así, `tests/test_features_recursivas_cuenta.py` verifica la paridad de
forma explícita e independiente (impulsando `EstadoRecursivoPorCuenta` fila
por fila y comparando contra el batch), para no depender silenciosamente de
que ambos caminos compartan código.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features_recursivas import LAMBDA_EWMA_MONTO, TiempoFueraDeOrden

__all__ = [
    "TiempoFueraDeOrden",
    "EstadoRecursivoPorCuenta",
    "calcular_features_recursivas_cuenta_batch",
    "calcular_frecuencia_poblacional_categoria",
    "hora_utc",
]

K_HUELLA_CATEGORIA = 20  # tamaño de la ventana causal por cuenta (por conteo, no por tiempo)
MIN_HISTORIAL_HUELLA = 5  # por debajo de esto se usa el respaldo poblacional, no un valor inventado


def hora_utc(unix_time) -> int:
    """Única fuente de verdad de "hora del día" — usada igual en modo batch
    (vectorizado, ver `calcular_features_recursivas_cuenta_batch`) y en el
    Ejecutor (fila por fila), en UTC explícito para que ambos modos
    coincidan sin depender de la zona horaria del sistema donde corra cada
    uno."""
    return datetime.fromtimestamp(int(unix_time), tz=timezone.utc).hour


def calcular_frecuencia_poblacional_categoria(df: pd.DataFrame) -> dict:
    """Frecuencia de cada `category` sobre `df` — pensado para calcularse
    UNA VEZ sobre TRAIN en `calibrador_arboles.py` y guardarse en el
    artefacto (`frecuencia_poblacional_categoria`), nunca recalcularse en
    caliente ni verse contaminado con datos de val/test."""
    return df["category"].value_counts(normalize=True).to_dict()


class _EstadoCuenta:
    """Estado compacto de una sola cuenta -- tamaño fijo (`deque(maxlen=k)`),
    reutilizado sin modificación tanto por el modo batch como por
    `EstadoRecursivoPorCuenta` (ver docstring del módulo)."""

    __slots__ = ("monto_ewma", "categorias", "ultimo_tiempo_visto")

    def __init__(self, k_huella: int):
        self.monto_ewma: float | None = None
        self.categorias: deque = deque(maxlen=k_huella)
        self.ultimo_tiempo_visto: float | None = None

    def leer(self, monto: float, categoria, frecuencia_categoria: dict, min_historial: int) -> tuple:
        ewma = monto if self.monto_ewma is None else self.monto_ewma
        if len(self.categorias) < min_historial:
            # Respaldo poblacional: sin historial suficiente de ESTA cuenta, no hay nada causal
            # que mirar todavía -- usar la frecuencia aprendida en TRAIN evita inventar una
            # fracción arbitraria (ej. 0.0 o 1.0) para las primeras transacciones de cada cuenta.
            # Categoría nunca vista en TRAIN -> 0.0 (sin señal de "inusual para la cuenta", que es
            # justamente lo que esta feature mide).
            huella = frecuencia_categoria.get(categoria, 0.0)
        else:
            coincidencias = sum(1 for c in self.categorias if c == categoria)
            huella = coincidencias / len(self.categorias)
        return ewma, huella

    def actualizar(self, monto: float, categoria, tiempo, lambda_ewma: float) -> None:
        if self.monto_ewma is None:
            self.monto_ewma = monto
        else:
            self.monto_ewma = lambda_ewma * self.monto_ewma + (1 - lambda_ewma) * monto
        self.categorias.append(categoria)
        self.ultimo_tiempo_visto = tiempo


def calcular_features_recursivas_cuenta_batch(
    df: pd.DataFrame,
    frecuencia_categoria: dict,
    lambda_ewma: float = LAMBDA_EWMA_MONTO,
    k_huella: int = K_HUELLA_CATEGORIA,
    min_historial: int = MIN_HISTORIAL_HUELLA,
) -> pd.DataFrame:
    """Agrega `monto_ewma_cuenta` y `huella_categoria_cuenta` a una copia de
    `df` (debe venir ordenado por `unix_time` -- responsabilidad del
    llamador, igual que `calcular_features_recursivas_batch` de Dominio 1).
    Recorre las filas en orden manteniendo un `_EstadoCuenta` por `cc_num`
    -- no vectorizado (la huella depende de la categoría específica de cada
    fila), pero solo corre en tiempo de calibración, nunca en el camino
    caliente."""
    df = df.copy()
    n = len(df)
    cc_nums = df["cc_num"].to_numpy()
    amts = df["amt"].to_numpy(dtype=float)
    categorias = df["category"].to_numpy()
    tiempos = df["unix_time"].to_numpy()

    monto_ewma_cuenta = np.empty(n, dtype=float)
    huella_categoria_cuenta = np.empty(n, dtype=float)
    estados: dict = {}

    for i in range(n):
        cc = cc_nums[i]
        estado = estados.get(cc)
        if estado is None:
            estado = _EstadoCuenta(k_huella)
            estados[cc] = estado
        if estado.ultimo_tiempo_visto is not None and tiempos[i] < estado.ultimo_tiempo_visto:
            raise TiempoFueraDeOrden(
                f"cuenta {cc}: evento en unix_time={tiempos[i]} es anterior al último "
                f"procesado de esa cuenta (unix_time={estado.ultimo_tiempo_visto})"
            )
        ewma, huella = estado.leer(amts[i], categorias[i], frecuencia_categoria, min_historial)
        monto_ewma_cuenta[i] = ewma
        huella_categoria_cuenta[i] = huella
        estado.actualizar(amts[i], categorias[i], tiempos[i], lambda_ewma)

    df["hora"] = pd.to_datetime(df["unix_time"], unit="s", utc=True).dt.hour
    df["monto_ewma_cuenta"] = monto_ewma_cuenta
    df["huella_categoria_cuenta"] = huella_categoria_cuenta
    return df


@dataclass
class EstadoRecursivoPorCuenta:
    """Contraparte incremental de `calcular_features_recursivas_cuenta_batch`,
    para uso del Ejecutor -- O(1) por transacción (un solo acceso/actualización
    de dict + una `deque` de tamaño fijo), nunca reescanea el historial
    completo. `frecuencia_categoria` viene del artefacto (aprendida en TRAIN
    por `calibrador_arboles.py`), no se recalcula aquí.

    No es thread-safe, por el mismo motivo documentado en `ejecutor.py`."""

    frecuencia_categoria: dict
    lambda_ewma: float = LAMBDA_EWMA_MONTO
    k_huella: int = K_HUELLA_CATEGORIA
    min_historial: int = MIN_HISTORIAL_HUELLA
    _cuentas: dict = field(default_factory=dict, init=False, repr=False)

    def leer_features(self, cc_num, monto: float, categoria, tiempo) -> tuple:
        estado = self._cuentas.get(cc_num)
        if estado is None:
            estado = _EstadoCuenta(self.k_huella)
            self._cuentas[cc_num] = estado
        if estado.ultimo_tiempo_visto is not None and tiempo < estado.ultimo_tiempo_visto:
            raise TiempoFueraDeOrden(
                f"cuenta {cc_num}: evento en tiempo={tiempo} es anterior al último "
                f"procesado de esa cuenta (tiempo={estado.ultimo_tiempo_visto})"
            )
        return estado.leer(monto, categoria, self.frecuencia_categoria, self.min_historial)

    def actualizar(self, cc_num, monto: float, categoria, tiempo) -> None:
        # leer_features() siempre se llama antes (mismo contrato que EstadoRecursivoGlobal) --
        # a esta altura la cuenta ya existe en _cuentas.
        self._cuentas[cc_num].actualizar(monto, categoria, tiempo, self.lambda_ewma)
