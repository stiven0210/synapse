"""PER-ACCOUNT recursive features (Domain 2 — Sparkov) — same principle as
`features_recursivas.py` (Domain 1: a single source of truth shared
between Calibrator and Executor, to avoid train/serve skew), but indexed
by `cc_num` instead of global. `features_recursivas.py` is left untouched:
its features (`monto_ewma_global`, `conteo_ventana_global`) are still
computed as-is (its API is already generic in its argument names,
`leer_features(monto, tiempo)`/`actualizar(monto, tiempo)`); this module
only adds the two new per-account personalization features validated in
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` (section 3).

**Why the batch mode literally reuses the same state class as the
incremental mode** (`_EstadoCuenta`, used both inside
`calcular_features_recursivas_cuenta_batch` and inside
`EstadoRecursivoPorCuenta`): the category footprint isn't simply
vectorizable (it depends on which specific category EACH row carries, not
a uniform average), so instead of writing two parallel implementations
that could diverge, a single state machine is used in both modes — batch
simply walks it in order over the whole history. Even so,
`tests/test_features_recursivas_cuenta.py` verifies parity explicitly and
independently (driving `EstadoRecursivoPorCuenta` row by row and comparing
against batch), so as not to silently rely on both paths sharing code.
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
    "EstadoFrecuenciaCategoriaGlobal",
    "calcular_features_recursivas_cuenta_batch",
    "calcular_frecuencia_categoria_expandida_batch",
    "calcular_frecuencia_poblacional_categoria",
    "hora_utc",
]

K_HUELLA_CATEGORIA = 20  # size of the causal per-account window (by count, not time)
MIN_HISTORIAL_HUELLA = 5  # below this, the population fallback is used, never a made-up value


def hora_utc(unix_time) -> int:
    """The single source of truth for "hour of day" — used the same way in
    batch mode (vectorized, see `calcular_features_recursivas_cuenta_batch`)
    and in the Executor (row by row), in explicit UTC so both modes agree
    without depending on the timezone of whichever system runs each one."""
    return datetime.fromtimestamp(int(unix_time), tz=timezone.utc).hour


def calcular_frecuencia_poblacional_categoria(df: pd.DataFrame) -> dict:
    """Frequency of each `category` over `df` — meant to be computed ONCE
    over TRAIN in `calibrador_arboles.py` and saved into the artifact
    (`frecuencia_poblacional_categoria`), never recomputed on the hot path
    or contaminated with val/test data."""
    return df["category"].value_counts(normalize=True).to_dict()


def calcular_frecuencia_categoria_expandida_batch(df: pd.DataFrame, n_categorias: int) -> pd.DataFrame:
    """Adds `frecuencia_categoria_expandida` (section 15.1 of the Domain 2
    doc) -- unlike `huella_categoria_cuenta`, this one is **global** (the
    whole population, not per account) and **expanding**: for each row,
    `(conteo_categoria_hasta_ahora + 1) / (conteo_total_hasta_ahora +
    n_categorias)` -- Laplace `alpha=1`, causal (only counts strictly
    earlier rows). Lives in this module alongside
    `calcular_frecuencia_poblacional_categoria` for cohesion (both are
    global `category` statistics), not because it's "per account".

    Vectorized, no loop: `groupby(...).cumcount()` already counts
    STRICTLY EARLIER occurrences of the same category in the DataFrame's
    order (causal by construction), and the row's position (`np.arange`)
    is the total number of earlier rows. `df` must already be sorted by
    `unix_time` (the caller's responsibility, same as the rest of the
    module). `n_categorias` is learned ONCE from TRAIN (see
    `calibrador_arboles.py`) and never recomputed on the hot path -- same
    principle as `frecuencia_poblacional_categoria`."""
    df = df.copy()
    conteo_categoria_hasta_ahora = df.groupby("category").cumcount().to_numpy()
    conteo_total_hasta_ahora = np.arange(len(df))
    df["frecuencia_categoria_expandida"] = (conteo_categoria_hasta_ahora + 1) / (conteo_total_hasta_ahora + n_categorias)
    return df


class EstadoFrecuenciaCategoriaGlobal:
    """Incremental counterpart of `calcular_frecuencia_categoria_expandida_batch`
    -- GLOBAL state (not per account), O(1) per transaction: a
    `categoria -> count` dict + a running total. `n_categorias` comes from
    the artifact (learned on TRAIN), never recomputed here. Verified in
    `tests/test_features_recursivas_cuenta.py` to match batch mode exactly
    over the same sequence."""

    __slots__ = ("conteos", "total", "n_categorias")

    def __init__(self, n_categorias: int):
        self.conteos: dict = {}
        self.total: int = 0
        self.n_categorias = n_categorias

    def leer(self, categoria) -> float:
        conteo_categoria = self.conteos.get(categoria, 0)
        return (conteo_categoria + 1) / (self.total + self.n_categorias)

    def actualizar(self, categoria) -> None:
        self.conteos[categoria] = self.conteos.get(categoria, 0) + 1
        self.total += 1


class _EstadoCuenta:
    """Compact state for a single account -- fixed size (`deque(maxlen=k)`),
    reused without modification by both batch mode and
    `EstadoRecursivoPorCuenta` (see the module docstring)."""

    __slots__ = ("monto_ewma", "categorias", "ultimo_tiempo_visto")

    def __init__(self, k_huella: int):
        self.monto_ewma: float | None = None
        self.categorias: deque = deque(maxlen=k_huella)
        self.ultimo_tiempo_visto: float | None = None

    def leer(self, monto: float, categoria, frecuencia_categoria: dict, min_historial: int) -> tuple:
        ewma = monto if self.monto_ewma is None else self.monto_ewma
        if len(self.categorias) < min_historial:
            # Population fallback: with not enough history for THIS account, there's nothing
            # causal to look at yet -- using the frequency learned on TRAIN avoids making up
            # an arbitrary fraction (e.g. 0.0 or 1.0) for an account's first few transactions.
            # A category never seen in TRAIN -> 0.0 (no signal of "unusual for the account",
            # which is exactly what this feature measures).
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
    """Adds `monto_ewma_cuenta` and `huella_categoria_cuenta` to a copy of
    `df` (must already be sorted by `unix_time` -- the caller's
    responsibility, same as Domain 1's `calcular_features_recursivas_batch`).
    Walks the rows in order, maintaining one `_EstadoCuenta` per `cc_num`
    -- not vectorized (the footprint depends on each row's specific
    category), but this only runs at calibration time, never on the hot
    path."""
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
    """Incremental counterpart of `calcular_features_recursivas_cuenta_batch`,
    for the Executor's use -- O(1) per transaction (a single dict
    access/update + one fixed-size `deque`), never rescans the full
    history. `frecuencia_categoria` comes from the artifact (learned on
    TRAIN by `calibrador_arboles.py`), never recomputed here.

    Not thread-safe, for the same reason documented in `ejecutor.py`."""

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
        # leer_features() is always called first (same contract as EstadoRecursivoGlobal) --
        # by this point the account already exists in _cuentas.
        self._cuentas[cc_num].actualizar(monto, categoria, tiempo, self.lambda_ewma)
