"""Recursive features shared between Calibrator and Executor.

**Why a single module, never duplicated**: if the Calibrator computes
these features one way (batch, over full history) and the Executor
computes them another (incremental, transaction by transaction), the
calibrated weights stop being valid in production — this is the
"train/serve skew" problem, which in practice is the most common reason a
model works in backtesting and fails in production. This module is the
single source of truth for each feature's mathematical definition; the
Calibrator uses it in "batch" mode (`calcular_features_recursivas_batch`)
and the Executor in "incremental" mode (`EstadoRecursivoGlobal`) — both
represent the same formula, verified in
`tests/test_features_recursivas.py` (batch and incremental must produce
exactly the same numbers over the same sequence).

Both features are **causal**: the value in row `i` depends only on
transactions strictly before `i`, never on `i` itself — this prevents
information leakage (a transaction can't "see itself" in its own recent
context).
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAMBDA_EWMA_MONTO = 0.98  # weight of history vs. the most recent transaction
VENTANA_CONTEO_SEG = 60.0  # time window for the transaction count


class TiempoFueraDeOrden(Exception):
    """An event with a `Time` earlier than the last one processed would
    silently corrupt the window (it assumes non-decreasing order) — it's
    rejected instead of corrupting the state. Found in the audit: a late
    event (out-of-order arrival, common in real streaming) used to get
    stuck in the middle of the window forever, never purged, silently
    contaminating every count after it."""


def calcular_features_recursivas_batch(
    df: pd.DataFrame, lambda_ewma: float = LAMBDA_EWMA_MONTO, ventana_seg: float = VENTANA_CONTEO_SEG
) -> pd.DataFrame:
    """Adds `monto_ewma_global` and `conteo_ventana_global` to a copy of
    `df` (must already be sorted by `Time` — the caller's responsibility;
    `cargar_dataset` already guarantees it). Vectorized, for use once over
    the whole history (Calibrator mode)."""
    df = df.copy()

    # Causal EWMA: shift(1) excludes the current transaction; the first row has no
    # prior history, filled with its own amount (a neutral reference, no signal).
    ewma = df["Amount"].ewm(alpha=1 - lambda_ewma, adjust=False).mean().shift(1)
    df["monto_ewma_global"] = ewma.fillna(df["Amount"].iloc[0])

    tiempos = df["Time"].to_numpy()
    conteos = np.empty(len(df))
    inicio = 0
    for i in range(len(df)):
        while tiempos[inicio] <= tiempos[i] - ventana_seg:
            inicio += 1
        conteos[i] = i - inicio  # transactions strictly before i, within the window
    df["conteo_ventana_global"] = conteos

    return df


@dataclass
class EstadoRecursivoGlobal:
    """Fixed-size compact state — the Executor keeps it in memory and
    updates it in O(1) per new transaction, with no recomputation over the
    history. It's the "sufficient statistic" for these two features.

    **Two audit fixes**:
    1. `tiempos_ventana` is a `deque`, not a list — `list.pop(0)` is O(n)
       (it shifts everything else), not O(1) despite what the original
       docstring claimed; under real traffic (hundreds/thousands of
       tx/sec) the cost would grow with the window size, exactly what this
       framework exists to avoid. `deque.popleft()` is genuinely O(1).
    2. Any event with a `Time` earlier than the last one processed is
       rejected (`TiempoFueraDeOrden`) — a late event (out-of-order
       arrival, common in real streaming) used to get stuck in the middle
       of the window forever, never purged, silently contaminating every
       count after it.
    """
    lambda_ewma: float = LAMBDA_EWMA_MONTO
    ventana_seg: float = VENTANA_CONTEO_SEG
    monto_ewma: float | None = None
    tiempos_ventana: deque = field(default_factory=deque)  # timestamps within the active window
    ultimo_tiempo_visto: float | None = field(default=None, init=False)

    def leer_features(self, monto_actual: float, tiempo_actual: float) -> tuple:
        """Returns (monto_ewma_global, conteo_ventana_global) for the
        current transaction, using only state accumulated BEFORE this
        transaction — same causal semantics as the batch version."""
        if self.ultimo_tiempo_visto is not None and tiempo_actual < self.ultimo_tiempo_visto:
            raise TiempoFueraDeOrden(
                f"evento en Time={tiempo_actual} es anterior al último procesado "
                f"(Time={self.ultimo_tiempo_visto}) — rechazado para no corromper la ventana"
            )
        monto_ewma = monto_actual if self.monto_ewma is None else self.monto_ewma
        while self.tiempos_ventana and self.tiempos_ventana[0] <= tiempo_actual - self.ventana_seg:
            self.tiempos_ventana.popleft()
        conteo = len(self.tiempos_ventana)
        return monto_ewma, conteo

    def actualizar(self, monto_actual: float, tiempo_actual: float) -> None:
        """Called AFTER leer_features/decidir — folds the current
        transaction into the state so the next one sees it as part of the
        history."""
        if self.monto_ewma is None:
            self.monto_ewma = monto_actual
        else:
            self.monto_ewma = self.lambda_ewma * self.monto_ewma + (1 - self.lambda_ewma) * monto_actual
        self.tiempos_ventana.append(tiempo_actual)
        self.ultimo_tiempo_visto = tiempo_actual
