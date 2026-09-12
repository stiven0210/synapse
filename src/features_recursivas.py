"""Features recursivas compartidas entre Calibrador y Ejecutor.

**Por qué un solo módulo, nunca duplicado**: si el Calibrador calcula estas
features de una forma (por lotes, sobre historia completa) y el Ejecutor las
calcula de otra (incremental, transacción por transacción), los pesos
calibrados dejan de ser válidos en producción — es el problema de
"train/serve skew" que en la práctica es la causa más común de que un
modelo funcione en el backtest y falle en producción. Este módulo es la
única fuente de verdad de la definición matemática de cada feature; el
Calibrador la usa en modo "por lotes" (`calcular_features_recursivas_batch`)
y el Ejecutor en modo "incremental" (`EstadoRecursivoGlobal`) — ambos
representan la misma fórmula, verificado en
`tests/test_features_recursivas.py` (el batch y el incremental deben
producir exactamente los mismos números sobre la misma secuencia).

Ambas features son **causales**: el valor en la fila `i` depende solo de
transacciones estrictamente anteriores a `i`, nunca de `i` misma — evita
fuga de información (la transacción no puede "verse a sí misma" en su
propio contexto reciente).
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAMBDA_EWMA_MONTO = 0.98  # peso del historial vs. la transacción más reciente
VENTANA_CONTEO_SEG = 60.0  # ventana de tiempo para el conteo de transacciones


class TiempoFueraDeOrden(Exception):
    """Un evento con `Time` anterior al último procesado corrompería la
    ventana en silencio (asume orden no decreciente) — se rechaza en vez de
    corromper el estado. Detectado en la auditoría: un evento tardío
    (llegada fuera de orden, común en streaming real) se quedaba atascado
    en medio de la ventana para siempre, nunca se purgaba, y contaminaba
    todos los conteos posteriores."""


def calcular_features_recursivas_batch(
    df: pd.DataFrame, lambda_ewma: float = LAMBDA_EWMA_MONTO, ventana_seg: float = VENTANA_CONTEO_SEG
) -> pd.DataFrame:
    """Agrega `monto_ewma_global` y `conteo_ventana_global` a una copia de
    `df` (debe venir ordenado por `Time` — responsabilidad del llamador,
    `cargar_dataset` ya lo garantiza). Vectorizado, para usarse una vez
    sobre todo el historial (modo Calibrador)."""
    df = df.copy()

    # EWMA causal: shift(1) excluye la transacción actual; la primera fila no tiene
    # historial previo, se rellena con su propio monto (referencia neutra, sin señal).
    ewma = df["Amount"].ewm(alpha=1 - lambda_ewma, adjust=False).mean().shift(1)
    df["monto_ewma_global"] = ewma.fillna(df["Amount"].iloc[0])

    tiempos = df["Time"].to_numpy()
    conteos = np.empty(len(df))
    inicio = 0
    for i in range(len(df)):
        while tiempos[inicio] <= tiempos[i] - ventana_seg:
            inicio += 1
        conteos[i] = i - inicio  # transacciones estrictamente antes de i, dentro de la ventana
    df["conteo_ventana_global"] = conteos

    return df


@dataclass
class EstadoRecursivoGlobal:
    """Estado compacto de tamaño fijo — el Ejecutor lo mantiene en memoria y
    lo actualiza en O(1) por transacción nueva, sin recalcular nada del
    historial. Es el "estadístico suficiente" para estas dos features.

    **Dos correcciones de la auditoría**:
    1. `tiempos_ventana` es un `deque`, no una lista — `list.pop(0)` es
       O(n) (desplaza todo lo demás), nada O(1) pese a lo que decía el
       docstring original; con tráfico real (cientos/miles de tx/seg) el
       costo crecería con el tamaño de la ventana, justo lo que este
       framework existe para evitar. `deque.popleft()` sí es O(1) real.
    2. Se rechaza cualquier evento con `Time` anterior al último procesado
       (`TiempoFueraDeOrden`) — un evento tardío (llegada fuera de orden,
       común en streaming real) antes se quedaba atascado en medio de la
       ventana para siempre, nunca se purgaba, y contaminaba todos los
       conteos posteriores en silencio.
    """
    lambda_ewma: float = LAMBDA_EWMA_MONTO
    ventana_seg: float = VENTANA_CONTEO_SEG
    monto_ewma: float | None = None
    tiempos_ventana: deque = field(default_factory=deque)  # timestamps dentro de la ventana activa
    ultimo_tiempo_visto: float | None = field(default=None, init=False)

    def leer_features(self, monto_actual: float, tiempo_actual: float) -> tuple:
        """Devuelve (monto_ewma_global, conteo_ventana_global) para la
        transacción actual, usando solo estado acumulado ANTES de esta
        transacción — misma semántica causal que la versión batch."""
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
        """Se llama DESPUÉS de leer_features/decidir — incorpora la
        transacción actual al estado para que la siguiente la vea como
        parte del historial."""
        if self.monto_ewma is None:
            self.monto_ewma = monto_actual
        else:
            self.monto_ewma = self.lambda_ewma * self.monto_ewma + (1 - self.lambda_ewma) * monto_actual
        self.tiempos_ventana.append(tiempo_actual)
        self.ultimo_tiempo_visto = tiempo_actual
