"""Tree executor (Domain 2, fast layer) — evaluates the artifact from
`artefacto_arboles.py` via the v3 lookup table (the model's real
thresholds, exact by construction — see that module's docstring), never
walking trees or loading sklearn on the hot path. Measured in
`tests/test_ejecutor_arboles.py` under real volume.

**Same low-level warnings as `ejecutor.py`** (Domain 1): `decidir()` can
return a `score` technically out of range if the artifact is corrupt in a
way that passed shape validation but not content validation; in Python
`nan >= umbral` is `False`, so using `resultado["es_sospechosa"]` directly
would treat that as "not suspicious". The correct way to use this is
through `ciclo_arboles.py::CicloDecisionArboles`, which forces the call
through `veto.py` (reused without modification) and through the Bridge.

**Not thread-safe**, for the same reason as `Ejecutor`: `EstadoRecursivoGlobal`
and `EstadoRecursivoPorCuenta` have no synchronization and assume a single
sequential stream of non-decreasing-in-time events.
"""
import bisect
import copy
import math
from dataclasses import dataclass, field

from src.artefacto_arboles import ArtefactoArbolesInvalido, validar_artefacto_arboles
from src.features_recursivas import EstadoRecursivoGlobal
from src.features_recursivas_cuenta import EstadoFrecuenciaCategoriaGlobal, EstadoRecursivoPorCuenta, hora_utc

__all__ = ["ArtefactoArbolesInvalido", "EjecutorArboles", "validar_artefacto_arboles"]


def _interpolar_isotonica(x: float, xs: list, ys: list) -> float:
    """Reproduces `IsotonicRegression.predict()` with `out_of_bounds="clip"`
    in pure Python: out of range clamps to the extreme, in range
    interpolates linearly between the two neighboring breakpoints --
    verified identical to `np.interp` (which itself reproduces sklearn
    exactly, see `src/calibrador_arboles.py`), with no numpy dependency on
    the hot path.

    **NaN is let through without raising**, same as Domain 1's
    `Ejecutor._sigmoide` (`math.exp(nan)` doesn't raise, it produces a
    clean `nan`) -- a corrupt table score (NaN) must reach `veto.py` as
    NaN, since that's what distinguishes it (ADR_002, invariant 3), rather
    than blowing up earlier on a `bisect` that can't compare NaN."""
    if isinstance(x, float) and math.isnan(x):
        return float("nan")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1  # xs[i] <= x < xs[i+1] (xs has no duplicates by construction)
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class EjecutorArboles:
    artefacto: dict
    estado_global: EstadoRecursivoGlobal = field(default_factory=EstadoRecursivoGlobal)
    estado_cuenta: EstadoRecursivoPorCuenta = None  # initialized in __post_init__ with the artifact's frequency
    estado_frecuencia_categoria: EstadoFrecuenciaCategoriaGlobal = None  # same, with the artifact's n_categorias

    def __post_init__(self) -> None:
        validar_artefacto_arboles(self.artefacto)
        # Defensive copy -- same audit finding as Domain 1's Executor: without this,
        # mutating the original dict after construction would invalidate the "validated once" guarantee.
        self.artefacto = copy.deepcopy(self.artefacto)

        if self.estado_cuenta is None:
            self.estado_cuenta = EstadoRecursivoPorCuenta(frecuencia_categoria=self.artefacto["frecuencia_poblacional_categoria"])
        if self.estado_frecuencia_categoria is None:
            # n_categorias = TRAIN's known-category vocabulary -- same criterion as
            # `calibrador_arboles.construir_features` (see that docstring), not a new artifact field.
            n_categorias = len(self.artefacto["frecuencia_poblacional_categoria"])
            self.estado_frecuencia_categoria = EstadoFrecuenciaCategoriaGlobal(n_categorias=n_categorias)

        # Precomputed ONCE at construction, not on every decidir(): strides to index the
        # flat table in O(1) with no numpy reshape on the hot path (same spirit as the
        # doc's "generated code" -- eliminating repeated overhead, not changing the algorithm).
        self._features = self.artefacto["features"]
        self._umbrales = self.artefacto["umbrales_por_feature"]
        forma = self.artefacto["tabla_busqueda_forma"]
        self._tabla = self.artefacto["tabla_busqueda_plana"]
        strides = [1] * len(forma)
        for i in range(len(forma) - 2, -1, -1):
            strides[i] = strides[i + 1] * forma[i + 1]
        self._strides = strides
        self._xs_isotonica = self.artefacto["calibracion_isotonica"]["x"]
        self._ys_isotonica = self.artefacto["calibracion_isotonica"]["y"]

    def decidir(self, transaccion: dict) -> dict:
        """`transaccion` carries at least `cc_num`, `amt`, `unix_time`,
        `category` (Sparkov schema, see `docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`).
        O(1) -- no operation here walks the full history or the trees. See
        the module warning: the result must NOT be used directly, it must
        go through `veto.evaluar()` via `CicloDecisionArboles`."""
        cc_num = transaccion["cc_num"]
        monto = transaccion["amt"]
        tiempo = transaccion["unix_time"]
        categoria = transaccion["category"]

        _, conteo_ventana_global = self.estado_global.leer_features(monto, tiempo)
        monto_ewma_cuenta, huella_categoria_cuenta = self.estado_cuenta.leer_features(cc_num, monto, categoria, tiempo)
        frecuencia_categoria_expandida = self.estado_frecuencia_categoria.leer(categoria)
        hora = hora_utc(tiempo)

        valores = {
            "amt": monto,
            "hora": hora,
            "conteo_ventana_global": conteo_ventana_global,
            "monto_ewma_cuenta": monto_ewma_cuenta,
            "huella_categoria_cuenta": huella_categoria_cuenta,
            "frecuencia_categoria_expandida": frecuencia_categoria_expandida,
        }

        indice_plano = 0
        for nombre, stride in zip(self._features, self._strides):
            umbrales_feature = self._umbrales[nombre]
            bin_idx = bisect.bisect_left(umbrales_feature, valores[nombre])
            indice_plano += bin_idx * stride

        score_crudo = self._tabla[indice_plano]
        score = _interpolar_isotonica(score_crudo, self._xs_isotonica, self._ys_isotonica)

        # State is updated AFTER reading/deciding -- same causal semantics as Domain 1:
        # the current transaction never sees itself in its own recent context.
        self.estado_global.actualizar(monto, tiempo)
        self.estado_cuenta.actualizar(cc_num, monto, categoria, tiempo)
        self.estado_frecuencia_categoria.actualizar(categoria)

        # Same as Domain 1's Executor: if score is NaN, `nan >= umbral` gives False in Python --
        # deliberate, not guarded against here. This is exactly why `veto.py` (which DOES
        # distinguish NaN) exists; see the module warning and `ciclo_arboles.py::CicloDecisionArboles`.
        return {"score": score, "es_sospechosa": score >= self.artefacto["umbral_decision"]}
