"""Ejecutor de árboles (Dominio 2, capa rápida) — evalúa el artefacto de
`artefacto_arboles.py` vía la tabla de búsqueda v3 (umbrales reales del
modelo, exacta por construcción — ver docstring de ese módulo), nunca
recorriendo árboles ni cargando sklearn en el camino caliente. Medido en
`tests/test_ejecutor_arboles.py` sobre volumen real.

**Mismas advertencias de bajo nivel que `ejecutor.py`** (Dominio 1):
`decidir()` puede devolver un `score` técnicamente fuera de rango si el
artefacto está corrupto de una forma que pasó la validación de forma pero no
de contenido; en Python `nan >= umbral` es `False`, así que usar
`resultado["es_sospechosa"]` directo trataría eso como "no sospechosa". La
forma correcta de usar esto es a través de `ciclo_arboles.py::CicloDecisionArboles`,
que fuerza el paso por `veto.py` (reusado sin modificar) y por el Puente.

**No es thread-safe**, por el mismo motivo que `Ejecutor`: `EstadoRecursivoGlobal`
y `EstadoRecursivoPorCuenta` no tienen sincronización y asumen un único
stream secuencial de eventos no decrecientes en el tiempo.
"""
import bisect
import copy
import math
from dataclasses import dataclass, field

from src.artefacto_arboles import ArtefactoArbolesInvalido, validar_artefacto_arboles
from src.features_recursivas import EstadoRecursivoGlobal
from src.features_recursivas_cuenta import EstadoRecursivoPorCuenta, hora_utc

__all__ = ["ArtefactoArbolesInvalido", "EjecutorArboles", "validar_artefacto_arboles"]


def _interpolar_isotonica(x: float, xs: list, ys: list) -> float:
    """Reproduce `IsotonicRegression.predict()` con `out_of_bounds="clip"`
    en Python puro: fuera de rango se clampa al extremo, dentro de rango se
    interpola linealmente entre los dos breakpoints vecinos -- verificado
    idéntico a `np.interp` (que a su vez reproduce sklearn exacto, ver
    `src/calibrador_arboles.py`), sin depender de numpy en el camino caliente.

    **NaN se deja pasar sin lanzar**, igual que `Ejecutor._sigmoide` de
    Dominio 1 (`math.exp(nan)` no lanza, produce `nan` limpio) -- un score
    de tabla corrupto (NaN) debe llegar como NaN hasta `veto.py`, que es
    quien lo distingue (ADR_002, invariante 3), no reventar antes por un
    `bisect` que no sabe comparar NaN."""
    if isinstance(x, float) and math.isnan(x):
        return float("nan")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1  # xs[i] <= x < xs[i+1] (xs no tiene duplicados por construcción)
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class EjecutorArboles:
    artefacto: dict
    estado_global: EstadoRecursivoGlobal = field(default_factory=EstadoRecursivoGlobal)
    estado_cuenta: EstadoRecursivoPorCuenta = None  # se inicializa en __post_init__ con la frecuencia del artefacto

    def __post_init__(self) -> None:
        validar_artefacto_arboles(self.artefacto)
        # Copia defensiva -- mismo hallazgo de auditoría que Ejecutor de Dominio 1: sin esto,
        # mutar el dict original después de construir invalidaría la garantía "validado una vez".
        self.artefacto = copy.deepcopy(self.artefacto)

        if self.estado_cuenta is None:
            self.estado_cuenta = EstadoRecursivoPorCuenta(frecuencia_categoria=self.artefacto["frecuencia_poblacional_categoria"])

        # Precómputo hecho UNA VEZ al construir, no en cada decidir(): strides para indexar la
        # tabla plana en O(1) sin reshape de numpy en el camino caliente (mismo espíritu que
        # "código generado" del doc -- eliminar overhead repetido, no cambiar el algoritmo).
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
        """`transaccion` trae al menos `cc_num`, `amt`, `unix_time`,
        `category` (esquema Sparkov, ver `docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`).
        O(1) -- ninguna operación aquí recorre el historial completo ni los
        árboles. Ver advertencia del módulo: el resultado NO debe usarse
        directo, debe pasar por `veto.evaluar()` vía `CicloDecisionArboles`."""
        cc_num = transaccion["cc_num"]
        monto = transaccion["amt"]
        tiempo = transaccion["unix_time"]
        categoria = transaccion["category"]

        _, conteo_ventana_global = self.estado_global.leer_features(monto, tiempo)
        monto_ewma_cuenta, huella_categoria_cuenta = self.estado_cuenta.leer_features(cc_num, monto, categoria, tiempo)
        hora = hora_utc(tiempo)

        valores = {
            "amt": monto,
            "hora": hora,
            "conteo_ventana_global": conteo_ventana_global,
            "monto_ewma_cuenta": monto_ewma_cuenta,
            "huella_categoria_cuenta": huella_categoria_cuenta,
        }

        indice_plano = 0
        for nombre, stride in zip(self._features, self._strides):
            umbrales_feature = self._umbrales[nombre]
            bin_idx = bisect.bisect_left(umbrales_feature, valores[nombre])
            indice_plano += bin_idx * stride

        score_crudo = self._tabla[indice_plano]
        score = _interpolar_isotonica(score_crudo, self._xs_isotonica, self._ys_isotonica)

        # El estado se actualiza DESPUÉS de leer/decidir -- misma semántica causal que Dominio 1:
        # la transacción actual nunca se ve a sí misma en su propio contexto reciente.
        self.estado_global.actualizar(monto, tiempo)
        self.estado_cuenta.actualizar(cc_num, monto, categoria, tiempo)

        # Igual que Ejecutor de Dominio 1: si score es NaN, `nan >= umbral` da False en Python --
        # deliberado, no se guarda aquí. La razón de ser de `veto.py` (que SÍ distingue NaN) es
        # exactamente esta; ver advertencia del módulo y `ciclo_arboles.py::CicloDecisionArboles`.
        return {"score": score, "es_sospechosa": score >= self.artefacto["umbral_decision"]}
