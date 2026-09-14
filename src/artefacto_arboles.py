"""Contrato del artefacto de política de Dominio 2 (árboles) — módulo
neutral, mismo rol que `artefacto.py` para Dominio 1: ni `ejecutor_arboles.py`
ni `puente_arboles.py` deben ser la fuente de la validación del otro (mismo
hallazgo de la auditoría de Dominio 1, evitado aquí desde el principio).

Formato, decidido en `src/calibrador_arboles.py` (ver
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`, secciones 6 y 7):
- El modelo (Gradient Boosting, 175 árboles) NO se serializa nodo por nodo
  para evaluarse en caliente -- eso ya se midió más lento (30.5 µs con
  código generado) que la alternativa elegida. Lo que el Ejecutor usa es la
  **tabla de búsqueda v3** (`tabla_busqueda_forma`/`tabla_busqueda_plana`):
  el score crudo del modelo (`predict_proba`) ya evaluado una sola vez, por
  adelantado, para cada combinación posible de "bin" de umbral real
  aprendido por el modelo -- exacta por construcción, no aproximada (ver
  sección 6, tercera ronda).
- `umbrales_por_feature` son los cortes reales que separan esos bins
  (extraídos de los nodos de los árboles) -- el Ejecutor solo necesita
  `bisect_left` sobre esta lista para saber en qué bin cae cada feature.
- La calibración isotónica (Paso 5 del afinamiento) se guarda como
  breakpoints `x`/`y` -- interpolación lineal en Python puro reproduce
  `IsotonicRegression.predict()` exactamente para `out_of_bounds="clip"`
  (verificado: máxima diferencia 0.0 contra `np.interp`, que a su vez es
  como se implementa la interpolación en `ejecutor_arboles.py`).
"""
CAMPOS_REQUERIDOS_ARBOLES = {
    "version", "fecha_calibracion", "modelo", "features",
    "umbrales_por_feature", "tabla_busqueda_forma", "tabla_busqueda_plana",
    "calibracion_isotonica", "umbral_decision", "metricas_validacion",
    "frecuencia_poblacional_categoria",
}


class ArtefactoArbolesInvalido(Exception):
    """El artefacto no cumple el contrato de Dominio 2 -- mismo principio que
    `ArtefactoInvalido` de Dominio 1 (ADR_002, invariante 1): sin artefacto
    válido, no hay decisión automática."""


def _es_numero(valor) -> bool:
    # Igual que artefacto.py (Dominio 1): NaN se acepta aquí a propósito -- la protección
    # contra un score NaN es responsabilidad de veto.py (ADR_002, invariante 3), no de esta
    # validación de forma/contenido. Excluirlo aquí solo movería el chequeo al lugar equivocado.
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def validar_artefacto_arboles(artefacto: dict) -> None:
    faltantes = CAMPOS_REQUERIDOS_ARBOLES - artefacto.keys()
    if faltantes:
        raise ArtefactoArbolesInvalido(f"faltan campos requeridos: {faltantes}")

    features = artefacto["features"]
    if not isinstance(features, list) or not features or not all(isinstance(f, str) and f for f in features):
        raise ArtefactoArbolesInvalido("features debe ser una lista no vacía de nombres de columna")

    umbrales = artefacto["umbrales_por_feature"]
    if not isinstance(umbrales, dict) or set(umbrales.keys()) != set(features):
        raise ArtefactoArbolesInvalido("umbrales_por_feature debe tener exactamente una entrada por feature")
    for f in features:
        lista = umbrales[f]
        if not isinstance(lista, list) or not all(_es_numero(v) for v in lista):
            raise ArtefactoArbolesInvalido(f"umbrales_por_feature[{f}] debe ser una lista numérica")
        if list(lista) != sorted(set(lista)):
            raise ArtefactoArbolesInvalido(f"umbrales_por_feature[{f}] debe estar ordenado y sin duplicados")

    forma = artefacto["tabla_busqueda_forma"]
    if not isinstance(forma, list) or len(forma) != len(features) or not all(isinstance(n, int) and n > 0 for n in forma):
        raise ArtefactoArbolesInvalido("tabla_busqueda_forma debe tener un entero positivo por feature")
    forma_esperada = [len(umbrales[f]) + 1 for f in features]
    if forma != forma_esperada:
        raise ArtefactoArbolesInvalido(f"tabla_busqueda_forma {forma} no coincide con umbrales_por_feature (esperado {forma_esperada})")

    tamano_esperado = 1
    for n in forma:
        tamano_esperado *= n
    plana = artefacto["tabla_busqueda_plana"]
    if not isinstance(plana, list) or len(plana) != tamano_esperado:
        raise ArtefactoArbolesInvalido(f"tabla_busqueda_plana debe tener {tamano_esperado} elementos, tiene {len(plana) if isinstance(plana, list) else 'N/A'}")
    if not all(_es_numero(v) for v in plana):
        raise ArtefactoArbolesInvalido("tabla_busqueda_plana debe contener solo números")

    calibracion = artefacto["calibracion_isotonica"]
    if not isinstance(calibracion, dict) or "x" not in calibracion or "y" not in calibracion:
        raise ArtefactoArbolesInvalido("calibracion_isotonica debe tener 'x' e 'y'")
    xs, ys = calibracion["x"], calibracion["y"]
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys) or not xs:
        raise ArtefactoArbolesInvalido("calibracion_isotonica.x e .y deben ser listas no vacías de igual longitud")
    if not all(_es_numero(v) for v in xs) or not all(_es_numero(v) for v in ys):
        raise ArtefactoArbolesInvalido("calibracion_isotonica.x e .y deben ser numéricas")
    if list(xs) != sorted(xs):
        raise ArtefactoArbolesInvalido("calibracion_isotonica.x debe estar ordenado ascendente")

    umbral = artefacto["umbral_decision"]
    if not _es_numero(umbral) or not (0.0 <= umbral <= 1.0):
        raise ArtefactoArbolesInvalido(f"umbral_decision fuera de [0,1] o no numérico: {umbral!r}")

    frecuencia = artefacto["frecuencia_poblacional_categoria"]
    if not isinstance(frecuencia, dict) or not frecuencia or not all(_es_numero(v) for v in frecuencia.values()):
        raise ArtefactoArbolesInvalido("frecuencia_poblacional_categoria debe ser un diccionario numérico no vacío")
