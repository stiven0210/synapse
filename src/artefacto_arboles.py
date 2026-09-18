"""Domain 2 (trees) policy artifact contract — neutral module, same role
as `artefacto.py` for Domain 1: neither `ejecutor_arboles.py` nor
`puente_arboles.py` should be the source of the other's validation (same
finding from the Domain 1 audit, avoided here from the start).

Format, decided in `src/calibrador_arboles.py` (see
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md`, sections 6 and 7):
- The model (Gradient Boosting, 175 trees) is NOT serialized node by node
  for hot-path evaluation -- that was already measured to be slower (30.5
  µs with generated code) than the chosen alternative. What the Executor
  uses is the **v3 lookup table** (`tabla_busqueda_forma`/`tabla_busqueda_plana`):
  the model's raw score (`predict_proba`) already evaluated once, ahead of
  time, for every possible combination of "bin" from the real thresholds
  the model learned -- exact by construction, not approximated (see
  section 6, third round).
- `umbrales_por_feature` are the real cut points separating those bins
  (extracted from the trees' nodes) -- the Executor only needs
  `bisect_left` over this list to know which bin each feature falls into.
- The isotonic calibration (tuning Step 5) is saved as `x`/`y` breakpoints
  -- pure-Python linear interpolation reproduces
  `IsotonicRegression.predict()` exactly for `out_of_bounds="clip"`
  (verified: maximum difference 0.0 against `np.interp`, which is also
  how the interpolation is implemented in `ejecutor_arboles.py`).
"""
CAMPOS_REQUERIDOS_ARBOLES = {
    "version", "fecha_calibracion", "modelo", "features",
    "umbrales_por_feature", "tabla_busqueda_forma", "tabla_busqueda_plana",
    "calibracion_isotonica", "umbral_decision", "metricas_validacion",
    "frecuencia_poblacional_categoria",
}


class ArtefactoArbolesInvalido(Exception):
    """The artifact doesn't satisfy the Domain 2 contract -- same principle
    as Domain 1's `ArtefactoInvalido` (ADR_002, invariant 1): no valid
    artifact, no automated decision."""


def _es_numero(valor) -> bool:
    # Same as artefacto.py (Domain 1): NaN is accepted here on purpose -- protection
    # against a NaN score is veto.py's responsibility (ADR_002, invariant 3), not this
    # shape/content validation. Excluding it here would just move the check to the wrong place.
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
