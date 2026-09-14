import pytest

from src.artefacto_arboles import ArtefactoArbolesInvalido, validar_artefacto_arboles

ARTEFACTO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "gradient_boosting",
    "features": ["amt", "hora", "conteo_ventana_global", "monto_ewma_cuenta", "huella_categoria_cuenta"],
    "umbrales_por_feature": {
        "amt": [100.0],
        "hora": [],
        "conteo_ventana_global": [],
        "monto_ewma_cuenta": [],
        "huella_categoria_cuenta": [],
    },
    "tabla_busqueda_forma": [2, 1, 1, 1, 1],
    "tabla_busqueda_plana": [0.1, 0.9],
    "calibracion_isotonica": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "umbral_decision": 0.5,
    "metricas_validacion": {},
    "frecuencia_poblacional_categoria": {"x": 1.0},
}


def test_artefacto_valido_no_lanza():
    validar_artefacto_arboles(ARTEFACTO)  # no debe lanzar


def test_falta_campo_requerido():
    invalido = {k: v for k, v in ARTEFACTO.items() if k != "tabla_busqueda_plana"}
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_umbrales_por_feature_debe_tener_una_entrada_por_feature():
    invalido = {**ARTEFACTO, "umbrales_por_feature": {"amt": [100.0]}}  # faltan las otras 4
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_umbrales_por_feature_debe_estar_ordenado_sin_duplicados():
    invalido = {**ARTEFACTO, "umbrales_por_feature": {**ARTEFACTO["umbrales_por_feature"], "amt": [100.0, 50.0]}}
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_tabla_busqueda_forma_debe_coincidir_con_umbrales():
    invalido = {**ARTEFACTO, "tabla_busqueda_forma": [3, 1, 1, 1, 1]}  # amt tiene 1 umbral -> debería ser 2, no 3
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_tabla_busqueda_plana_debe_tener_el_tamano_esperado():
    invalido = {**ARTEFACTO, "tabla_busqueda_plana": [0.1, 0.9, 0.5]}  # forma dice 2*1*1*1*1=2, no 3
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_calibracion_isotonica_x_debe_estar_ordenada():
    invalido = {**ARTEFACTO, "calibracion_isotonica": {"x": [1.0, 0.0], "y": [0.0, 1.0]}}
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_umbral_decision_fuera_de_rango():
    invalido = {**ARTEFACTO, "umbral_decision": 1.5}
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_frecuencia_poblacional_categoria_no_puede_estar_vacia():
    invalido = {**ARTEFACTO, "frecuencia_poblacional_categoria": {}}
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)


def test_features_no_puede_estar_vacio():
    invalido = {**ARTEFACTO, "features": []}
    with pytest.raises(ArtefactoArbolesInvalido):
        validar_artefacto_arboles(invalido)
