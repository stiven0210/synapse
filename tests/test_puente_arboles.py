import json

import pytest

from src.artefacto_arboles import ArtefactoArbolesInvalido
from src.puente_arboles import leer_vigente, publicar

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


def test_publicar_y_leer_round_trip(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    assert leer_vigente(ruta) == ARTEFACTO


def test_publicar_rechaza_artefacto_invalido_sin_escribir_nada(tmp_path):
    ruta = tmp_path / "artefacto.json"
    invalido = {**ARTEFACTO, "umbral_decision": 2.0}

    with pytest.raises(ArtefactoArbolesInvalido):
        publicar(invalido, ruta)

    assert not ruta.exists()


def test_publicar_no_deja_archivo_temporal_residual(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    assert not (tmp_path / "artefacto.json.tmp").exists()


def test_leer_vigente_sin_archivo_lanza_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_vigente(tmp_path / "no_existe.json")


def test_leer_vigente_archivo_corrupto_lanza_artefacto_invalido(tmp_path):
    ruta = tmp_path / "artefacto.json"
    ruta.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ArtefactoArbolesInvalido):
        leer_vigente(ruta)
