import json

import pytest

from src.ejecutor import ArtefactoInvalido
from src.puente import leer_vigente, publicar

ARTEFACTO = {
    "version": 1,
    "fecha_calibracion": "2026-01-01T00:00:00",
    "modelo": "regresion_logistica",
    "features": ["Amount"],
    "coeficientes": [0.1],
    "intercepto": -1.0,
    "umbral_decision": 0.5,
    "metricas_validacion": {},
}


def test_publicar_y_leer_round_trip(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    assert leer_vigente(ruta) == ARTEFACTO


def test_publicar_rechaza_artefacto_invalido_sin_escribir_nada(tmp_path):
    ruta = tmp_path / "artefacto.json"
    invalido = {**ARTEFACTO, "umbral_decision": 2.0}

    with pytest.raises(ArtefactoInvalido):
        publicar(invalido, ruta)

    assert not ruta.exists()  # an invalid artifact is never published, not even partially


def test_publicar_no_deja_archivo_temporal_residual(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    assert not (tmp_path / "artefacto.json.tmp").exists()


def test_publicar_dos_veces_la_segunda_gana(tmp_path):
    ruta = tmp_path / "artefacto.json"
    publicar(ARTEFACTO, ruta)
    segundo = {**ARTEFACTO, "version": 2, "umbral_decision": 0.7}
    publicar(segundo, ruta)

    vigente = leer_vigente(ruta)
    assert vigente["version"] == 2
    assert vigente["umbral_decision"] == 0.7


def test_leer_vigente_sin_archivo_lanza_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_vigente(tmp_path / "no_existe.json")


def test_leer_vigente_archivo_corrupto_lanza_artefacto_invalido(tmp_path):
    ruta = tmp_path / "artefacto.json"
    ruta.write_text(json.dumps({"version": 1}), encoding="utf-8")  # missing almost all the fields
    with pytest.raises(ArtefactoInvalido):
        leer_vigente(ruta)
