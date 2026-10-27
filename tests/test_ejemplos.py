import importlib.util
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]


def _cargar(nombre):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "ejemplos" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_cuantofacturan_sin_margen_no_rompe_la_salida(capsys, monkeypatch):
    modulo = _cargar("caso_cuantofacturan")
    monkeypatch.setattr(modulo, "N", 2_000)
    monkeypatch.setattr(modulo, "HORAS", (8.0,))
    monkeypatch.setattr(modulo, "COMPRADORES", (30,))
    monkeypatch.setattr(sys, "argv", ["caso_cuantofacturan.py"])
    modulo.main()
    salida = capsys.readouterr().out
    assert "sin margen" in salida
    assert "Descartar" in salida


def test_cuantofacturan_json_estandar(capsys, monkeypatch):
    import json
    modulo = _cargar("caso_cuantofacturan")
    monkeypatch.setattr(modulo, "N", 2_000)
    monkeypatch.setattr(modulo, "HORAS", (2.0, 8.0))
    monkeypatch.setattr(modulo, "COMPRADORES", (60,))
    monkeypatch.setattr(sys, "argv", ["caso_cuantofacturan.py", "--json"])
    modulo.main()
    datos = json.loads(capsys.readouterr().out)
    assert datos[1]["theta_min"] is None
