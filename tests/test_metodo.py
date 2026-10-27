import math

import numpy as np
import pytest
from scipy import stats

from tres_dimensiones import (DESCARTAR, CreenciaConversion, DesviacionCoste, Mercado,
                              NivelEvidencia, OpcionEntrega, Umbrales, conversion_equilibrio,
                              decidir, delta_por_perdidas, desde_cuantiles,
                              evidencia_actualiza_theta, perdida_asumible,
                              tolerancia_riesgo_propietario, valor_informacion)
from tres_dimensiones.adaptadores.motor_caja import financiacion_por_motor, motor_referencia
from tres_dimensiones.exposicion import caja_desde_flujos
from tres_dimensiones.regla import Aleatorios

OPCION = OpcionEntrega("Prueba", 2, construccion=3_000, fijo_mensual=150, coste_variable=120,
                       horas_contrato=4.0, horas_fijas_mes=4, coste_hora=45, capacidad_mes=60)
MERCADO = Mercado(precio=900, compradores_horizonte=150)


# Dimension 1 -------------------------------------------------------------

def test_actualizacion_beta_conjugada():
    c = CreenciaConversion().actualizar(12, 3)
    assert (c.alpha, c.beta) == (4, 10)
    assert c.media == pytest.approx(4 / 14)


def test_intervalo_del_ejemplo_del_texto():
    lo, hi = CreenciaConversion().actualizar(12, 3).intervalo(0.90)
    assert lo == pytest.approx(0.1127, abs=1e-3)
    assert hi == pytest.approx(0.4947, abs=1e-3)


def test_cero_compras_no_descarta_tasas_moderadas():
    # Beta(1, 11): percentil 95 en torno al 24 %
    c = CreenciaConversion().actualizar(10, 0)
    assert stats.beta.ppf(0.95, c.alpha, c.beta) == pytest.approx(0.2384, abs=1e-3)


def test_solo_las_compras_actualizan_theta():
    assert evidencia_actualiza_theta(NivelEvidencia.COMPRA)
    for nivel in NivelEvidencia:
        if nivel is not NivelEvidencia.COMPRA:
            assert not evidencia_actualiza_theta(nivel)


def test_actualizacion_rechaza_datos_imposibles():
    with pytest.raises(ValueError):
        CreenciaConversion().actualizar(3, 4)


def test_conversion_equilibrio():
    assert conversion_equilibrio(9_000, 150, 600) == pytest.approx(0.1)
    assert conversion_equilibrio(1, 150, 0) == math.inf


def test_delta_regla_bayesiana_dos_acciones():
    assert delta_por_perdidas(beneficio_omitido=1, perdida_lanzamiento=3) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        delta_por_perdidas(0, 0)


# Dimension 2 -------------------------------------------------------------

def test_parametros_lognormales_desde_cuantiles():
    mu, sigma = desde_cuantiles(1.25, 3.0)
    assert mu == pytest.approx(0.2231, abs=1e-4)
    assert sigma == pytest.approx(0.6831, abs=1e-4)
    assert DesviacionCoste(1.25, 3.0).media_lognormal == pytest.approx(1.58, abs=0.01)


def test_media_lognormal_supera_a_la_mediana_con_mu_cero():
    d = DesviacionCoste(mediana=1.0, p90=2.0)
    assert d.media_lognormal > 1.0


def test_cola_pareto_es_mas_gruesa():
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
    base = DesviacionCoste(1.25, 3.0).muestrear(rng1, 200_000)
    estres = DesviacionCoste(1.25, 3.0, cola_pareto_alfa=2.3).muestrear(rng2, 200_000)
    assert np.quantile(estres, 0.999) > np.quantile(base, 0.999)


def test_capacidad_raciona_contratos():
    assert OPCION.contratos_maximos_mes() == 14.0  # (60 - 4) / 4


# Dimension 3 -------------------------------------------------------------

def test_perdida_asumible_delquie():
    r = tolerancia_riesgo_propietario(60_000)
    assert r == pytest.approx(10_000)
    assert perdida_asumible(r, 1 / 20) == pytest.approx(10_000 * math.log(20))
    assert perdida_asumible(r, 0.05) / r == pytest.approx(3.0, abs=0.01)


def test_exposicion_sobre_caja_minima_no_sobre_van():
    flujos = np.array([[-10.0, -10.0, 30.0]])
    caja = caja_desde_flujos(flujos, 0.0)
    assert caja.van[0] == pytest.approx(10.0)
    assert caja.caja_minima[0] == pytest.approx(-20.0)
    assert caja.prob_exposicion(15.0) == 1.0


def test_adaptador_motor_caja_coincide_con_referencia():
    cobros = np.array([[0.0, 5.0, 30.0], [0.0, 0.0, 1.0]])
    pagos = np.array([[10.0, 10.0, 0.0], [2.0, 2.0, 2.0]])
    fin = financiacion_por_motor(cobros, pagos, motor_referencia)
    caja = caja_desde_flujos(cobros - pagos, 0.0)
    assert np.allclose(fin, caja.financiacion_adicional)


# Regla y valor de la informacion -----------------------------------------

def _alea(n=20_000, desv=None):
    return Aleatorios.generar(7, n, MERCADO.meses, desv or DesviacionCoste(1.25, 3.0))


def test_regla_conjuntiva_descarta_si_falla_exposicion():
    cara = OpcionEntrega("Cara", 4, construccion=200_000, fijo_mensual=900, coste_variable=40,
                         horas_contrato=0.5, horas_fijas_mes=10, coste_hora=45, capacidad_mes=60)
    dec = decidir(CreenciaConversion().actualizar(12, 3), (cara,), MERCADO,
                  Umbrales(epsilon=0.05, perdida=30_000), _alea())
    assert dec.elegida == DESCARTAR
    assert not dec.evaluaciones[0].umbrales["d3_exposicion"]


def test_valor_informacion_no_negativo_y_nulo_si_decision_invariante():
    vi = valor_informacion(CreenciaConversion().actualizar(12, 3), 5, 400, (OPCION,), MERCADO,
                           Umbrales(epsilon=0.05, perdida=30_000), _alea())
    assert all(r.decision == vi.decision_actual for r in vi.ramas)
    assert abs(vi.evsi_bruto) < 0.01 * abs(vi.valor_actual) + 50  # error de simulacion
    assert not vi.realizar


def test_probabilidades_predictivas_suman_uno():
    vi = valor_informacion(CreenciaConversion().actualizar(5, 0), 5, 500, (OPCION,), MERCADO,
                           Umbrales(epsilon=0.05, perdida=30_000), _alea(5_000))
    assert sum(r.probabilidad for r in vi.ramas) == pytest.approx(1.0)


def test_reproducibilidad_con_semilla():
    a = decidir(CreenciaConversion().actualizar(12, 3), (OPCION,), MERCADO,
                Umbrales(epsilon=0.05, perdida=30_000), _alea())
    b = decidir(CreenciaConversion().actualizar(12, 3), (OPCION,), MERCADO,
                Umbrales(epsilon=0.05, perdida=30_000), _alea())
    assert a.valor == b.valor
