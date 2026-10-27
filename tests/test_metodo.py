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
from tres_dimensiones.modelo import simular_opcion
from tres_dimensiones.regla import Aleatorios, valor_informacion_replicado

OPCION = OpcionEntrega("Prueba", 2, construccion=3_000, fijo_mensual=150, coste_variable=120,
                       horas_contrato=4.0, horas_fijas_mes=4, coste_hora=45, capacidad_mes=60)
MERCADO = Mercado(precio=900, compradores_horizonte=150)
UMBRALES = Umbrales(delta=0.25, epsilon=0.05, perdida=30_000)


def _alea(n=20_000, desv=None, semilla=7, meses=12):
    return Aleatorios.generar(semilla, n, meses, desv or DesviacionCoste(1.25, 3.0))


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
    c = CreenciaConversion().actualizar(10, 0)
    assert stats.beta.ppf(0.95, c.alpha, c.beta) == pytest.approx(0.2384, abs=1e-3)


def test_solo_las_compras_actualizan_theta():
    assert evidencia_actualiza_theta(NivelEvidencia.COMPRA)
    assert not any(evidencia_actualiza_theta(n) for n in NivelEvidencia if n is not NivelEvidencia.COMPRA)


@pytest.mark.parametrize("n,k", [(3, 4), (2.5, 0), (3, 0.5), (-1, 0)])
def test_actualizacion_rechaza_datos_imposibles(n, k):
    with pytest.raises(ValueError):
        CreenciaConversion().actualizar(n, k)


def test_conversion_equilibrio():
    assert conversion_equilibrio(9_000, 150, 600) == pytest.approx(0.1)
    assert conversion_equilibrio(1, 150, 0) == math.inf


def test_umbral_bayesiano_equivale_a_valor_esperado_no_negativo():
    # E[V] = (1 - p) b - p c >= 0  <=>  p <= b / (b + c)
    b, c = 300.0, 900.0
    umbral = delta_por_perdidas(b, c)
    for p in (umbral - 1e-6, umbral + 1e-6):
        assert ((1 - p) * b - p * c >= 0) == (p <= umbral)


# Dimension 2 -------------------------------------------------------------

def test_parametros_lognormales_desde_cuantiles():
    mu, sigma = desde_cuantiles(1.25, 3.0)
    assert mu == pytest.approx(0.2231, abs=1e-4)
    assert sigma == pytest.approx(0.6831, abs=1e-4)
    assert DesviacionCoste(1.25, 3.0).media_lognormal == pytest.approx(1.58, abs=0.01)


def test_cola_pareto_usa_forma_de_supervivencia_alfa_menos_uno():
    d = DesviacionCoste(1.25, 3.0, alfa_densidad=2.3, umbral_cola=2.0)
    assert d.forma_cola == pytest.approx(1.3)
    # supervivencia empalmada: S(x) = S(2) (x/2) ** -1.3
    assert d.supervivencia(10) / d.supervivencia(2) == pytest.approx(5 ** -1.3)
    muestra = d.muestrear(np.random.default_rng(3), 400_000)
    assert (muestra > 10).mean() == pytest.approx(d.supervivencia(10), rel=0.05)


def test_estres_domina_al_central_en_toda_la_cola():
    central = DesviacionCoste(1.25, 3.0)
    estres = DesviacionCoste(1.25, 3.0, alfa_densidad=2.3)
    for x in (2.5, 3, 5, 10, 50):
        assert estres.supervivencia(x) >= central.supervivencia(x)


@pytest.mark.parametrize("kwargs", [dict(umbral_cola=0.0, alfa_densidad=2.3),
                                    dict(umbral_cola=-1.0, alfa_densidad=2.3),
                                    dict(alfa_densidad=2.0), dict(mediana=float("nan"))])
def test_desviacion_rechaza_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        DesviacionCoste(**{"mediana": 1.25, "p90": 3.0, **kwargs})


def test_capacidad_raciona_contratos():
    assert OPCION.contratos_maximos_mes() == 14.0  # (60 - 4) / 4


# Dimension 3 -------------------------------------------------------------

def test_perdida_asumible_y_limite_de_capacidad():
    r = tolerancia_riesgo_propietario(60_000)
    assert r == pytest.approx(10_000)
    assert perdida_asumible(r, 0.05) / r == pytest.approx(math.log(20))
    assert perdida_asumible(r, 0.001, capacidad_financiera=60_000) == 60_000


def test_cota_de_delquie_es_supremo_no_alcanzado_con_ganancia_finita():
    r, eps = 10_000.0, 0.05
    l = perdida_asumible(r, eps)

    def aceptable(perdida, ganancia):
        return eps * math.exp(perdida / r) + (1 - eps) * math.exp(-ganancia / r) <= 1

    assert not aceptable(l, 5e4)            # en la cota, una ganancia finita no basta
    assert aceptable(0.99 * l, 1e5)         # por debajo de la cota, una ganancia grande basta
    assert not aceptable(0.99 * l, 1e3)     # pero no una ganancia pequena


def test_exposicion_sobre_caja_minima_no_sobre_van():
    caja = caja_desde_flujos(np.array([[-10.0, -10.0, 30.0]]), 0.0)
    assert caja.van[0] == pytest.approx(10.0)
    assert caja.caja_minima[0] == pytest.approx(-20.0)
    assert caja.prob_exposicion(15.0) == 1.0
    with pytest.raises(ValueError):
        caja.prob_exposicion(float("nan"))


def test_calendario_paga_variables_al_entregar_aunque_el_cobro_quede_fuera():
    op = OpcionEntrega("Calendario", 2, construccion=0, fijo_mensual=0, coste_variable=80,
                       horas_contrato=0, horas_fijas_mes=0, coste_hora=0, capacidad_mes=10)
    mercado = Mercado(precio=100, compradores_horizonte=1, meses=1, retraso_cobro=2, tasa_anual=0.0)
    u = np.array([[0.9]])  # Poisson(1) en el cuantil 0,9 = 2 contratos
    sim = simular_opcion(op, mercado, np.array([1.0]), np.array([1.0]), u)
    contratos = stats.poisson.ppf(0.9, 1.0)
    assert sim.caja_minima[0] == pytest.approx(-80 * contratos)
    assert sim.van_caja[0] == pytest.approx(20 * contratos)


@pytest.mark.parametrize("retraso", [0, 1, 5])
def test_calendario_con_varios_retrasos(retraso):
    mercado = Mercado(precio=900, compradores_horizonte=150, retraso_cobro=retraso, tasa_anual=0.0)
    alea = _alea(2_000)
    sim = simular_opcion(OPCION, mercado, np.full(2_000, 0.3), alea.d, alea.u_ventas)
    assert np.isfinite(sim.van_caja).all()


def test_adaptador_motor_caja_coincide_con_referencia():
    cobros = np.array([[0.0, 5.0, 30.0], [0.0, 0.0, 1.0]])
    pagos = np.array([[10.0, 10.0, 0.0], [2.0, 2.0, 2.0]])
    fin = financiacion_por_motor(cobros, pagos, motor_referencia)
    assert np.allclose(fin, caja_desde_flujos(cobros - pagos, 0.0).financiacion_adicional)


@pytest.mark.parametrize("kwargs", [dict(delta=0.25, epsilon=0.05, perdida=float("nan")),
                                    dict(delta=float("nan"), epsilon=0.05, perdida=1.0),
                                    dict(delta=0.25, epsilon=0.0, perdida=1.0)])
def test_umbrales_rechazan_entradas_no_finitas(kwargs):
    with pytest.raises(ValueError):
        Umbrales(**kwargs)


def test_formas_invalidas_se_rechazan():
    with pytest.raises(ValueError):
        simular_opcion(OPCION, MERCADO, np.full(5, 0.3), np.ones(4), np.full((5, 12), 0.5))


# Regla y valor de la informacion -----------------------------------------

def test_regla_conjuntiva_descarta_si_falla_exposicion():
    cara = OpcionEntrega("Cara", 4, construccion=200_000, fijo_mensual=900, coste_variable=40,
                         horas_contrato=0.5, horas_fijas_mes=10, coste_hora=45, capacidad_mes=60)
    dec = decidir(CreenciaConversion().actualizar(12, 3), (cara,), MERCADO, UMBRALES, _alea())
    assert dec.elegida == DESCARTAR
    assert not dec.evaluaciones[0].umbrales["d3_exposicion"]


def test_politica_invariante_tiene_incremento_nulo_salvo_error_mc():
    vi = valor_informacion(CreenciaConversion().actualizar(12, 3), 5, 400, (OPCION,), MERCADO,
                           UMBRALES, _alea())
    assert all(r.decision == vi.decision_actual for r in vi.ramas)
    assert abs(vi.incremento_politica) <= 3 * vi.error_incremento + 1e-9
    assert not vi.realizar


def test_incremento_de_politica_restringida_puede_ser_negativo():
    # Contraejemplo analitico: valor esperado de lanzar = 1000 theta - 100.
    op = OpcionEntrega("Simple", 4, construccion=100, fijo_mensual=0, coste_variable=0,
                       horas_contrato=0, horas_fijas_mes=0, coste_hora=0, capacidad_mes=1)
    mercado = Mercado(precio=10, compradores_horizonte=100, meses=1, retraso_cobro=0, tasa_anual=0.0)
    alea = Aleatorios.generar(11, 200_000, 1, DesviacionCoste(1.0 + 1e-9, 1.0 + 2e-9))
    alea = Aleatorios(alea.u_theta, np.ones_like(alea.d), alea.u_ventas)
    u = Umbrales(delta=0.11, epsilon=0.99, perdida=1e9)
    vi = valor_informacion(CreenciaConversion(), 1, 0, (op,), mercado, u, alea)
    assert vi.incremento_politica == pytest.approx(-116.67, abs=3)
    assert vi.evsi_clasico >= -3 * vi.error_evsi
    assert vi.error_evsi > 0


def test_probabilidades_predictivas_suman_uno():
    vi = valor_informacion(CreenciaConversion().actualizar(5, 0), 5, 500, (OPCION,), MERCADO,
                           UMBRALES, _alea(5_000))
    assert sum(r.probabilidad for r in vi.ramas) == pytest.approx(1.0)


def test_reproducibilidad_y_estabilidad_entre_semillas():
    c = CreenciaConversion().actualizar(12, 3)
    a = decidir(c, (OPCION,), MERCADO, UMBRALES, _alea(semilla=1))
    b = decidir(c, (OPCION,), MERCADO, UMBRALES, _alea(semilla=1))
    assert a.valor == b.valor
    otra = decidir(c, (OPCION,), MERCADO, UMBRALES, _alea(semilla=2))
    assert otra.elegida == a.elegida
    assert abs(otra.valor - a.valor) <= 5 * math.hypot(a.error_mc, otra.error_mc)


def test_cola_pesada_no_declara_error_estandar():
    alea = _alea(5_000, DesviacionCoste(1.25, 3.0, alfa_densidad=2.3))
    assert alea.cola_pesada
    dec = decidir(CreenciaConversion().actualizar(12, 3), (OPCION,), MERCADO, UMBRALES, alea)
    assert dec.error_mc is None


@pytest.mark.parametrize("u", [0.0, 1.0, float("nan")])
def test_aleatorios_rechaza_uniformes_no_interiores(u):
    base = _alea(10)
    u_theta = base.u_theta.copy(); u_theta[0] = u
    with pytest.raises(ValueError):
        Aleatorios(u_theta, base.d, base.u_ventas)


def test_aleatorios_rechaza_tamano_insuficiente():
    with pytest.raises(ValueError):
        Aleatorios(np.array([0.5]), np.array([1.0]), np.full((1, 12), 0.5))


def test_nombres_duplicados_se_rechazan():
    with pytest.raises(ValueError):
        decidir(CreenciaConversion(), (OPCION, OPCION), MERCADO, UMBRALES, _alea(100))


def test_replicas_independientes_recogen_incertidumbre_de_seleccion():
    rep = valor_informacion_replicado(CreenciaConversion().actualizar(12, 3), 5, 400, (OPCION,), MERCADO,
                                      UMBRALES, DesviacionCoste(1.25, 3.0), (1, 2, 3), 5_000)
    assert len(rep.replicas) == 3
    assert rep.decisiones_estables
    assert not rep.realizar
    assert rep.error_replicas >= 0
