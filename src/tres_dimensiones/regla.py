"""Regla conjuntiva de decision y valor de la informacion muestral.

La regla admite una opcion si supera tres umbrales por separado:
- d1, demanda: Pr(theta < theta_min) <= delta, con delta fijado por el
  decisor como restriccion normativa (proteccion frente a errores de
  especificacion del modelo de valor). Como diagnostico se informa del umbral
  bayesiano implicito b/(b+c), con el que d1 coincidiria con exigir valor
  esperado no negativo.
- d2, capacidad: las horas propias permiten atender el volumen de equilibrio.
- d3, exposicion: Pr(caja acumulada minima < -L) <= epsilon.
Entre las admisibles con valor economico esperado positivo se elige la de
mayor valor; descartar vale 0.

Valor de la informacion. Para una prueba de m ofertas se enumeran los
resultados k = 0..m con su probabilidad predictiva beta-binomial exacta y se
vuelve a decidir con cada posterior. Se informan dos magnitudes:
- incremento preposterior de la politica conjuntiva: puede ser negativo
  incluso con integracion exacta, porque las restricciones dependen de la
  posterior y pueden excluir una accion que antes era admisible;
- EVSI clasico de la politica de valor esperado sin restricciones, que con
  integracion exacta es no negativo.
theta se integra por simulacion con numeros aleatorios comunes, de modo que
ambas magnitudes llevan error de Monte Carlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .coste import DesviacionCoste
from .demanda import CreenciaConversion, conversion_equilibrio, delta_por_perdidas
from .modelo import Mercado, OpcionEntrega, simular_opcion

DESCARTAR = "Descartar o seguir investigando"


@dataclass(frozen=True, slots=True)
class Umbrales:
    """Umbrales de la regla conjuntiva, fijados antes de decidir."""

    delta: float          # Pr(theta < theta_min) admisible (dimension 1)
    epsilon: float        # Pr(caja minima < -L) admisible (dimension 3)
    perdida: float        # L, perdida asumible

    def __post_init__(self) -> None:
        for nombre in ("delta", "epsilon"):
            v = getattr(self, nombre)
            if not (math.isfinite(v) and 0 < v < 1):
                raise ValueError(f"{nombre} debe estar en (0, 1)")
        if not (math.isfinite(self.perdida) and self.perdida >= 0):
            raise ValueError("La perdida asumible debe ser finita y no negativa")


@dataclass(frozen=True, slots=True)
class Aleatorios:
    """Numeros aleatorios comunes para comparar opciones y ramas."""

    u_theta: np.ndarray
    d: np.ndarray
    u_ventas: np.ndarray

    @classmethod
    def generar(cls, semilla: int, n: int, meses: int, desviacion: DesviacionCoste) -> "Aleatorios":
        if not isinstance(n, int) or n <= 1 or not isinstance(meses, int) or meses <= 0:
            raise ValueError("n > 1 y meses > 0 deben ser enteros")
        rng = np.random.default_rng(semilla)
        tiny = np.finfo(float).tiny
        u_theta = np.clip(rng.random(n), tiny, 1 - 1e-16)
        u_ventas = np.clip(rng.random((n, meses)), tiny, 1 - 1e-16)
        return cls(u_theta=u_theta, d=desviacion.muestrear(rng, n), u_ventas=u_ventas)


@dataclass(frozen=True, slots=True)
class EvaluacionOpcion:
    opcion: str
    theta_min: float
    prob_theta_bajo_min: float
    delta_bayes: float
    resumen: dict
    umbrales: dict
    admisible: bool


@dataclass(frozen=True, slots=True)
class Decision:
    creencia: CreenciaConversion
    evaluaciones: tuple[EvaluacionOpcion, ...]
    elegida: str
    valor: float
    error_mc: float
    elegida_valor_esperado: str
    valor_valor_esperado: float


def _delta_bayes(valor: np.ndarray, theta: np.ndarray, tmin: float) -> float:
    """Umbral bayesiano implicito b/(b+c) con c = -E[V|A], b = E[V|no A]."""
    bajo = theta < tmin
    if not bajo.any() or bajo.all():
        return float("nan")
    c, b = -float(valor[bajo].mean()), float(valor[~bajo].mean())
    if c <= 0 or b <= 0:
        return float("nan")
    return delta_por_perdidas(b, c)


def decidir(creencia: CreenciaConversion, opciones: tuple[OpcionEntrega, ...],
            mercado: Mercado, umbrales: Umbrales, aleatorios: Aleatorios) -> Decision:
    theta = np.asarray(creencia.cuantil(aleatorios.u_theta), dtype=float)
    n = theta.size
    evaluaciones, valores = [], {}
    for op in opciones:
        sim = simular_opcion(op, mercado, theta, aleatorios.d, aleatorios.u_ventas)
        res = sim.resumen(umbrales.perdida)
        tmin = conversion_equilibrio(op.costes_fijos_economicos(mercado.meses),
                                     mercado.compradores_horizonte,
                                     op.margen_unitario_economico(mercado.precio))
        p_bajo = creencia.prob_inferior(tmin) if math.isfinite(tmin) else 1.0
        cumple = {
            "d1_demanda": p_bajo <= umbrales.delta,
            "d2_capacidad": math.isfinite(tmin) and op.capacidad_cubre(tmin * mercado.compradores_horizonte / mercado.meses),
            "d3_exposicion": res["prob_exposicion"] <= umbrales.epsilon,
        }
        valores[op.nombre] = sim.valor_economico
        evaluaciones.append(EvaluacionOpcion(op.nombre, tmin, p_bajo,
                                             _delta_bayes(sim.valor_economico, theta, tmin),
                                             res, cumple, all(cumple.values())))

    def mejor(candidatas):
        candidatas = [e for e in candidatas if e.resumen["valor_economico_medio"] > 0]
        return max(candidatas, key=lambda e: e.resumen["valor_economico_medio"]) if candidatas else None

    elegida = mejor([e for e in evaluaciones if e.admisible])
    ev = mejor(evaluaciones)
    error = 0.0 if elegida is None else float(valores[elegida.opcion].std(ddof=1) / math.sqrt(n))
    return Decision(creencia, tuple(evaluaciones),
                    elegida.opcion if elegida else DESCARTAR,
                    elegida.resumen["valor_economico_medio"] if elegida else 0.0, error,
                    ev.opcion if ev else DESCARTAR,
                    ev.resumen["valor_economico_medio"] if ev else 0.0)


@dataclass(frozen=True, slots=True)
class RamaPrueba:
    compras: int
    probabilidad: float
    decision: str
    valor: float
    decision_valor_esperado: str
    valor_valor_esperado: float


@dataclass(frozen=True, slots=True)
class ValorInformacion:
    ofertas: int
    coste: float
    decision_actual: str
    valor_actual: float
    ramas: tuple[RamaPrueba, ...]
    incremento_politica: float
    evsi_clasico: float
    error_mc: float

    @property
    def realizar(self) -> bool:
        """La prueba compensa si el incremento de la politica supera su coste
        con un margen de dos errores de Monte Carlo."""
        return self.incremento_politica - 2 * self.error_mc > self.coste


def valor_informacion(creencia: CreenciaConversion, ofertas_prueba: int, coste_prueba: float,
                      opciones: tuple[OpcionEntrega, ...], mercado: Mercado,
                      umbrales: Umbrales, aleatorios: Aleatorios) -> ValorInformacion:
    if not isinstance(ofertas_prueba, int) or ofertas_prueba <= 0:
        raise ValueError("ofertas_prueba debe ser un entero positivo")
    if not (math.isfinite(coste_prueba) and coste_prueba >= 0):
        raise ValueError("coste_prueba debe ser finito y no negativo")
    actual = decidir(creencia, opciones, mercado, umbrales, aleatorios)
    predictiva = stats.betabinom(ofertas_prueba, creencia.alpha, creencia.beta)
    ramas, esperado, esperado_ev, var = [], 0.0, 0.0, actual.error_mc ** 2
    for k in range(ofertas_prueba + 1):
        dec = decidir(creencia.actualizar(ofertas_prueba, k), opciones, mercado, umbrales, aleatorios)
        p = float(predictiva.pmf(k))
        esperado += p * dec.valor
        esperado_ev += p * dec.valor_valor_esperado
        var += (p * dec.error_mc) ** 2
        ramas.append(RamaPrueba(k, p, dec.elegida, dec.valor,
                                dec.elegida_valor_esperado, dec.valor_valor_esperado))
    return ValorInformacion(ofertas_prueba, coste_prueba, actual.elegida, actual.valor,
                            tuple(ramas), esperado - actual.valor,
                            esperado_ev - actual.valor_valor_esperado, math.sqrt(var))
