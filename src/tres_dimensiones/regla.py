"""Regla conjuntiva de decision y valor esperado de la informacion muestral."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .coste import DesviacionCoste
from .demanda import CreenciaConversion, conversion_equilibrio, delta_por_perdidas
from .modelo import Mercado, OpcionEntrega, simular_opcion

DESCARTAR = "Descartar o seguir investigando"


@dataclass(frozen=True, slots=True)
class Umbrales:
    """Umbrales de la regla conjuntiva, fijados antes de decidir."""

    epsilon: float        # Pr(caja minima < -L) admisible (dimension 3)
    perdida: float        # L, perdida asumible
    delta: float | None = None  # Pr(theta < theta_min) admisible; None = b/(b+c) por opcion


@dataclass(frozen=True, slots=True)
class Aleatorios:
    """Numeros aleatorios comunes para comparar ramas sin ruido de simulacion."""

    u_theta: np.ndarray
    d: np.ndarray
    u_ventas: np.ndarray

    @classmethod
    def generar(cls, semilla: int, n: int, meses: int, desviacion: DesviacionCoste) -> "Aleatorios":
        rng = np.random.default_rng(semilla)
        return cls(u_theta=rng.random(n), d=desviacion.muestrear(rng, n),
                   u_ventas=rng.random((n, meses)))


@dataclass(frozen=True, slots=True)
class EvaluacionOpcion:
    opcion: str
    theta_min: float
    prob_theta_bajo_min: float
    resumen: dict
    umbrales: dict
    admisible: bool


@dataclass(frozen=True, slots=True)
class Decision:
    creencia: CreenciaConversion
    evaluaciones: tuple[EvaluacionOpcion, ...]
    elegida: str
    valor: float


def _delta_simulado(valor: np.ndarray, theta: np.ndarray, tmin: float) -> float:
    """delta = b / (b + c) estimado con la simulacion de la propia opcion.

    c: perdida media si se lanza y theta < theta_min.
    b: valor medio que se deja de ganar si no se lanza y theta >= theta_min.
    """
    bajo = theta < tmin
    c = float(np.maximum(0.0, -valor[bajo]).mean()) if bajo.any() else 0.0
    b = float(np.maximum(0.0, valor[~bajo]).mean()) if (~bajo).any() else 0.0
    if b + c == 0:
        return 0.0
    return delta_por_perdidas(b, c)


def decidir(creencia: CreenciaConversion, opciones: tuple[OpcionEntrega, ...],
            mercado: Mercado, umbrales: Umbrales, aleatorios: Aleatorios) -> Decision:
    """Aplica la regla conjuntiva y ordena las admisibles por valor economico esperado.

    Descartar vale 0 y esta siempre disponible.
    """
    theta = creencia.cuantil(aleatorios.u_theta)
    evaluaciones = []
    for op in opciones:
        sim = simular_opcion(op, mercado, theta, aleatorios.d, aleatorios.u_ventas)
        res = sim.resumen(umbrales.perdida)
        tmin = conversion_equilibrio(op.costes_fijos_economicos(mercado.meses),
                                     mercado.compradores_horizonte,
                                     op.margen_unitario_economico(mercado.precio))
        p_bajo = creencia.prob_inferior(tmin)
        delta = umbrales.delta if umbrales.delta is not None else _delta_simulado(sim.valor_economico, theta, tmin)
        res["delta"] = delta
        cumple = {
            "d1_demanda": p_bajo <= delta,
            "d2_capacidad": op.capacidad_cubre(tmin * mercado.compradores_horizonte / mercado.meses),
            "d3_exposicion": res["prob_exposicion"] <= umbrales.epsilon,
        }
        evaluaciones.append(EvaluacionOpcion(op.nombre, tmin, p_bajo, res, cumple, all(cumple.values())))
    admisibles = [e for e in evaluaciones
                  if e.admisible and e.resumen["valor_economico_medio"] > 0]
    if not admisibles:
        return Decision(creencia, tuple(evaluaciones), DESCARTAR, 0.0)
    mejor = max(admisibles, key=lambda e: e.resumen["valor_economico_medio"])
    return Decision(creencia, tuple(evaluaciones), mejor.opcion, mejor.resumen["valor_economico_medio"])


@dataclass(frozen=True, slots=True)
class RamaPrueba:
    compras: int
    probabilidad: float
    decision: str
    valor: float


@dataclass(frozen=True, slots=True)
class ValorInformacion:
    ofertas: int
    coste: float
    decision_actual: str
    valor_actual: float
    ramas: tuple[RamaPrueba, ...]
    evsi_bruto: float

    @property
    def realizar(self) -> bool:
        return self.evsi_bruto > self.coste

    @property
    def evsi_neto(self) -> float:
        return self.evsi_bruto - self.coste


def valor_informacion(creencia: CreenciaConversion, ofertas_prueba: int, coste_prueba: float,
                      opciones: tuple[OpcionEntrega, ...], mercado: Mercado,
                      umbrales: Umbrales, aleatorios: Aleatorios) -> ValorInformacion:
    """Analisis preposterior exacto en theta (predictiva beta-binomial).

    Para cada resultado posible k de la prueba se actualiza la creencia, se
    vuelve a aplicar la regla conjuntiva y se pondera el valor de la accion
    elegida por su probabilidad predictiva. El EVSI bruto es esa media menos
    el valor de decidir sin la prueba.
    """
    actual = decidir(creencia, opciones, mercado, umbrales, aleatorios)
    predictiva = stats.betabinom(ofertas_prueba, creencia.alpha, creencia.beta)
    ramas = []
    esperado = 0.0
    for k in range(ofertas_prueba + 1):
        posterior = creencia.actualizar(ofertas_prueba, k)
        dec = decidir(posterior, opciones, mercado, umbrales, aleatorios)
        prob = float(predictiva.pmf(k))
        esperado += prob * dec.valor
        ramas.append(RamaPrueba(k, prob, dec.elegida, dec.valor))
    return ValorInformacion(ofertas_prueba, coste_prueba, actual.elegida, actual.valor,
                            tuple(ramas), esperado - actual.valor)
