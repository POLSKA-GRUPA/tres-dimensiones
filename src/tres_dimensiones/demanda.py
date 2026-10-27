"""Dimension 1: demanda pagadora.

La tasa de conversion contractual theta (proporcion de compradores
cualificados que se comprometen a pagar el precio) se modela con una
distribucion Beta que solo actualizan las compras de nivel 4 de la escalera
de evidencia: contratos firmados o pagos al precio ofrecido.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import math

import numpy as np
from scipy import stats


class NivelEvidencia(IntEnum):
    """Escalera de evidencia, ordenada por lo que arriesga el comprador."""

    OPINION = 0
    INTENCION = 1
    PILOTO_GRATUITO = 2
    RESERVA_RECUPERABLE = 3
    COMPRA = 4
    RENOVACION = 5


@dataclass(frozen=True, slots=True)
class CreenciaConversion:
    """Creencia Beta(alpha, beta) sobre la conversion contractual theta."""

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.alpha) and math.isfinite(self.beta)) or self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Los parametros de la Beta deben ser positivos")

    def actualizar(self, ofertas: int, compras: int) -> "CreenciaConversion":
        """Actualiza con n ofertas cualificadas y k compras de nivel 4.

        Supone ensayos intercambiables con una conversion theta comun.
        """
        if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                   for v in (ofertas, compras)):
            raise ValueError("ofertas y compras deben ser enteros")
        if not 0 <= compras <= ofertas:
            raise ValueError("Se requiere 0 <= compras <= ofertas")
        return CreenciaConversion(self.alpha + compras, self.beta + ofertas - compras)

    @property
    def media(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def intervalo(self, masa: float = 0.90) -> tuple[float, float]:
        cola = (1 - masa) / 2
        dist = stats.beta(self.alpha, self.beta)
        return float(dist.ppf(cola)), float(dist.ppf(1 - cola))

    def prob_inferior(self, umbral: float) -> float:
        """Pr(theta < umbral) bajo la creencia actual."""
        return float(stats.beta.cdf(umbral, self.alpha, self.beta))

    def cuantil(self, u):
        return stats.beta.ppf(u, self.alpha, self.beta)


def evidencia_actualiza_theta(nivel: NivelEvidencia) -> bool:
    """Solo las compras de nivel 4 actualizan theta.

    Los niveles 0 a 3 pueden informar la distribucion inicial; las
    renovaciones (nivel 5) actualizan un parametro de retencion distinto.
    """
    return nivel == NivelEvidencia.COMPRA


def conversion_equilibrio(costes_fijos: float, compradores: float, margen_unitario: float) -> float:
    """theta_min = F / (N * m): conversion que iguala margen y costes fijos."""
    if compradores <= 0 or margen_unitario <= 0:
        return float("inf")
    return costes_fijos / (compradores * margen_unitario)


def delta_por_perdidas(beneficio_omitido: float, perdida_lanzamiento: float) -> float:
    """Umbral b / (b + c) de la regla bayesiana de dos acciones.

    Con A = {theta < theta_min}, p = Pr(A) y V el valor de lanzar frente a
    descartar, si c = -E[V | A] y b = E[V | no A] (ambos positivos), entonces
    E[V] >= 0 equivale a p <= b / (b + c). Con esos valores el umbral de
    demanda coincide con el criterio de valor esperado; por eso el metodo lo
    usa como diagnostico y fija delta como restriccion normativa aparte.
    """
    if beneficio_omitido < 0 or perdida_lanzamiento < 0:
        raise ValueError("Las perdidas deben ser no negativas")
    total = beneficio_omitido + perdida_lanzamiento
    if total == 0:
        raise ValueError("Al menos una de las perdidas debe ser positiva")
    return beneficio_omitido / total
