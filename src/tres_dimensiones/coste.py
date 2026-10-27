"""Dimension 2: sencillez de entrega.

El coste total se modela como coste estimado por un factor de desviacion D.
Por defecto, D es lognormal en el cuerpo de la distribucion, con una prueba
de estres de cola de Pareto, siguiendo la evidencia de cola gruesa en
proyectos de tecnologias de la informacion (Flyvbjerg et al., 2022).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z90 = 1.2815515655446004  # cuantil 0,90 de la normal tipificada


@dataclass(frozen=True, slots=True)
class DesviacionCoste:
    """Factor D = coste real / coste estimado."""

    mediana: float = 1.25
    p90: float = 3.0
    cola_pareto_alfa: float | None = None  # p. ej. 2.3 para la prueba de estres
    cola_pareto_umbral: float = 2.0

    def __post_init__(self) -> None:
        if self.mediana <= 0 or self.p90 <= self.mediana:
            raise ValueError("Se requiere 0 < mediana < p90")
        if self.cola_pareto_alfa is not None and self.cola_pareto_alfa <= 1:
            raise ValueError("El indice de cola debe ser mayor que 1 para que exista la media")

    @property
    def mu(self) -> float:
        return math.log(self.mediana)

    @property
    def sigma(self) -> float:
        return (math.log(self.p90) - math.log(self.mediana)) / Z90

    @property
    def media_lognormal(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)

    def muestrear(self, rng: np.random.Generator, n: int) -> np.ndarray:
        d = rng.lognormal(self.mu, self.sigma, n)
        if self.cola_pareto_alfa is not None:
            # por encima del umbral, la cola se sustituye por una Pareto de indice alfa
            cola = d > self.cola_pareto_umbral
            u = rng.random(int(cola.sum()))
            d[cola] = self.cola_pareto_umbral * (1 - u) ** (-1 / self.cola_pareto_alfa)
        return d


def desde_cuantiles(mediana: float, p90: float) -> tuple[float, float]:
    """mu = ln(m), sigma = (ln q90 - ln m) / z0,90."""
    return math.log(mediana), (math.log(p90) - math.log(mediana)) / Z90
