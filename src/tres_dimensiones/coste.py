"""Dimension 2: sencillez de entrega.

El factor de desviacion D = coste real / coste estimado multiplica la
construccion y los costes monetarios fijos de mantenimiento y soporte, que
son las partidas cuya estimacion previa esta sujeta a sesgo de planificacion.
No se aplica a los costes variables por contrato ni a las horas propias, que
se modelan por separado.

Por defecto, D es lognormal en el cuerpo de la distribucion. La prueba de
estres sustituye la cola por encima de un umbral x0 por una cola de Pareto
empalmada en x0: S(x) = S_ln(x0) * (x / x0) ** (-forma) para x > x0.
Flyvbjerg et al. (2022) estiman una densidad proporcional a x ** (-2,3) por
encima de D = 2; la funcion de supervivencia correspondiente decae con
forma 1,3, de modo que la media existe y la varianza no.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

Z90 = 1.2815515655446004  # cuantil 0,90 de la normal tipificada


def _finito_positivo(nombre: str, valor: float) -> None:
    if not (isinstance(valor, (int, float)) and math.isfinite(valor) and valor > 0):
        raise ValueError(f"{nombre} debe ser un numero finito y positivo")


@dataclass(frozen=True, slots=True)
class DesviacionCoste:
    """Factor D = coste real / coste estimado.

    alfa_densidad: exponente de la densidad de la cola de Pareto (p. ej. 2,3).
    La forma de la funcion de supervivencia es alfa_densidad - 1.
    """

    mediana: float = 1.25
    p90: float = 3.0
    alfa_densidad: float | None = None
    umbral_cola: float = 2.0

    def __post_init__(self) -> None:
        _finito_positivo("mediana", self.mediana)
        _finito_positivo("p90", self.p90)
        if self.p90 <= self.mediana:
            raise ValueError("Se requiere mediana < p90")
        _finito_positivo("umbral_cola", self.umbral_cola)
        if self.alfa_densidad is not None:
            _finito_positivo("alfa_densidad", self.alfa_densidad)
            if self.alfa_densidad <= 2:
                raise ValueError("alfa_densidad debe ser mayor que 2 para que exista la media")

    @property
    def mu(self) -> float:
        return math.log(self.mediana)

    @property
    def sigma(self) -> float:
        return (math.log(self.p90) - math.log(self.mediana)) / Z90

    @property
    def forma_cola(self) -> float | None:
        return None if self.alfa_densidad is None else self.alfa_densidad - 1

    @property
    def media_lognormal(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)

    def supervivencia(self, x: float) -> float:
        """Pr(D > x) de la distribucion empleada."""
        s_ln = float(stats.lognorm.sf(x, self.sigma, scale=math.exp(self.mu)))
        if self.forma_cola is None or x <= self.umbral_cola:
            return s_ln
        s0 = float(stats.lognorm.sf(self.umbral_cola, self.sigma, scale=math.exp(self.mu)))
        return s0 * (x / self.umbral_cola) ** (-self.forma_cola)

    def muestrear(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("n debe ser un entero positivo")
        d = rng.lognormal(self.mu, self.sigma, n)
        if self.forma_cola is not None:
            cola = d > self.umbral_cola
            u = rng.random(int(cola.sum()))
            d[cola] = self.umbral_cola * (1 - u) ** (-1 / self.forma_cola)
        return d


def desde_cuantiles(mediana: float, p90: float) -> tuple[float, float]:
    """mu = ln(m), sigma = (ln q90 - ln m) / z0,90."""
    return math.log(mediana), (math.log(p90) - math.log(mediana)) / Z90
