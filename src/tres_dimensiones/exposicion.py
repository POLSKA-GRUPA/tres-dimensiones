"""Dimension 3: retorno y exposicion.

La exposicion se mide sobre la caja acumulada, no sobre el VAN, porque la
continuidad de la empresa se pierde en el peor momento del recorrido. La
restriccion es Pr(min_t caja acumulada_t < -L) <= epsilon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def tolerancia_riesgo_propietario(colchon_liquido: float) -> float:
    """R = W / 6, regla para el propietario con toda su riqueza en la empresa.

    Smith (2004) interpreta asi la regla de Howard (1988) de una sexta parte
    del patrimonio. Es un valor por defecto que el decisor puede sustituir.
    """
    if colchon_liquido <= 0:
        raise ValueError("El colchon liquido debe ser positivo")
    return colchon_liquido / 6


def perdida_asumible(tolerancia_riesgo: float, epsilon: float) -> float:
    """L = R * ln(1 / epsilon).

    Con utilidad exponencial de tolerancia R, una perdida L con probabilidad
    epsilon es aceptable en el margen cuando epsilon = exp(-L / R); con
    epsilon = 1/20 se obtiene L aproximadamente igual a 3R (Delquie, 2008).
    """
    if not 0 < epsilon < 1:
        raise ValueError("epsilon debe estar en (0, 1)")
    return tolerancia_riesgo * math.log(1 / epsilon)


def tasa_mensual(tasa_anual: float) -> float:
    return (1 + tasa_anual) ** (1 / 12) - 1


@dataclass(frozen=True, slots=True)
class ResultadoCaja:
    flujos: np.ndarray            # (simulaciones, meses + 1)
    caja_acumulada: np.ndarray    # (simulaciones, meses + 1)
    van: np.ndarray               # (simulaciones,)

    @property
    def caja_minima(self) -> np.ndarray:
        return self.caja_acumulada.min(axis=1)

    def prob_exposicion(self, perdida: float) -> float:
        return float((self.caja_minima < -perdida).mean())

    @property
    def financiacion_adicional(self) -> np.ndarray:
        """Equivalente a la financiacion adicional de un motor determinista de caja."""
        return np.maximum(0.0, -self.caja_minima)


def caja_desde_flujos(flujos: np.ndarray, tasa_anual: float) -> ResultadoCaja:
    r = tasa_mensual(tasa_anual)
    descuento = (1 + r) ** -np.arange(flujos.shape[1])
    return ResultadoCaja(flujos=flujos, caja_acumulada=np.cumsum(flujos, axis=1),
                         van=flujos @ descuento)
