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
    """R = W / 6.

    Adaptacion del autor: Howard (1988) propone como regla practica una
    tolerancia al riesgo de un sexto del patrimonio, y Smith (2004) la
    interpreta para un gestor con toda su riqueza en la empresa. Aqui se
    aplica al colchon liquido W del propietario de una empresa pequena, lo
    que es una eleccion normativa sustituible por el decisor.
    """
    if not (math.isfinite(colchon_liquido) and colchon_liquido > 0):
        raise ValueError("El colchon liquido debe ser finito y positivo")
    return colchon_liquido / 6


def perdida_asumible(tolerancia_riesgo: float, epsilon: float,
                     capacidad_financiera: float | None = None) -> float:
    """L = min(R ln(1/epsilon), capacidad financiera).

    Con utilidad exponencial de tolerancia R, una apuesta que pierde L con
    probabilidad epsilon y gana G en otro caso es aceptable si
    epsilon e^(L/R) + (1 - epsilon) e^(-G/R) <= 1. Cuando G crece sin limite,
    la perdida maxima aceptable tiende a R ln(1/epsilon) (Delquie, 2008, da
    el caso epsilon = 1/20, L aproximadamente 3R). La formula es por tanto una
    cota superior de la preferencia, y la capacidad financiera impone el
    segundo limite: nadie puede perder mas de lo que tiene.
    """
    if not (math.isfinite(tolerancia_riesgo) and tolerancia_riesgo > 0):
        raise ValueError("La tolerancia al riesgo debe ser finita y positiva")
    if not (math.isfinite(epsilon) and 0 < epsilon < 1):
        raise ValueError("epsilon debe estar en (0, 1)")
    preferencia = tolerancia_riesgo * math.log(1 / epsilon)
    if capacidad_financiera is None:
        return preferencia
    if not (math.isfinite(capacidad_financiera) and capacidad_financiera > 0):
        raise ValueError("La capacidad financiera debe ser finita y positiva")
    return min(preferencia, capacidad_financiera)


def tasa_mensual(tasa_anual: float) -> float:
    if not (math.isfinite(tasa_anual) and tasa_anual > -1):
        raise ValueError("La tasa anual debe ser finita y mayor que -1")
    return (1 + tasa_anual) ** (1 / 12) - 1


@dataclass(frozen=True, slots=True)
class ResultadoCaja:
    flujos: np.ndarray            # (simulaciones, periodos)
    caja_acumulada: np.ndarray    # (simulaciones, periodos)
    van: np.ndarray               # (simulaciones,)

    @property
    def caja_minima(self) -> np.ndarray:
        return np.minimum(self.caja_acumulada.min(axis=1), 0.0)

    def prob_exposicion(self, perdida: float) -> float:
        if not (math.isfinite(perdida) and perdida >= 0):
            raise ValueError("La perdida asumible debe ser finita y no negativa")
        return float((self.caja_minima < -perdida).mean())

    @property
    def financiacion_adicional(self) -> np.ndarray:
        """Equivalente a la financiacion adicional de un motor determinista de caja."""
        return -self.caja_minima


def caja_desde_flujos(flujos: np.ndarray, tasa_anual: float) -> ResultadoCaja:
    if flujos.ndim != 2 or not np.isfinite(flujos).all():
        raise ValueError("Los flujos deben ser una matriz finita (simulaciones, periodos)")
    r = tasa_mensual(tasa_anual)
    descuento = (1 + r) ** -np.arange(flujos.shape[1])
    return ResultadoCaja(flujos=flujos, caja_acumulada=np.cumsum(flujos, axis=1),
                         van=flujos @ descuento)
