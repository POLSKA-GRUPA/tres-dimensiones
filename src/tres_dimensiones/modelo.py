"""Opciones de entrega, mercado y simulacion de cada opcion.

Convenio temporal (meses enteros):
- t = 0: se paga la construccion (multiplicada por D).
- t = 1..T: llegan compradores, se firman y entregan contratos; en el mismo
  mes se pagan los costes variables, los costes fijos (multiplicados por D)
  y se consumen las horas propias.
- t + retraso: se cobra el precio de los contratos del mes t. Los cobros
  posteriores a T se mantienen en el calendario (derechos de cobro), sin
  nuevas ventas ni costes.

Llegadas: los compradores cualificados llegan segun un proceso de Poisson con
media N/T al mes y cada uno firma con probabilidad theta, de modo que los
contratos mensuales son Poisson de media theta N / T. N es por tanto una
intensidad esperada, no una poblacion fija de compradores distintos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .exposicion import caja_desde_flujos, tasa_mensual


def _no_negativo(nombre: str, valor: float) -> None:
    if not (math.isfinite(valor) and valor >= 0):
        raise ValueError(f"{nombre} debe ser finito y no negativo")


@dataclass(frozen=True, slots=True)
class Mercado:
    precio: float
    compradores_horizonte: float   # llegadas esperadas de compradores cualificados en T meses
    meses: int = 12
    retraso_cobro: int = 1
    tasa_anual: float = 0.10

    def __post_init__(self) -> None:
        _no_negativo("precio", self.precio)
        _no_negativo("compradores_horizonte", self.compradores_horizonte)
        if not isinstance(self.meses, int) or self.meses <= 0:
            raise ValueError("meses debe ser un entero positivo")
        if not isinstance(self.retraso_cobro, int) or self.retraso_cobro < 0:
            raise ValueError("retraso_cobro debe ser un entero no negativo")
        tasa_mensual(self.tasa_anual)


@dataclass(frozen=True, slots=True)
class OpcionEntrega:
    """Forma de entregar el resultado (escala de sencillez, niveles 1 a 5)."""

    nombre: str
    nivel_sencillez: int
    construccion: float          # coste estimado de construccion (t = 0)
    fijo_mensual: float          # mantenimiento y soporte estimados (t = 1..T)
    coste_variable: float        # coste monetario por contrato, pagado al entregar
    horas_contrato: float        # horas propias por contrato
    horas_fijas_mes: float       # horas propias fijas al mes
    coste_hora: float            # coste de oportunidad de la hora propia
    capacidad_mes: float         # horas propias disponibles al mes

    def __post_init__(self) -> None:
        for campo in ("construccion", "fijo_mensual", "coste_variable", "horas_contrato",
                      "horas_fijas_mes", "coste_hora", "capacidad_mes"):
            _no_negativo(campo, getattr(self, campo))
        if self.nivel_sencillez not in (1, 2, 3, 4, 5):
            raise ValueError("nivel_sencillez debe estar entre 1 y 5")

    def margen_unitario_economico(self, precio: float) -> float:
        return precio - self.coste_variable - self.horas_contrato * self.coste_hora

    def costes_fijos_economicos(self, meses: int) -> float:
        return self.construccion + meses * (self.fijo_mensual + self.horas_fijas_mes * self.coste_hora)

    def contratos_maximos_mes(self) -> float:
        """Contratos que caben en la capacidad de horas propias tras las horas fijas."""
        libres = self.capacidad_mes - self.horas_fijas_mes
        if libres < 0:
            return 0.0
        if self.horas_contrato == 0:
            return math.inf
        return float(math.floor(libres / self.horas_contrato))

    def capacidad_cubre(self, contratos_mes: float) -> bool:
        """La capacidad permite atender un volumen mensual dado de contratos."""
        return self.capacidad_mes >= self.horas_fijas_mes and contratos_mes <= self.contratos_maximos_mes()


@dataclass(frozen=True, slots=True)
class Simulacion:
    van_caja: np.ndarray
    valor_economico: np.ndarray
    caja_minima: np.ndarray
    demanda_perdida: np.ndarray   # fraccion de la demanda no atendida por capacidad

    def resumen(self, perdida: float) -> dict:
        q = np.quantile
        return {
            "van_caja_medio": float(self.van_caja.mean()),
            "valor_economico_medio": float(self.valor_economico.mean()),
            "valor_economico_p10": float(q(self.valor_economico, 0.10)),
            "valor_economico_p90": float(q(self.valor_economico, 0.90)),
            "caja_minima_p05": float(q(self.caja_minima, 0.05)),
            "prob_exposicion": float((self.caja_minima < -perdida).mean()),
            "demanda_perdida_media": float(self.demanda_perdida.mean()),
        }


def simular_opcion(opcion: OpcionEntrega, mercado: Mercado, theta: np.ndarray,
                   d: np.ndarray, u_ventas: np.ndarray) -> Simulacion:
    """Simula calendario de caja y valor economico de una opcion.

    theta y d tienen forma (n,); u_ventas, forma (n, T) con uniformes en (0, 1),
    comunes a todas las opciones y ramas para reducir el ruido de simulacion.
    """
    n, t = theta.shape[0], mercado.meses
    if theta.ndim != 1 or d.shape != (n,) or u_ventas.shape != (n, t):
        raise ValueError("Formas esperadas: theta (n,), d (n,), u_ventas (n, meses)")
    if not (np.isfinite(theta).all() and ((theta >= 0) & (theta <= 1)).all()):
        raise ValueError("theta debe estar en [0, 1]")
    if not (np.isfinite(d).all() and (d > 0).all()):
        raise ValueError("Los factores de desviacion deben ser finitos y positivos")
    if not ((u_ventas > 0) & (u_ventas < 1)).all():
        raise ValueError("Los uniformes deben estar en el intervalo abierto (0, 1)")

    demanda = stats.poisson.ppf(u_ventas, theta[:, None] * mercado.compradores_horizonte / t)
    contratos = np.minimum(demanda, opcion.contratos_maximos_mes())

    periodos = t + 1 + mercado.retraso_cobro
    flujos = np.zeros((n, periodos))
    flujos[:, 0] -= opcion.construccion * d
    flujos[:, 1:t + 1] -= (opcion.fijo_mensual * d)[:, None]
    flujos[:, 1:t + 1] -= contratos * opcion.coste_variable
    flujos[:, 1 + mercado.retraso_cobro:t + 1 + mercado.retraso_cobro] += contratos * mercado.precio
    caja = caja_desde_flujos(flujos, mercado.tasa_anual)

    horas = contratos * opcion.horas_contrato + opcion.horas_fijas_mes
    descuento = (1 + tasa_mensual(mercado.tasa_anual)) ** -np.arange(1, t + 1)
    coste_horas = (horas * opcion.coste_hora) @ descuento

    total = demanda.sum(axis=1)
    perdida = np.divide((demanda - contratos).sum(axis=1), total, out=np.zeros(n), where=total > 0)
    return Simulacion(van_caja=caja.van, valor_economico=caja.van - coste_horas,
                      caja_minima=caja.caja_minima, demanda_perdida=perdida)
