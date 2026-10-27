"""Opciones de entrega, mercado y simulacion de cada opcion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coste import DesviacionCoste
from .exposicion import caja_desde_flujos, tasa_mensual


@dataclass(frozen=True, slots=True)
class Mercado:
    precio: float
    compradores_horizonte: float   # compradores cualificados expuestos en el horizonte
    meses: int = 12
    retraso_cobro: int = 1
    tasa_anual: float = 0.10


@dataclass(frozen=True, slots=True)
class OpcionEntrega:
    """Forma de entregar el resultado (escala de sencillez, niveles 1 a 5)."""

    nombre: str
    nivel_sencillez: int
    construccion: float          # coste estimado de construccion
    fijo_mensual: float          # mantenimiento y soporte estimados
    coste_variable: float        # coste monetario por contrato
    horas_contrato: float        # horas propias por contrato
    horas_fijas_mes: float       # horas propias fijas al mes
    coste_hora: float            # coste de oportunidad de la hora propia
    capacidad_mes: float         # horas propias disponibles al mes

    def margen_unitario_economico(self, precio: float) -> float:
        return precio - self.coste_variable - self.horas_contrato * self.coste_hora

    def costes_fijos_economicos(self, meses: int) -> float:
        return self.construccion + meses * (self.fijo_mensual + self.horas_fijas_mes * self.coste_hora)

    def contratos_maximos_mes(self) -> float:
        """Contratos que caben en la capacidad de horas propias tras las horas fijas."""
        libres = self.capacidad_mes - self.horas_fijas_mes
        if libres <= 0:
            return 0.0
        if self.horas_contrato <= 0:
            return float("inf")
        return float(np.floor(libres / self.horas_contrato))

    def capacidad_cubre(self, contratos_mes: float) -> bool:
        """La capacidad permite atender un volumen mensual dado de contratos."""
        return contratos_mes <= self.contratos_maximos_mes()


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
    """Simula calendario de caja y valor economico.

    theta y d tienen una fila por simulacion; u_ventas son uniformes comunes
    (simulaciones x meses) para que las ramas del analisis preposterior usen
    numeros aleatorios comunes.
    """
    from scipy import stats

    t = mercado.meses
    expuestos_mes = mercado.compradores_horizonte / t
    demanda = stats.poisson.ppf(u_ventas, theta[:, None] * expuestos_mes)
    # la capacidad de horas propias limita los contratos atendidos (racionamiento)
    tope = opcion.contratos_maximos_mes()
    contratos = np.minimum(demanda, tope)
    n = theta.size
    cobros = np.zeros((n, t + 1 + mercado.retraso_cobro))
    cobros[:, mercado.retraso_cobro:mercado.retraso_cobro + t] = contratos * (mercado.precio - opcion.coste_variable)
    cobros = cobros[:, : t + 1]
    pagos = np.zeros((n, t + 1))
    pagos[:, 0] = opcion.construccion * d
    pagos[:, 1:] = (opcion.fijo_mensual * d)[:, None]
    caja = caja_desde_flujos(cobros - pagos, mercado.tasa_anual)
    horas = contratos * opcion.horas_contrato + opcion.horas_fijas_mes
    descuento = (1 + tasa_mensual(mercado.tasa_anual)) ** -np.arange(1, t + 1)
    coste_horas = (horas * opcion.coste_hora) @ descuento
    perdidas = np.where(demanda.sum(axis=1) > 0,
                        (demanda - contratos).sum(axis=1) / np.maximum(demanda.sum(axis=1), 1), 0.0)
    return Simulacion(van_caja=caja.van, valor_economico=caja.van - coste_horas,
                      caja_minima=caja.caja_minima, demanda_perdida=perdidas)
