"""Adaptador opcional a un motor determinista de caja externo.

Algunos motores de decision calculan, para un escenario con calendario
mensual de cobros y pagos, la financiacion adicional necesaria: el peor
saldo negativo del recorrido. Esa magnitud coincide con la exposicion de la
dimension 3. Este adaptador permite delegar en un motor asi el calculo de
cada trayectoria simulada, sin duplicar reglas de caja ya auditadas.

El adaptador no importa ningun motor concreto: recibe una funcion que, dada
una secuencia de pares (cobros, pagos) por mes, devuelve la financiacion
adicional. Asi la libreria no depende de software privado.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

MotorCaja = Callable[[Sequence[tuple[float, float]]], float]


def financiacion_por_motor(cobros: np.ndarray, pagos: np.ndarray, motor: MotorCaja) -> np.ndarray:
    """Aplica el motor a cada trayectoria (filas) y devuelve la financiacion adicional."""
    if cobros.shape != pagos.shape:
        raise ValueError("cobros y pagos deben tener la misma forma")
    return np.array([motor(list(zip(c.tolist(), p.tolist()))) for c, p in zip(cobros, pagos)])


def motor_referencia(meses: Sequence[tuple[float, float]]) -> float:
    """Motor de referencia: max(0, -minimo del saldo acumulado)."""
    saldo, minimo = 0.0, 0.0
    for cobro, pago in meses:
        saldo += cobro - pago
        minimo = min(minimo, saldo)
    return max(0.0, -minimo)
