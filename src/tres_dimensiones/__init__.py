"""Metodo de decision empresarial en tres dimensiones.

Demanda pagadora, sencillez de entrega y retorno con exposicion acotada.
Acompana al documento de trabajo de Martin Alcantara (2026).
"""

from .coste import DesviacionCoste, desde_cuantiles
from .demanda import (CreenciaConversion, NivelEvidencia, conversion_equilibrio,
                      delta_por_perdidas, evidencia_actualiza_theta)
from .exposicion import perdida_asumible, tolerancia_riesgo_propietario
from .modelo import Mercado, OpcionEntrega
from .regla import (DESCARTAR, Aleatorios, Umbrales, decidir, valor_informacion,
                    valor_informacion_replicado)

__all__ = [
    "Aleatorios", "CreenciaConversion", "DESCARTAR", "DesviacionCoste", "Mercado",
    "NivelEvidencia", "OpcionEntrega", "Umbrales", "conversion_equilibrio", "decidir",
    "delta_por_perdidas", "desde_cuantiles", "evidencia_actualiza_theta",
    "perdida_asumible", "tolerancia_riesgo_propietario", "valor_informacion", "valor_informacion_replicado",
]
__version__ = "0.1.0"
