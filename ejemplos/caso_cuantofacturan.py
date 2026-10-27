"""Caso real y prospectivo: diagnostico de cuantofacturan.com (seccion 7.5).

Datos reales a 27 de octubre de 2026: precio del diagnostico 350 euros mas
IVA; construccion reaprovechada de herramientas propias, con coste
monetario de construccion practicamente nulo; una muestra gratuita en uso y
ninguna compra todavia (evidencia de nivel 2, que no actualiza theta).

Datos desconocidos que se tratan como sensibilidad declarada: horas propias
por diagnostico, compradores cualificados alcanzables en doce meses y coste
fijo mensual. No se presentan como observados.

    python ejemplos/caso_cuantofacturan.py           # tabla legible
    python ejemplos/caso_cuantofacturan.py --json    # resultados para el documento
"""

from __future__ import annotations

import json
import math
import sys

from tres_dimensiones import (CreenciaConversion, Mercado, OpcionEntrega, Umbrales, decidir,
                              perdida_asumible, tolerancia_riesgo_propietario,
                              valor_informacion)
from tres_dimensiones.calibracion import DELTA, DESVIACION_CENTRAL, EPSILON
from tres_dimensiones.regla import Aleatorios

SEMILLA = 20261027
N = 50_000
PRECIO = 350.0                       # dato real, euros sin IVA
CONSTRUCCION = 0.0                   # dato real: reaprovechamiento de herramientas propias
COLCHON = 60_000                     # mismo supuesto que la seccion 7
HORAS = (2.0, 4.0, 8.0)              # sensibilidad: horas propias por diagnostico
COMPRADORES = (30, 60, 120)          # sensibilidad: compradores cualificados en doce meses
FIJO_MENSUAL = 30.0                  # supuesto: alojamiento y servicios
PRUEBA = (5, 0.0)                    # cinco ofertas de pago a clientes ya captados, sin coste monetario


def caso(horas: float, compradores: int) -> dict:
    op = OpcionEntrega("Diagnostico cuantofacturan (nivel 2)", 2, construccion=CONSTRUCCION,
                       fijo_mensual=FIJO_MENSUAL, coste_variable=0.0, horas_contrato=horas,
                       horas_fijas_mes=2.0, coste_hora=45.0, capacidad_mes=60.0)
    mercado = Mercado(precio=PRECIO, compradores_horizonte=compradores)
    r = tolerancia_riesgo_propietario(COLCHON)
    u = Umbrales(delta=DELTA, epsilon=EPSILON,
                 perdida=perdida_asumible(r, EPSILON, capacidad_financiera=COLCHON))
    alea = Aleatorios.generar(SEMILLA, N, mercado.meses, DESVIACION_CENTRAL)
    creencia = CreenciaConversion()   # nivel 2: la muestra gratuita no actualiza theta
    dec = decidir(creencia, (op,), mercado, u, alea)
    vi = valor_informacion(creencia, PRUEBA[0], PRUEBA[1], (op,), mercado, u, alea)
    e = dec.evaluaciones[0]
    return {
        "horas": horas, "compradores": compradores,
        "theta_min": e.theta_min if math.isfinite(e.theta_min) else None,  # None: margen unitario no positivo
        "p_theta_bajo_min": e.prob_theta_bajo_min, "valor": e.resumen["valor_economico_medio"],
        "p10": e.resumen["valor_economico_p10"], "prob_exposicion": e.resumen["prob_exposicion"],
        "umbrales": e.umbrales, "decision": dec.elegida,
        "incremento_prueba": vi.incremento_politica, "error_prueba": vi.error_incremento,
        "ramas": [{"k": b.compras, "prob": b.probabilidad, "decision": b.decision} for b in vi.ramas],
    }


def main() -> None:
    filas = [caso(h, c) for h in HORAS for c in COMPRADORES]
    if "--json" in sys.argv:
        print(json.dumps(filas, indent=2, ensure_ascii=False, allow_nan=False))
        return
    for f in filas:
        print(f"horas {f['horas']:>3} | compradores {f['compradores']:>3} | theta_min {f['theta_min']:.3f} | "
              f"Pr(theta<min) {f['p_theta_bajo_min']:.3f} | valor {f['valor']:>9,.0f} (P10 {f['p10']:>8,.0f}) | "
              f"Pr(exp) {f['prob_exposicion']:.3f} | {f['decision']} | prueba +{f['incremento_prueba']:,.0f} "
              f"| ramas {''.join('L' if b['decision'].startswith('Diag') else 'd' for b in f['ramas'])}")


if __name__ == "__main__":
    main()
