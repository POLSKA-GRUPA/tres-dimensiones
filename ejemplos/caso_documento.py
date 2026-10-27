"""Casos hipoteticos del documento de trabajo (seccion 7). Reproducen sus cifras.

    python ejemplos/caso_documento.py          # informe legible
    python ejemplos/caso_documento.py --json   # resultados para tablas

Todos los datos son hipoteticos. Los parametros por defecto proceden de
tres_dimensiones.calibracion y se justifican en la seccion 6.4.
"""

from __future__ import annotations

import json
import math
import sys

from tres_dimensiones import (CreenciaConversion, Mercado, OpcionEntrega, Umbrales, decidir,
                              perdida_asumible, tolerancia_riesgo_propietario, valor_informacion)
from tres_dimensiones.calibracion import (DELTA, DESVIACION_ALTA, DESVIACION_BAJA,
                                          DESVIACION_CENTRAL, DESVIACION_ESTRES, EPSILON)
from tres_dimensiones.regla import Aleatorios, valor_informacion_replicado

SEMILLA = 20261027
REPLICAS = (101, 202, 303, 404, 505)
N = 100_000
COLCHON = 60_000  # caja liquida del decisor (hipotetica)

ASISTIDO = OpcionEntrega("Servicio asistido (nivel 2)", 2, construccion=3_000, fijo_mensual=150,
                         coste_variable=120, horas_contrato=4.0, horas_fijas_mes=4,
                         coste_hora=45, capacidad_mes=60)
AUTOMATIZADO = OpcionEntrega("Producto automatizado (nivel 4)", 4, construccion=24_000,
                             fijo_mensual=900, coste_variable=40, horas_contrato=0.5,
                             horas_fijas_mes=10, coste_hora=45, capacidad_mes=60)
LIGERO = OpcionEntrega("Producto ligero (nivel 4)", 4, construccion=6_000, fijo_mensual=400,
                       coste_variable=40, horas_contrato=0.5, horas_fijas_mes=8,
                       coste_hora=45, capacidad_mes=60)
MEDIO = OpcionEntrega("Producto intermedio (nivel 4)", 4, construccion=9_000, fijo_mensual=400,
                      coste_variable=40, horas_contrato=0.5, horas_fijas_mes=8,
                      coste_hora=45, capacidad_mes=60)

MERCADO_A = Mercado(precio=900, compradores_horizonte=150)
MERCADO_BC = Mercado(precio=900, compradores_horizonte=200)


def umbrales(epsilon: float = EPSILON, delta: float = DELTA) -> Umbrales:
    r = tolerancia_riesgo_propietario(COLCHON)
    return Umbrales(delta=delta, epsilon=epsilon,
                    perdida=perdida_asumible(r, epsilon, capacidad_financiera=COLCHON))


def evaluar(creencia, opciones, mercado, desviacion=DESVIACION_CENTRAL, epsilon=EPSILON,
            delta=DELTA, prueba: tuple[int, float] | None = None, semilla: int = SEMILLA) -> dict:
    alea = Aleatorios.generar(semilla, N, mercado.meses, desviacion)
    u = umbrales(epsilon, delta)
    dec = decidir(creencia, opciones, mercado, u, alea)
    out = {
        "creencia": [creencia.alpha, creencia.beta], "media": creencia.media,
        "ic90": creencia.intervalo(0.90), "L": u.perdida, "epsilon": u.epsilon, "delta": u.delta,
        "opciones": [{"opcion": e.opcion, "theta_min": e.theta_min,
                      "p_theta_bajo_min": e.prob_theta_bajo_min, "delta_bayes": e.delta_bayes,
                      **e.resumen, "umbrales": e.umbrales, "admisible": e.admisible}
                     for e in dec.evaluaciones],
        "decision": dec.elegida, "valor": dec.valor, "error_mc": dec.error_mc,
        "decision_valor_esperado": dec.elegida_valor_esperado,
    }
    if prueba:
        vi = valor_informacion(creencia, prueba[0], prueba[1], opciones, mercado, u, alea)
        rep = valor_informacion_replicado(creencia, prueba[0], prueba[1], opciones, mercado, u,
                                          desviacion, REPLICAS, N // 4)
        out["replicas"] = {"semillas": list(REPLICAS), "n_por_replica": N // 4,
                           "incremento_medio": rep.incremento_medio,
                           "error_replicas": rep.error_replicas,
                           "evsi_clasico_medio": rep.evsi_clasico_medio,
                           "decisiones_estables": rep.decisiones_estables,
                           "indeterminado": rep.indeterminado,
                           "realizar": rep.realizar}
        out["prueba"] = {"ofertas": vi.ofertas, "coste": vi.coste,
                         "incremento_politica": vi.incremento_politica,
                         "evsi_clasico": vi.evsi_clasico,
                         "error_incremento": vi.error_incremento, "error_evsi": vi.error_evsi,
                         "realizar": vi.realizar,
                         "ramas": [{"k": r.compras, "prob": r.probabilidad, "decision": r.decision,
                                    "valor": r.valor} for r in vi.ramas]}
    return out


def casos() -> dict:
    a = CreenciaConversion().actualizar(12, 3)
    return {
        "A_asistido_vs_automatizado": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, prueba=(5, 400)),
        "B_producto_ligero_evidencia_debil": evaluar(CreenciaConversion().actualizar(5, 0), (LIGERO,),
                                                     MERCADO_BC, prueba=(5, 500)),
        "C_producto_intermedio_coste_central": evaluar(CreenciaConversion().actualizar(6, 2), (MEDIO,),
                                                       MERCADO_BC),
        "C_producto_intermedio_coste_acotado": evaluar(CreenciaConversion().actualizar(6, 2), (MEDIO,),
                                                       MERCADO_BC, desviacion=DESVIACION_BAJA),
        "sensibilidad_A": {
            "desviacion_baja": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, DESVIACION_BAJA),
            "desviacion_alta": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, DESVIACION_ALTA),
            "estres_pareto": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, DESVIACION_ESTRES),
            "epsilon_0.01": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, epsilon=0.01),
            "epsilon_0.10": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, epsilon=0.10),
            "delta_0.10": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, delta=0.10),
            "delta_0.50": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, delta=0.50),
            "semilla_alternativa": evaluar(a, (ASISTIDO, AUTOMATIZADO), MERCADO_A, semilla=7),
            "inicial_jeffreys": evaluar(CreenciaConversion(0.5, 0.5).actualizar(12, 3),
                                        (ASISTIDO, AUTOMATIZADO), MERCADO_A),
        },
    }


def _fmt(x) -> str:
    return "no definido (varianza infinita)" if x is None else f"{x:,.0f}"


def _sin_nan(obj):
    """Diagnosticos indefinidos (NaN) como null para producir JSON estandar."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sin_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sin_nan(v) for v in obj]
    return obj


def _imprimir(nombre: str, r: dict) -> None:
    lo, hi = r["ic90"]
    print(f"\n== {nombre} ==\nBeta({r['creencia'][0]:g}, {r['creencia'][1]:g}) media {r['media']:.3f} "
          f"IC90 {lo:.3f}-{hi:.3f} | L {r['L']:,.0f} EUR | epsilon {r['epsilon']}")
    for o in r["opciones"]:
        print(f"  {o['opcion']}: theta_min {o['theta_min']:.3f}, Pr(theta<min) {o['p_theta_bajo_min']:.3f}, "
              f"delta bayes {'n.d.' if math.isnan(o['delta_bayes']) else format(o['delta_bayes'], '.3f')}, valor econ. {o['valor_economico_medio']:,.0f} "
              f"(P10 {o['valor_economico_p10']:,.0f}; P90 {o['valor_economico_p90']:,.0f}), "
              f"Pr(exposicion) {o['prob_exposicion']:.3f}, {o['umbrales']}")
    print(f"  DECISION: {r['decision']} ({r['valor']:,.0f} EUR, error MC {_fmt(r['error_mc'])}); "
          f"criterio de valor esperado sin restricciones: {r['decision_valor_esperado']}")
    if "prueba" in r:
        p = r["prueba"]
        for rama in p["ramas"]:
            print(f"    k={rama['k']} p={rama['prob']:.3f} -> {rama['decision']} ({rama['valor']:,.0f})")
        rp = r["replicas"]
        print(f"  Replicas independientes ({len(rp['semillas'])} x {rp['n_por_replica']:,}): incremento medio "
              f"{rp['incremento_medio']:,.0f} (error {rp['error_replicas']:,.0f}), decisiones estables "
              f"{rp['decisiones_estables']}, realizar {rp['realizar']}")
        print(f"  Incremento de la politica {p['incremento_politica']:,.0f} (EVSI clasico {p['evsi_clasico']:,.0f}; "
              f"errores MC {_fmt(p['error_incremento'])} y {_fmt(p['error_evsi'])}) frente a coste "
              f"{p['coste']:,.0f}: realizar {p['realizar']}")


if __name__ == "__main__":
    resultados = casos()
    if "--json" in sys.argv:
        print(json.dumps(_sin_nan(resultados), indent=2, ensure_ascii=False, allow_nan=False))
    else:
        for nombre, r in resultados.items():
            if nombre == "sensibilidad_A":
                for sub, rr in r.items():
                    _imprimir(f"Sensibilidad A: {sub}", rr)
            else:
                _imprimir(nombre, r)
