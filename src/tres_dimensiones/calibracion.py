"""Valores por defecto documentados. Cada uno es sustituible por el decisor.

Fuentes (detalle en el documento de trabajo, seccion 6.4):
- Desviacion de costes: Flyvbjerg et al. (2022), 4.677 proyectos de TI con
  coste estimado y real; mediana de D igual a 1,0, media 1,8 y cola de
  Pareto con indice 2,3 por encima de D = 2. Para un primer producto sin
  historial propio se toma, como eleccion del metodo, mediana 1,25 y
  percentil 90 de 3,0, con prueba de estres de cola de Pareto.
- Tolerancia al riesgo: R = W / 6 (Howard, 1988, segun Smith, 2004).
- Perdida asumible: L = R ln(1/epsilon) (derivacion a partir de Delquie, 2008).
- epsilon = 0,05, convencion del 95 % de RiskMetrics (J.P. Morgan/Reuters, 1996).
- delta = b / (b + c), regla bayesiana de dos acciones.
- Distribucion inicial de la conversion: Beta(1, 1); sensibilidad con
  Jeffreys, Beta(0,5; 0,5). No se ha encontrado una tasa base publicada y
  fiable de conversion de propuesta a contrato.
"""

from .coste import DesviacionCoste

DESVIACION_BAJA = DesviacionCoste(mediana=1.00, p90=2.0)
DESVIACION_CENTRAL = DesviacionCoste(mediana=1.25, p90=3.0)
DESVIACION_ALTA = DesviacionCoste(mediana=1.50, p90=5.0)
DESVIACION_ESTRES = DesviacionCoste(mediana=1.25, p90=3.0, cola_pareto_alfa=2.3)

EPSILON = 0.05
EPSILON_SENSIBILIDAD = (0.01, 0.05, 0.10)
FRACCION_TOLERANCIA = 1 / 6
