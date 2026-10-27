"""Valores por defecto documentados. Cada uno es sustituible por el decisor.

Fuentes (detalle en el documento de trabajo, seccion 6.5):
- Desviacion de costes: Flyvbjerg et al. (2022), 4.677 proyectos de TI con
  coste estimado y real; mediana de D igual a 1,0, media 1,8; densidad de la
  cola proporcional a D ** (-2,3) por encima de D = 2. Jorgensen (2014) cifra
  el sobrecoste medio de software en torno al 30 %. Para un primer producto
  sin historial propio se toma, como eleccion del metodo, mediana 1,25 y
  percentil 90 de 3,0, con prueba de estres de cola de Pareto.
- Tolerancia al riesgo: R = W / 6, adaptacion de Howard (1988) y Smith (2004).
- Perdida asumible: L = min(R ln(1/epsilon), W), cota a partir de Delquie (2008).
- epsilon = 0,05, convencion del 95 % de RiskMetrics (J.P. Morgan/Reuters, 1996).
- delta = 0,25 como restriccion normativa; sensibilidad 0,10 a 0,50.
- Distribucion inicial de la conversion: Beta(1, 1); sensibilidad con
  Jeffreys, Beta(0,5; 0,5). No se ha encontrado una tasa base publicada y
  fiable de conversion de propuesta a contrato.
"""

from .coste import DesviacionCoste

DESVIACION_BAJA = DesviacionCoste(mediana=1.00, p90=2.0)
DESVIACION_CENTRAL = DesviacionCoste(mediana=1.25, p90=3.0)
DESVIACION_ALTA = DesviacionCoste(mediana=1.50, p90=5.0)
DESVIACION_ESTRES = DesviacionCoste(mediana=1.25, p90=3.0, alfa_densidad=2.3, umbral_cola=2.0)

EPSILON = 0.05
DELTA = 0.25
DELTA_SENSIBILIDAD = (0.10, 0.25, 0.50)
EPSILON_SENSIBILIDAD = (0.01, 0.05, 0.10)
FRACCION_TOLERANCIA = 1 / 6
