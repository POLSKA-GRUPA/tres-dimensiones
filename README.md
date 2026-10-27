# tres-dimensiones

Implementación de referencia del método de decisión empresarial en tres dimensiones descrito en el documento de trabajo *Cuando construir deja de ser el problema: decisión empresarial en tres dimensiones* (Martín Alcantara, 2026, DOI [10.5281/zenodo.23687822](https://doi.org/10.5281/zenodo.23687822)).

*Reference implementation of a three-dimensional business decision method (paying demand, delivery simplicity, return with bounded exposure). English summary below.*

## Qué hace

Dada una o varias formas de entregar un resultado, el método decide entre lanzar una de ellas, descartar o seguir investigando:

1. **Demanda pagadora.** La conversión contractual θ se modela con una distribución Beta que solo actualizan las compras reales (nivel 4 de la escalera de evidencia).
2. **Sencillez de entrega.** La construcción y los costes fijos monetarios se multiplican por un factor de desviación D, lognormal en el cuerpo y con prueba de estrés de cola de Pareto, calibrado con la evidencia sobre proyectos de tecnologías de la información. Los costes variables y las horas propias se modelan aparte, y la capacidad de horas raciona los contratos atendidos.
3. **Retorno y exposición.** Se simula el calendario de caja y se exige que la probabilidad de que la caja acumulada caiga por debajo de la pérdida asumible −L no supere ε.

Las tres dimensiones se combinan con una **regla conjuntiva**: cada opción debe superar los tres umbrales por separado; entre las admisibles se elige la de mayor valor económico esperado (VAN de caja menos coste de oportunidad de las horas propias). Para valorar una prueba de mercado se enumeran sus resultados posibles con la predictiva beta-binomial exacta y se vuelve a decidir con cada posterior; θ, D y las ventas se integran por simulación con números aleatorios comunes, y cada cifra se acompaña de su error de Monte Carlo. Se informan dos magnitudes: el **incremento preposterior de la política conjuntiva**, que puede ser negativo porque las restricciones dependen de la posterior, y el **EVSI clásico** de la política de valor esperado sin restricciones.

Convenio temporal: construcción en t = 0; contratos, costes variables, costes fijos y horas en t = 1..T; cobro en t + retraso, también después de T. Los compradores cualificados llegan según un proceso de Poisson de media N/T al mes y firman con probabilidad θ.

## Valores por defecto

| Parámetro | Valor | Origen |
| --- | --- | --- |
| Desviación de coste D | mediana 1,25; percentil 90 de 3,0 | Elección del método a partir de Flyvbjerg et al. (2022) y Jørgensen (2014) |
| Prueba de estrés | cola de Pareto empalmada en D = 2, densidad ∝ D^−2,3 (forma de supervivencia 1,3; varianza infinita) | Flyvbjerg et al. (2022) |
| Tolerancia al riesgo R | una sexta parte del colchón líquido W | Adaptación de Howard (1988) y Smith (2004) |
| Pérdida asumible L | min(R ln(1/ε), W) | Cota superior a partir de Delquié (2008), limitada por la capacidad financiera |
| ε | 0,05 | Convención del 95 % de RiskMetrics (1996) |
| δ | 0,25 (sensibilidad 0,10 a 0,50) | Restricción normativa; el umbral bayesiano implícito b/(b+c) se informa como diagnóstico |
| Distribución inicial de θ | Beta(1, 1); sensibilidad con Beta(0,5; 0,5) | Sin tasa base publicada fiable |

Todos son sustituibles por el decisor. Los casos de ejemplo son hipotéticos.

## Uso

```bash
pip install -e ".[dev]"
python -m pytest
python ejemplos/caso_documento.py          # reproduce las cifras del documento
python ejemplos/caso_documento.py --json   # salida para tablas
```

```python
from tres_dimensiones import (CreenciaConversion, Mercado, OpcionEntrega, Umbrales, decidir,
                              perdida_asumible, tolerancia_riesgo_propietario)
from tres_dimensiones.calibracion import DESVIACION_CENTRAL
from tres_dimensiones.regla import Aleatorios

creencia = CreenciaConversion().actualizar(ofertas=12, compras=3)
mercado = Mercado(precio=900, compradores_horizonte=150)
opcion = OpcionEntrega("Servicio asistido", 2, construccion=3_000, fijo_mensual=150,
                       coste_variable=120, horas_contrato=4, horas_fijas_mes=4,
                       coste_hora=45, capacidad_mes=60)
L = perdida_asumible(tolerancia_riesgo_propietario(60_000), 0.05, capacidad_financiera=60_000)
decision = decidir(creencia, (opcion,), mercado, Umbrales(delta=0.25, epsilon=0.05, perdida=L),
                   Aleatorios.generar(20261027, 100_000, mercado.meses, DESVIACION_CENTRAL))
print(decision.elegida, round(decision.valor))
```

## Motores de caja externos

`tres_dimensiones.adaptadores.motor_caja` permite delegar el cálculo de la financiación adicional de cada trayectoria en un motor determinista de caja ya auditado, sin dependencias de software privado.

## Límites

Es una propuesta normativa sin evaluación empírica todavía. No es asesoramiento financiero ni jurídico. Los umbrales expresan preferencias del decisor y deben fijarse antes de decidir.

## English summary

Python implementation of a conjunctive, three-dimensional decision rule for small firms when building is cheap and choosing is expensive: Beta-binomial demand updating restricted to actual purchases, lognormal cost overruns with a Pareto stress tail, a chance constraint on cumulative cash (affordable loss), and exact preposterior value of sample information. Defaults are documented and replaceable. Requires Python ≥ 3.10, NumPy and SciPy.

## Cita

Cita el software (véase `CITATION.cff`) y el documento de trabajo que lo describe (DOI 10.5281/zenodo.23687822). Licencia Apache-2.0. © 2026 Kenyi Martín Alcantara.
