# Modelo Matematico

## Objetivo

Asignar instalaciones a un conjunto fijo de tecnicos minimizando distancia total y desequilibrio de carga.

## Conjuntos y Parametros

- U: conjunto de instalaciones, con cardinal N.
- C: conjunto de tecnicos, con cardinal M.
- W_i: carga de trabajo asociada a la instalacion i.
- d_ic: distancia euclidea entre instalacion i y centro c.
- D_ref: referencia para normalizar el termino de distancia.
- E_ref: referencia para normalizar el termino de equilibrio.
- lambda en [0,1]: peso relativo de distancia frente a equidad.
- epsilon: tolerancia maxima de desequilibrio.

## Variables de Decision

- X_ic en {0,1}: vale 1 si la instalacion i se asigna al tecnico c.
- L_max >= 0: carga maxima entre tecnicos.
- L_min >= 0: carga minima entre tecnicos.

## Funcion Objetivo

Se minimiza una combinacion normalizada de eficiencia y equidad:

$$
\min Z = \lambda \cdot \frac{\sum_{i\in U}\sum_{c\in C} d_{ic}X_{ic}}{D_{ref}} + (1-\lambda) \cdot \frac{L_{max}-L_{min}}{E_{ref}}
$$

Interpretacion:

- lambda cercano a 1 prioriza proximidad.
- lambda cercano a 0 prioriza equilibrio de carga.

## Restricciones

1. Asignacion unica:

$$
\sum_{c\in C} X_{ic} = 1 \quad \forall i\in U
$$

2. Cota superior de carga por tecnico:

$$
\sum_{i\in U} W_iX_{ic} \le L_{max} \quad \forall c\in C
$$

3. Cota inferior de carga por tecnico:

$$
\sum_{i\in U} W_iX_{ic} \ge L_{min} \quad \forall c\in C
$$

4. Restriccion de equilibrio operativo:

$$
L_{max} - L_{min} \le \epsilon
$$

En el pipeline, epsilon se define como:

$$
\epsilon = \rho \cdot E_{ref}
$$

con rho configurable (por ejemplo, 0.15).

## Resolucion Hibrida (K-Means + MILP + ALA)

1. K-Means inicializa centros geograficos.
2. El MILP asigna instalaciones minimizando Z.
3. Se recalculan centroides por asignacion y se repite (ALA).
4. Se exploran multiples lambdas para construir el frente de Pareto.

## Salidas

- solucion.csv: asignacion final seleccionada.
- resumen_balance.csv: cargas por tecnico.
- metricas_pareto.csv: resultados por lambda.
- soluciones_maestro_pareto.csv: asignaciones completas por lambda.
- mapas y graficas para presentacion.
