# Replicación del artículo de Kurt (2024)

Réplica completa, en Python y **sin librerías de machine learning**, de:

> Kurt, B. (2024). *Evaluation of aircraft engine performance during takeoff phase with
> machine learning methods.* Neural Computing and Applications, 36:19173–19190.
> <https://doi.org/10.1007/s00521-024-10220-3>

El artículo predice el parámetro **Fuel Flow T/O** de motores aeronáuticos a partir de
7 variables del ICAO Engine Emissions Databank, comparando regresión lineal múltiple,
GPR, SVM y MLP, y usa el mejor modelo para detectar degradación de performance mediante
intervalos de confianza al 99%.

## Ejecutar

```bash
python run_replication.py
```

Tarda unos 4 minutos. Imprime cada tabla del artículo junto a la réplica, y genera
figuras y CSV en `results/`.

Dependencias: `numpy`, `openpyxl`, `matplotlib`.

```bash
python -m pip install numpy openpyxl matplotlib
```

## Qué está construido desde cero

Ninguno de los modelos usa scikit-learn, TensorFlow ni PyTorch. NumPy se emplea sólo como
librería de arrays y álgebra lineal básica.

| Componente | Implementación propia |
|---|---|
| Regresión lineal múltiple | Descomposición QR de Householder |
| p-valores de t y F | Función beta incompleta, fracción continua de Lentz |
| Gaussian process regression | Cholesky *right-looking* + BFGS sobre la verosimilitud marginal |
| Support vector regression | SMO con solución exacta del subproblema de 2 variables |
| Multilayer perceptron | Levenberg–Marquardt (jacobiano analítico), Rprop y gradiente conjugado Powell–Beale |

## Resultados

Todas las conclusiones del artículo se reproducen. Informe detallado con la comparación
tabla a tabla en **[`REPLICACION.md`](REPLICACION.md)**.

Incluye además un hallazgo sobre el artículo original: la Table 4 contiene valores
físicamente imposibles de presión barométrica y temperatura, cuyo origen se reconstruye
por aritmética a partir de los estadísticos publicados.

## Evidencia adicional: ediciones del EEDB posteriores al paper (2023-2026)

El paper sólo llega hasta EEDB-07/2021. `python run_new_edition_evidence.py <fichero.xlsx>`
aplica el mismo MLP a cualquier edición posterior indicada (pide el fichero si no se pasa
por argumento) como punto extra de generalización. Se probó con las ediciones de 2023,
2025 y la vigente de 2026 (MAPE 3.42-3.48%, "very good" de Lewis en las tres, igual que
2019 y 2021). Detalle en **[`REPLICACION.md` §6](REPLICACION.md)**.

## Estructura

```
data/                          EEDB: ediciones del paper (09/2019, 07/2021) + ediciones
                                posteriores (2023, 2025, vigente 03/2026)
src/                           módulos: datos, métricas, y los cuatro modelos
run_replication.py             pipeline completo del paper
run_new_edition_evidence.py    evidencia adicional sobre la edición vigente del EEDB
results/                       tablas, figuras y CSV preprocesados
tasks/todo.md                  plan y revisión final
tasks/lessons.md               lecciones del desarrollo
tasks/RESUMEN_SESION.md        mapa de orientación rápida del proyecto
```
