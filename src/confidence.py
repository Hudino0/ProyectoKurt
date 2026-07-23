"""
Intervalos de confianza para detectar degradación de performance
(Section 3.5 del paper).

Eq. (8) del paper:

    ff_{T/O} - z_{a/2} * sigma/sqrt(n)  <  FF_{T/O}  <  ff_{T/O} + z_{a/2} * sigma/sqrt(n)

donde ff es la predicción del MLP, FF el valor real del EEDB, sigma la
desviación típica y z el valor crítico de la Table 18.

Criterio de interpretación del paper: si el fuel flow T/O medido cae dentro del
intervalo, la performance del motor es normal; si cae fuera, hay indicio de
degradación.

Nota sobre `n`. Tomada literalmente, con n = 565 muestras, la semianchura sería
z*sigma/sqrt(565) ~ 0.005 kg/s, unas veinte veces más estrecha que las bandas
que se ven en las Figs. 10 y 11 del paper (~0.13 kg/s en torno a valores de
~2.5 kg/s). Esas figuras corresponden a n = 1, es decir, al intervalo de
predicción para **un motor individual**, que además es lo único coherente con el
objetivo declarado: decidir si *este* motor concreto está degradado. Se
implementan las dos lecturas y se usa la de n=1 para reproducir las figuras.
"""

import numpy as np

# Table 18 del paper: valores críticos de la normal estándar.
CRITICAL_VALUES = {
    0.99: 2.58,
    0.95: 1.96,
    0.90: 1.645,
}


def residual_sigma(y_true, y_predicted):
    """
    Desviación típica de los errores del modelo, que es el sigma de la Eq. (8).

    Se usa ddof=1 (estimador muestral insesgado), coherente con el resto del
    análisis estadístico del paper.
    """
    return float(np.std(y_true - y_predicted, ddof=1))


def confidence_bounds(y_predicted, sigma, level=0.99, n=1):
    """
    Cotas inferior y superior de la Eq. (8).

    `n` = 1 da el intervalo de predicción para un motor individual (el que
    reproduce las Figs. 10 y 11); `n` = tamaño de muestra da la lectura literal
    de la fórmula, un intervalo para la media.
    """
    if level not in CRITICAL_VALUES:
        raise ValueError(f"nivel no tabulado en Table 18: {level}")
    margin = CRITICAL_VALUES[level] * sigma / np.sqrt(n)
    return y_predicted - margin, y_predicted + margin


def flag_anomalies(y_true, y_predicted, sigma, level=0.99, n=1):
    """
    Marca los motores cuyo fuel flow T/O medido queda fuera del intervalo.

    Devuelve (mascara_booleana, cota_inferior, cota_superior). Un True indica
    posible degradación de performance en la fase de despegue.
    """
    lower, upper = confidence_bounds(y_predicted, sigma, level, n)
    outside = (y_true < lower) | (y_true > upper)
    return outside, lower, upper
