"""
Criterios de error del artículo (Section 3, Eqs. 5-7) y coeficiente R.

Notación del paper:
    FF_{T/O}  = valor real tomado del EEDB
    ff_{T/O}  = valor predicho por el modelo
    n         = número de observaciones
"""

import numpy as np


def mse(y_true, y_pred):
    """Error cuadrático medio.  Eq. (5):  MSE = (1/n) * sum (FF - ff)^2"""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Error absoluto medio.  Eq. (6):  MAE = (1/n) * sum |FF - ff|"""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    """
    Error porcentual absoluto medio.  Eq. (7):

        MAPE = (100/n) * sum |(FF - ff) / FF|

    Según Lewis (ref. 49 del paper), MAPE < 10% clasifica el modelo como
    "muy bueno". El EEDB no contiene fuel flows nulos (el mínimo es 0.148 kg/s),
    así que la división es siempre segura.
    """
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))


def r_coefficient(y_true, y_pred):
    """
    Coeficiente de correlación de Pearson entre valores reales y predichos.

    Es la "R" que MATLAB muestra en los gráficos de regresión (Figs. 6, 7, 9)
    y la que reportan las Tables 10, 12 y 14.
    """
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(a * b) / denominator)


def all_metrics(y_true, y_pred):
    """Devuelve un diccionario con las cuatro medidas de una sola pasada."""
    return {
        "R":    r_coefficient(y_true, y_pred),
        "MSE":  mse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def regression_line(y_true, y_pred):
    """
    Ajuste `Output = a*Target + b` que MATLAB dibuja en los gráficos de
    regresión. Se usa para reproducir los subtítulos de las Figs. 6, 7 y 9.
    """
    a, b = np.polyfit(y_true, y_pred, 1)
    return float(a), float(b)
