"""
Reproducción de las figuras del paper con matplotlib.

    Figs. 6, 7, 9  - gráficos de regresión (Output vs Target) y de MSE
                     (medido vs predicho) para GPR, SVM y MLP
    Figs. 10, 11   - fuel flow T/O con intervalos de confianza al 99%
"""

import os

import matplotlib
matplotlib.use("Agg")           # backend sin ventana, para ejecutar sin GUI
import matplotlib.pyplot as plt
import numpy as np

from metrics import mse, r_coefficient, regression_line


def regression_and_mse_figure(splits, model_label, path):
    """
    Réplica de las Figs. 6, 7 y 9: una fila por bloque (entrenamiento,
    validación y test), con el gráfico de regresión a la izquierda y la
    comparación medido/predicho a la derecha.

    `splits` es una lista de (nombre, y_real, y_predicho).
    """
    figure, axes = plt.subplots(len(splits), 2, figsize=(11, 3.4 * len(splits)))
    if len(splits) == 1:
        axes = axes[None, :]

    for row, (name, y_true, y_pred) in enumerate(splits):
        # --- izquierda: dispersión Output vs Target con recta de ajuste ---
        left = axes[row, 0]
        left.plot(y_true, y_pred, "k.", markersize=4, label="Data")
        slope, intercept = regression_line(y_true, y_pred)
        line = np.array([y_true.min(), y_true.max()])
        left.plot(line, slope * line + intercept, "b-", linewidth=1.2, label="Fit")
        left.plot(line, line, "k--", linewidth=0.9, label="Y = T")
        left.set_xlabel("Target")
        left.set_ylabel(f"Output ~= {slope:.2f}*Target + {intercept:.3f}")
        left.set_title(f"{name}: R={r_coefficient(y_true, y_pred):.5f}")
        left.legend(fontsize=7, loc="upper left")
        left.grid(alpha=0.25)

        # --- derecha: valor medido frente a valor predicho, muestra a muestra ---
        right = axes[row, 1]
        right.plot(y_true, ".", color="tab:blue", markersize=4,
                   label="Measured Fuel Flow T/O (EEDB)")
        right.plot(y_pred, ".", color="tab:red", markersize=4,
                   label=f"Predicted Fuel Flow T/O ({model_label})")
        right.set_xlabel("Samples")
        right.set_ylabel("Fuel Flow T/O")
        right.set_title(f"{name} results, MSE:{mse(y_true, y_pred):.6g}")
        right.legend(fontsize=7, loc="upper left")
        right.grid(alpha=0.25)

    figure.suptitle(f"Regression and MSE plots of the {model_label} model")
    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def confidence_interval_figure(y_true, lower, upper, dataset_label, path):
    """
    Réplica de las Figs. 10 y 11: fuel flow T/O medido junto a las cotas del
    intervalo de confianza al 99%, motor a motor.
    """
    figure, axis = plt.subplots(figsize=(12, 4.5))
    index = np.arange(len(y_true))
    axis.plot(index, upper, "-", color="red", linewidth=0.8, label="Upper bounder")
    axis.plot(index, lower, "-", color="black", linewidth=0.8, label="Lower bounder")
    axis.plot(index, y_true, ".", color="tab:blue", markersize=3,
              label="Fuel Flow T/O from EEDB")
    # Los motores fuera del intervalo son los candidatos a degradación.
    outside = (y_true < lower) | (y_true > upper)
    axis.plot(index[outside], y_true[outside], "x", color="darkorange",
              markersize=5, linewidth=0.8,
              label=f"Outside interval ({outside.sum()})")
    axis.set_xlabel(f"The ICAO Aircraft Engine Emissions Databank Samples ({dataset_label})")
    axis.set_ylabel("Fuel Flow T / O")
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
