"""
Evidencia adicional: generalizacion del MLP a ediciones del EEDB no usadas en el paper.

El articulo de Kurt (2024) solo llega hasta EEDB-07/2021 (Table 17). Este script
extiende ese mismo analisis a cualquier edicion posterior del EEDB que se le indique
(archivo .xlsx en `data/`, con el mismo formato tabular plano que 2021 y la vigente),
para comprobar si el MLP replicado sigue generalizando a datos mas recientes. Puede
evaluar varias ediciones en una sola corrida (por ejemplo, todas las disponibles entre
2021 y la actual) reutilizando el mismo MLP ya entrenado.

No modifica ningun modelo: reutiliza tal cual `eedb_data.load_current()`, `splits.py`
y `mlp.py` ya construidos y verificados en `run_replication.py`. Solo entrena el MLP
(el modelo ganador del articulo) con el mismo procedimiento -- 3 inicializaciones de
`trainlm`, se elige la de menor MSE en test -- y anade cada edicion indicada como un
grupo de validacion mas, igual que la Table 17 y la Table 18 del paper.

Uso:
    python run_new_edition_evidence.py                         (pide el/los fichero(s))
    python run_new_edition_evidence.py EEDB_v32_2026.xlsx
    python run_new_edition_evidence.py EEDB_2022.xlsx EEDB_2023.xlsx EEDB_2024.xlsx
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import eedb_data as D
from confidence import CRITICAL_VALUES, flag_anomalies, residual_sigma
from figures import confidence_interval_figure
from metrics import all_metrics
from mlp import MultilayerPerceptron
from splits import MinMaxScaler, Standardizer, divide_random

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SEED = 0


def prompt_for_editions():
    """
    Pide por consola el fichero (o ficheros) de EEDB a evaluar, cuando no se pasan
    como argumentos de linea de comandos. Lista los .xlsx ya presentes en data/ como
    ayuda, pero acepta cualquier nombre de fichero que exista ahi.
    """
    existentes = sorted(f for f in os.listdir(D.DATA_DIR) if f.lower().endswith(".xlsx"))
    print("Ficheros .xlsx disponibles en data/:")
    for name in existentes:
        print(f"  {name}")
    raw = input("\nEdicion(es) a evaluar (nombres separados por coma): ")
    filenames = [n.strip() for n in raw.split(",") if n.strip()]
    if not filenames:
        raise SystemExit("No se indico ningun fichero.")
    return filenames


def train_best_mlp(X, y, blocks):
    """
    Entrena el MLP 3 veces con 'trainlm' (distintas semillas de inicializacion) y
    devuelve el de menor MSE en test -- el mismo criterio de seleccion que usa
    `run_replication.py` para elegir "el" MLP entre los 9 modelos del paper.
    """
    x_scaler = MinMaxScaler().fit(X[blocks["train"]])
    y_scaler = MinMaxScaler().fit(y[blocks["train"]].reshape(-1, 1))
    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

    def to_original(z):
        return y_scaler.inverse_transform(z.reshape(-1, 1)).ravel()

    best_network, best_mse = None, float("inf")
    for seed in (1, 2, 3):
        network = MultilayerPerceptron(D.N_FEATURES, seed=seed)
        network.train_lm(
            X_scaled[blocks["train"]], y_scaled[blocks["train"]],
            X_scaled[blocks["validation"]], y_scaled[blocks["validation"]],
            epochs=1000)
        pred_test = to_original(network.predict(X_scaled[blocks["test"]]))
        mse_test = all_metrics(y[blocks["test"]], pred_test)["MSE"]
        if mse_test < best_mse:
            best_network, best_mse = network, mse_test

    return best_network, x_scaler, y_scaler


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    filenames = sys.argv[1:] if len(sys.argv) > 1 else prompt_for_editions()

    print("Cargando EEDB-09/2019 (entrenamiento del MLP, como en el paper)...")
    matrix_2019, _ = D.load_2019()
    print(f"  EEDB-09/2019: N = {len(matrix_2019)}")

    # Mismo entrenamiento que en el paper: MLP sobre la particion 70/15/15 de 2019.
    # Se entrena una sola vez y se reutiliza para evaluar todas las ediciones pedidas.
    X, y = D.split_xy(matrix_2019)
    train, validation, test = divide_random(len(y), seed=SEED)
    blocks = {"train": train, "validation": validation, "test": test}
    network, x_scaler, y_scaler = train_best_mlp(X, y, blocks)

    def mlp_predict(matrix):
        features = matrix[:, :D.N_FEATURES]
        scaled = network.predict(x_scaler.transform(features))
        return y_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()

    y_train_pred = mlp_predict(matrix_2019[train])
    sigma = residual_sigma(y[train], y_train_pred)

    print("\nGeneralizacion del MLP a cada edicion indicada (analogo Table 17):")
    print(f"{'Data group':<34}{'N':>6}{'MAE':>10}{'MSE':>10}{'R':>10}{'MAPE %':>10}"
          f"{'fuera CI 99%':>14}")

    for filename in filenames:
        label = os.path.splitext(os.path.basename(filename))[0]
        matrix, _ = D.load_current(filename)
        D.save_csv(matrix, os.path.join(RESULTS_DIR, f"{label}_preprocesado.csv"))

        y_true = matrix[:, D.N_FEATURES]
        y_pred = mlp_predict(matrix)
        m = all_metrics(y_true, y_pred)
        outside, lower, upper = flag_anomalies(y_true, y_pred, sigma, level=0.99, n=1)

        print(f"{label:<34}{len(matrix):>6}{m['MAE']:>10.4f}{m['MSE']:>10.4f}"
              f"{m['R']:>10.5f}{m['MAPE']:>10.4f}"
              f"{int(outside.sum()):>9} ({100.0 * outside.mean():.2f}%)")
        clase_lewis = "very good" if m["MAPE"] < 10 else "aceptable/pobre"
        print(f"  MAPE = {m['MAPE']:.2f}% -> clase de Lewis: {clase_lewis}"
              "  (paper: MAPE < 10% en 2019 y 2021)")

        figure_path = os.path.join(FIGURES_DIR, f"fig_ci_{label}.png")
        confidence_interval_figure(y_true, lower, upper, label, figure_path)
        print(f"  figura:  {os.path.relpath(figure_path, ROOT)}")
        print(f"  csv:     results/{label}_preprocesado.csv\n")


if __name__ == "__main__":
    main()
