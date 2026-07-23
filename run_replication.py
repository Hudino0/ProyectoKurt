"""
Replicación completa de:

    Kurt, B. (2024). "Evaluation of aircraft engine performance during takeoff
    phase with machine learning methods". Neural Computing and Applications,
    36:19173-19190.  https://doi.org/10.1007/s00521-024-10220-3

Ejecuta de principio a fin el proceso descrito en el artículo:

    1. Carga del ICAO Engine Emissions Databank, ediciones 09/2019 y 07/2021
       -> Tables 4, 5 y 6
    2. Análisis de regresión lineal múltiple           -> Tables 7, 8, 9 y Eq. 2
    3. Gaussian process regression (3 kernels)         -> Tables 10, 11, Fig. 6
    4. Support vector machine (3 kernels)              -> Tables 12, 13, Fig. 7
    5. Multilayer perceptron (9 modelos)               -> Tables 14, 15, Fig. 9
    6. Comparación de los tres métodos                 -> Table 16
    7. Validación del mejor modelo sobre EEDB-07/2021  -> Table 17
    8. Intervalos de confianza al 99%                  -> Table 18, Figs. 10, 11

Todos los modelos están construidos desde cero (sin scikit-learn, TensorFlow ni
PyTorch); sólo se usa NumPy para álgebra lineal, openpyxl para leer los Excel y
matplotlib para las figuras.

Uso:   python run_replication.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import eedb_data as D
from confidence import CRITICAL_VALUES, flag_anomalies, residual_sigma
from figures import confidence_interval_figure, regression_and_mse_figure
from gpr import GaussianProcessRegressor
from metrics import all_metrics
from mlp import MultilayerPerceptron
from mlr import MultipleLinearRegression
from splits import MinMaxScaler, Standardizer, divide_random
from svr import SupportVectorRegressor, default_hyperparameters

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SEED = 0

FEATURE_NAMES = D.COLUMNS[:D.N_FEATURES]

# Salida simultánea por pantalla y a fichero.
_output_lines = []


def say(text=""):
    print(text)
    _output_lines.append(text)


def rule(title):
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------

def report_dataset(matrix, label, paper_stats):
    """Imprime la estadística descriptiva junto a la reportada en el paper."""
    say(f"\n{label}   (N replicado = {len(matrix)}, N del paper = {paper_stats['N']})")
    say(f"{'Parameter':<18}{'Min':>11}{'Max':>12}{'Mean':>12}{'Std Dev':>12}"
        f"{'|':>3}{'paper Mean':>12}{'paper Std':>12}")
    for (name, _, minimum, maximum, mean, std) in D.describe(matrix):
        pm, ps = paper_stats[name]
        say(f"{name:<18}{minimum:>11.5g}{maximum:>12.6g}{mean:>12.6g}{std:>12.6g}"
            f"{'|':>3}{pm:>12.6g}{ps:>12.6g}")


PAPER_TABLE_4 = {   # EEDB-09/2019
    "N": 565,
    "engine_type": (1.2248, 0.41781), "bypass_ratio": (6.1787, 2.39113),
    "press_ratio": (30.4341, 8.30142), "rated_output": (189.7491, 116.31773),
    "ambient_baro": (8004.3, 85867.3925), "ambient_temp": (286.9116, 10.30618),
    "ambient_humidity": (0.0069, 0.00366), "fuel_flow_to": (1.7095, 0.94789),
}

PAPER_TABLE_5 = {   # EEDB-07/2021
    "N": 773,
    "engine_type": (1.2096, 0.40727), "bypass_ratio": (6.5978, 2.64116),
    "press_ratio": (31.3034, 8.52827), "rated_output": (186.5325, 118.29263),
    "ambient_baro": (99.4889, 1.67661), "ambient_temp": (287.1879, 8.14578),
    "ambient_humidity": (0.0070, 0.00377), "fuel_flow_to": (1.6410, 0.96349),
}


# ---------------------------------------------------------------------------
# 2. Regresión lineal múltiple
# ---------------------------------------------------------------------------

def run_multiple_regression(X, y):
    rule("2. ANALISIS DE REGRESION LINEAL MULTIPLE  (Tables 7, 8, 9 y Eq. 2)")
    model = MultipleLinearRegression().fit(X, y, FEATURE_NAMES)

    say("\nTable 7 - Resumen del modelo")
    say(f"{'':22}{'replica':>12}{'paper':>12}")
    for key, paper in [("R", 0.990), ("R_square", 0.979),
                       ("adjusted_R_square", 0.979), ("std_error", 0.13733)]:
        say(f"{key:<22}{model.summary[key]:>12.5f}{paper:>12.5f}")

    say("\nTable 8 - ANOVA")
    a = model.anova
    say(f"{'':22}{'replica':>14}{'paper':>14}")
    for key, label, paper in [("ss_regression", "Sum sq. regression", 496.246),
                              ("ss_residual", "Sum sq. residual", 10.505),
                              ("ss_total", "Sum sq. total", 506.751),
                              ("df_regression", "df regression", 7),
                              ("df_residual", "df residual", 557),
                              ("df_total", "df total", 564),
                              ("ms_regression", "Mean sq. regression", 70.892),
                              ("ms_residual", "Mean sq. residual", 0.019),
                              ("F", "F", 3759.001)]:
        say(f"{label:<22}{a[key]:>14.4f}{paper:>14.4f}")
    say(f"{'Sig.':<22}{a['Sig']:>14.4f}{0.000:>14.4f}")

    say("\nTable 9 - Coeficientes")
    paper_coefficients = {
        "(Constant)": (1.302, 6.721, 0.000), "engine_type": (-0.49, -2.962, 0.003),
        "bypass_ratio": (-0.100, -23.782, 0.000),
        "press_ratio": (-7.308e-5, -0.046, 0.963),
        "rated_output": (0.009, 100.231, 0.000),
        "ambient_baro": (2.242e-7, 3.234, 0.001),
        "ambient_temp": (-0.002, -2.646, 0.008),
        "ambient_humidity": (-4.612, -2.346, 0.019),
    }
    say(f"{'':20}{'B':>13}{'Std.Err':>11}{'Beta':>9}{'t':>10}{'Sig.':>8}"
        f"{'|':>3}{'paper B':>13}{'paper t':>10}{'p Sig.':>8}")
    for row in model.coefficient_table:
        pb, pt, ps = paper_coefficients[row["name"]]
        say(f"{row['name']:<20}{row['beta']:>13.5g}{row['std_error']:>11.4g}"
            f"{row['standardized']:>9.3f}{row['t']:>10.3f}{row['Sig']:>8.3f}"
            f"{'|':>3}{pb:>13.5g}{pt:>10.3f}{ps:>8.3f}")

    say("\nEq. (2) replicada:")
    say("  " + model.equation(FEATURE_NAMES))
    say("\nEq. (2) del paper:")
    say("  Fuel Flow T/O = 1.302 - 0.49 x engine type - 0.1 x bypass ratio")
    say("                  - 7.308e-05 x press ratio + 0.009 x rated output")
    say("                  + 2.242e-07 x ambient baro - 0.002 x ambient temp")
    say("                  - 4.612 x ambient humidity")
    return model


# ---------------------------------------------------------------------------
# 3-5. Modelos de aprendizaje automático
# ---------------------------------------------------------------------------

def evaluate(predict, y, blocks):
    """Métricas en los tres bloques. `blocks` es un dict nombre -> índices."""
    return {name: all_metrics(y[idx], predict(idx)) for name, idx in blocks.items()}


def print_model_table(title, header, rows, paper_rows):
    """Tabla de comparación R/MSE en los tres bloques, junto a la del paper."""
    say(f"\n{title}")
    say(f"{header:<24}{'R train':>10}{'R valid':>10}{'R test':>10}"
        f"{'MSE train':>13}{'MSE valid':>13}{'MSE test':>13}")
    for name, r in rows:
        say(f"{name:<24}{r['train']['R']:>10.5f}{r['validation']['R']:>10.5f}"
            f"{r['test']['R']:>10.5f}{r['train']['MSE']:>13.5g}"
            f"{r['validation']['MSE']:>13.5g}{r['test']['MSE']:>13.5g}")
        p = paper_rows.get(name)
        if p:
            say(f"{'  (paper)':<24}{p[0]:>10.5f}{p[1]:>10.5f}{p[2]:>10.5f}"
                f"{p[3]:>13.5g}{p[4]:>13.5g}{p[5]:>13.5g}")


def run_gpr(X_standardized, y, blocks):
    rule("3. GAUSSIAN PROCESS REGRESSION  (Tables 10, 11 y Fig. 6)")
    paper = {
        "Exponential GPR":        (1.0, 0.98086, 0.9964, 1.302e-6, 0.02783, 0.0066),
        "Rational quadratic GPR": (0.99997, 0.98337, 0.9953, 5.188e-5, 0.02467, 0.0090),
        "Squared exponential GPR":(0.99996, 0.98245, 0.9856, 6.447e-5, 0.02660, 0.0272),
    }
    labels = {"exponential": "Exponential GPR",
              "rationalquadratic": "Rational quadratic GPR",
              "squaredexponential": "Squared exponential GPR"}

    results, models = [], {}
    for kernel, label in labels.items():
        start = time.time()
        model = GaussianProcessRegressor(kernel).fit(
            X_standardized[blocks["train"]], y[blocks["train"]], restarts=2, seed=SEED)
        metrics = evaluate(lambda i, m=model: m.predict(X_standardized[i]), y, blocks)
        say(f"  {label:<26} ajustado en {time.time() - start:5.1f}s")
        results.append((label, metrics))
        models[label] = model

    print_model_table("Table 10 - Comparacion de modelos GPR", "Kernel function",
                      results, paper)

    # El paper selecciona el mejor modelo por MSE en el conjunto de test.
    best_label, best_metrics = min(results, key=lambda r: r[1]["test"]["MSE"])
    best = models[best_label]
    say(f"\nMejor modelo GPR por MSE en test: {best_label}"
        f"   (el paper selecciona: Exponential GPR)")

    say("\nTable 11 - Estructura del modelo GPR seleccionado")
    say(f"  {'paper: Basis=Constant, Kernel scale=Auto, Sigma=0.0096, Beta=1.7404, Optimizer=Quasinewton'}")
    for key, value in best.structure().items():
        say(f"  {key:<30}{value if isinstance(value, str) else f'{value:.6g}'}")

    regression_and_mse_figure(
        [("Training", y[blocks["train"]], best.predict(X_standardized[blocks["train"]])),
         ("Validation", y[blocks["validation"]], best.predict(X_standardized[blocks["validation"]])),
         ("Test", y[blocks["test"]], best.predict(X_standardized[blocks["test"]]))],
        best_label, os.path.join(FIGURES_DIR, "fig06_gpr.png"))
    return best_label, best_metrics, best


def run_svm(X_standardized, y, blocks):
    rule("4. SUPPORT VECTOR MACHINE  (Tables 12, 13 y Fig. 7)")
    paper = {
        "Quadratic SVM": (0.99655, 0.97499, 0.99709, 0.00646, 0.04503, 0.00542),
        "Cubic SVM":     (0.99624, 0.44859, 0.99459, 0.00717, 1.46345, 0.01016),
        "Linear SVM":    (0.98939, 0.98859, 0.98906, 0.02005, 0.01735, 0.02062),
    }
    C, epsilon = default_hyperparameters(y[blocks["train"]])
    say(f"  Hiperparametros por defecto de fitrsvm sobre el bloque de entrenamiento:")
    say(f"    Box constraint = iqr(y)/1.349  = {C:.4f}   (paper: 1.2076)")
    say(f"    Epsilon        = iqr(y)/13.49  = {epsilon:.4f}   (paper: 0.1208)")

    results, models = [], {}
    for degree, label in [(2, "Quadratic SVM"), (3, "Cubic SVM"), (1, "Linear SVM")]:
        model = SupportVectorRegressor(degree=degree).fit(
            X_standardized[blocks["train"]], y[blocks["train"]])
        metrics = evaluate(lambda i, m=model: m.predict(X_standardized[i]), y, blocks)
        results.append((label, metrics))
        models[label] = model

    print_model_table("Table 12 - Comparacion de modelos SVM", "Kernel function",
                      results, paper)

    best_label, best_metrics = min(results, key=lambda r: r[1]["test"]["MSE"])
    best = models[best_label]
    say(f"\nMejor modelo SVM por MSE en test: {best_label}"
        f"   (el paper selecciona: Quadratic SVM)")

    say("\nTable 13 - Estructura del modelo SVM seleccionado")
    say("  paper: Function=Polynomial, Kernel scale=Auto, Box constraint=1.2076, "
        "Bias=1.7458, Epsilon=0.1208")
    for key, value in best.structure().items():
        say(f"  {key:<30}{value if isinstance(value, str) else f'{value:.6g}'}")

    regression_and_mse_figure(
        [("Training", y[blocks["train"]], best.predict(X_standardized[blocks["train"]])),
         ("Validation", y[blocks["validation"]], best.predict(X_standardized[blocks["validation"]])),
         ("Test", y[blocks["test"]], best.predict(X_standardized[blocks["test"]]))],
        best_label, os.path.join(FIGURES_DIR, "fig07_svm.png"))
    return best_label, best_metrics, best


def run_mlp(X, y, blocks):
    rule("5. MULTILAYER PERCEPTRON  (Tables 14, 15 y Fig. 9)")

    # Las redes de MATLAB reescalan entradas y objetivo a [-1,1] (mapminmax).
    x_scaler = MinMaxScaler().fit(X[blocks["train"]])
    y_scaler = MinMaxScaler().fit(y[blocks["train"]].reshape(-1, 1))
    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

    def to_original(z):
        return y_scaler.inverse_transform(z.reshape(-1, 1)).ravel()

    # 9 modelos: 3 por algoritmo, con semillas de inicialización distintas,
    # igual que el paper entrena varias redes por función de entrenamiento.
    plan = ([("trainlm", s) for s in (1, 2, 3)]
            + [("trainrp", s) for s in (1, 2, 3)]
            + [("traincgb", s) for s in (1, 2, 3)])
    method = {"trainlm": "train_lm", "trainrp": "train_rp", "traincgb": "train_cgb"}

    trained = []
    for index, (algorithm, seed) in enumerate(plan, start=1):
        network = MultilayerPerceptron(D.N_FEATURES, seed=seed)
        getattr(network, method[algorithm])(
            X_scaled[blocks["train"]], y_scaled[blocks["train"]],
            X_scaled[blocks["validation"]], y_scaled[blocks["validation"]],
            epochs=1000)
        metrics = evaluate(
            lambda i, n=network: to_original(n.predict(X_scaled[i])), y, blocks)
        trained.append({"index": index, "algorithm": algorithm,
                        "network": network, "metrics": metrics})

    # La Table 14 ordena los modelos por MSE creciente en el conjunto de test.
    trained.sort(key=lambda m: m["metrics"]["test"]["MSE"])

    say("\nTable 14 - Comparacion de los 9 modelos MLP (ordenados por MSE en test)")
    say(f"{'Model':<10}{'Algorithm':<12}{'R train':>10}{'R valid':>10}{'R test':>10}"
        f"{'MSE train':>12}{'MSE valid':>12}{'MSE test':>12}")
    for position, model in enumerate(trained, start=1):
        m = model["metrics"]
        say(f"{'Model ' + str(position):<10}{model['algorithm']:<12}"
            f"{m['train']['R']:>10.5f}{m['validation']['R']:>10.5f}{m['test']['R']:>10.5f}"
            f"{m['train']['MSE']:>12.5g}{m['validation']['MSE']:>12.5g}"
            f"{m['test']['MSE']:>12.5g}")
    say("\n  paper Model 1 'trainlm':  R = 0.9991 / 0.9970 / 0.9986"
        "   MSE = 0.00168 / 0.00427 / 0.0025488")

    best = trained[0]
    say(f"\nMejor modelo MLP: entrenado con '{best['algorithm']}'"
        f"   (el paper selecciona: trainlm)")

    say("\nTable 15 - Estructura del modelo MLP")
    for key, value in best["network"].structure().items():
        say(f"  {key:<16}{value}")
    say(f"  {'Training set':<16}{len(blocks['train'])} (70%)   "
        f"Validation {len(blocks['validation'])} (15%)   "
        f"Test {len(blocks['test'])} (15%)")
    say("  paper: capas 7-10-8-4-3-1, 'tansig'-'logsig'-'tansig'-'purelin', "
        "salida lineal, trainlm")

    network = best["network"]
    predict = lambda i: to_original(network.predict(X_scaled[i]))
    regression_and_mse_figure(
        [("Training", y[blocks["train"]], predict(blocks["train"])),
         ("Validation", y[blocks["validation"]], predict(blocks["validation"])),
         ("Test", y[blocks["test"]], predict(blocks["test"]))],
        "MLP", os.path.join(FIGURES_DIR, "fig09_mlp.png"))

    return best, x_scaler, y_scaler


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    rule("1. DATOS - ICAO ENGINE EMISSIONS DATABANK  (Tables 4 y 5)")
    matrix_2019, uids_2019 = D.load_2019()
    matrix_2021, uids_2021 = D.load_2021()
    D.save_csv(matrix_2019, os.path.join(RESULTS_DIR, "eedb_2019_preprocesado.csv"))
    D.save_csv(matrix_2021, os.path.join(RESULTS_DIR, "eedb_2021_preprocesado.csv"))

    report_dataset(matrix_2019, "Table 4 - EEDB-09/2019 (Issue 26B, 2019-09-20)",
                   PAPER_TABLE_4)
    say("\n  AVISO: el paper reporta para 2019 ambient baro max=1019394.5, "
        "mean=8004.3, std=85867.39")
    say("  y ambient temp min=144. Son valores fisicamente imposibles (la presion "
        "atmosferica")
    say("  ronda 95-104 kPa). Provienen de celdas corruptas en la importacion del "
        "autor, no del")
    say("  EEDB. Vease results/REPLICACION.md. La replica usa los valores correctos.")

    report_dataset(matrix_2021, "Table 5 - EEDB-07/2021 (Issue 28C, 2021-07-20)",
                   PAPER_TABLE_5)

    say("\nTable 6 - Muestras concretas del EEDB (comprobacion del preprocesamiento)")
    say(f"{'UID':<10}{'type':>6}{'bypass':>9}{'press':>8}{'rated':>9}"
        f"{'baro':>9}{'temp':>9}{'humidity':>11}{'fuelflow':>10}")
    for uid in ("1AS001", "4AL002", "1AA001"):
        if uid in uids_2021:
            row = matrix_2021[uids_2021.index(uid)]
            say(f"{uid:<10}{row[0]:>6.0f}{row[1]:>9.2f}{row[2]:>8.2f}{row[3]:>9.2f}"
                f"{row[4]:>9.3f}{row[5]:>9.2f}{row[6]:>11.5f}{row[7]:>10.3f}")
    say("  paper: 1AS001 -> 1, 2.64, 13.9, 15.6, 97.4, 286.5, 0.00765, 0.205")
    say("         1AA001 -> 2, 0.85, 18.4, 66.64, 103, 293, 0.01048, 1.670")

    # -- preparación de los bloques de datos -------------------------------
    X, y = D.split_xy(matrix_2019)
    train, validation, test = divide_random(len(y), seed=SEED)
    blocks = {"train": train, "validation": validation, "test": test}
    say(f"\nParticion 70/15/15: {len(train)} entrenamiento, "
        f"{len(validation)} validacion, {len(test)} test "
        f"(paper: 395 / 85 / 85)")

    standardizer = Standardizer().fit(X[train])
    X_standardized = standardizer.transform(X)

    # -- modelos -----------------------------------------------------------
    run_multiple_regression(X, y)
    gpr_label, gpr_metrics, _ = run_gpr(X_standardized, y, blocks)
    svm_label, svm_metrics, _ = run_svm(X_standardized, y, blocks)
    mlp_best, x_scaler, y_scaler = run_mlp(X, y, blocks)
    mlp_metrics = mlp_best["metrics"]

    # -- Table 16: comparación de los tres métodos -------------------------
    rule("6. COMPARACION DE LOS TRES MODELOS  (Table 16)")
    paper_table_16 = {
        ("Training", "Exponential GPR"): (1.3029736e-06, 5.33330e-04, 0.035374),
        ("Training", "MLP"):             (0.0016894, 0.00260, 2.113601),
        ("Training", "Quadratic SVM"):   (0.0064694, 0.06229, 5.385750),
        ("Validation", "MLP"):             (0.0042769, 0.0427, 3.524579),
        ("Validation", "Exponential GPR"): (0.0278303, 0.042242, 6.688691),
        ("Validation", "Quadratic SVM"):   (0.0450386, 0.084739, 10.415391),
        ("Test", "MLP"):             (0.0025488, 0.0314, 2.215001),
        ("Test", "Exponential GPR"): (0.0066794, 0.025113, 5.725879),
        ("Test", "Quadratic SVM"):   (0.0054221, 0.059196, 4.909031),
    }
    replica = {"MLP": mlp_metrics, gpr_label: gpr_metrics, svm_label: svm_metrics}
    paper_names = {"MLP": "MLP", gpr_label: "Exponential GPR",
                   svm_label: "Quadratic SVM"}

    say(f"{'Data set':<12}{'Model':<24}{'MSE':>13}{'MAE':>11}{'MAPE %':>11}"
        f"{'|':>3}{'paper MSE':>13}{'paper MAE':>11}{'paper MAPE':>11}")
    for block, block_label in [("train", "Training"), ("validation", "Validation"),
                               ("test", "Test")]:
        ordered = sorted(replica.items(), key=lambda kv: kv[1][block]["MSE"])
        for name, metrics in ordered:
            m = metrics[block]
            key = (block_label, paper_names.get(name, name))
            p = paper_table_16.get(key, (float("nan"),) * 3)
            say(f"{block_label:<12}{name:<24}{m['MSE']:>13.6g}{m['MAE']:>11.5g}"
                f"{m['MAPE']:>11.5g}{'|':>3}{p[0]:>13.6g}{p[1]:>11.5g}{p[2]:>11.5g}")
        say("")

    best_overall = min(replica.items(), key=lambda kv: kv[1]["test"]["MSE"])[0]
    say(f"Mejor modelo global por MSE en test: {best_overall}"
        f"   (el paper concluye: MLP)")

    # -- Table 17: generalización al conjunto de 2021 ----------------------
    rule("7. VALIDACION DEL MLP SOBRE LOS DOS CONJUNTOS  (Table 17)")
    network = mlp_best["network"]

    def mlp_predict(matrix):
        features = matrix[:, :D.N_FEATURES]
        scaled = network.predict(x_scaler.transform(features))
        return y_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()

    say(f"{'Data group':<34}{'Size':>10}{'MAE':>10}{'MSE':>10}{'R':>10}{'MAPE %':>10}")
    predictions = {}
    for label, matrix, paper in [
            ("EEDB (09/2019) gaseous emis. and smoke", matrix_2019,
             (0.0294, 0.0022, 0.99878, 2.341126)),
            ("EEDB (07/2021) gaseous emis. and smoke", matrix_2021,
             (0.0298, 0.0219, 0.98839, 2.897489))]:
        y_true = matrix[:, D.N_FEATURES]
        y_pred = mlp_predict(matrix)
        predictions[label] = (y_true, y_pred)
        m = all_metrics(y_true, y_pred)
        say(f"{label:<34}{len(matrix):>7} x 8{m['MAE']:>10.4f}{m['MSE']:>10.4f}"
            f"{m['R']:>10.5f}{m['MAPE']:>10.4f}")
        say(f"{'  (paper)':<34}{'':>10}{paper[0]:>10.4f}{paper[1]:>10.4f}"
            f"{paper[2]:>10.5f}{paper[3]:>10.4f}")

    # -- Table 18 y Figs. 10, 11: intervalos de confianza ------------------
    rule("8. INTERVALOS DE CONFIANZA AL 99%  (Table 18, Figs. 10 y 11)")
    say("\nTable 18 - Valores criticos")
    say(f"{'Level (%)':<12}{'z':>8}{'1 - alpha':>12}")
    for level, z in sorted(CRITICAL_VALUES.items(), reverse=True):
        say(f"{int(level * 100):<12}{z:>8.3f}{level:>12.2f}")

    y_train_true = y[train]
    y_train_pred = mlp_predict(matrix_2019[train])
    sigma = residual_sigma(y_train_true, y_train_pred)
    say(f"\nsigma (desviacion tipica del error del MLP en entrenamiento) = {sigma:.5f} kg/s")
    say(f"semianchura del intervalo al 99% = 2.58 * sigma = "
        f"{CRITICAL_VALUES[0.99] * sigma:.5f} kg/s")

    total_engines = 0
    for label, filename, dataset_label in [
            ("EEDB (09/2019) gaseous emis. and smoke", "fig10_ci_2019.png", "EEDB-09/2019"),
            ("EEDB (07/2021) gaseous emis. and smoke", "fig11_ci_2021.png", "EEDB-07/2021")]:
        y_true, y_pred = predictions[label]
        outside, lower, upper = flag_anomalies(y_true, y_pred, sigma, level=0.99, n=1)
        total_engines += len(y_true)
        say(f"\n{dataset_label}: {len(y_true)} motores, "
            f"{int(outside.sum())} fuera del intervalo al 99% "
            f"({100.0 * outside.mean():.2f}% -> posible degradacion de performance)")
        confidence_interval_figure(y_true, lower, upper, dataset_label,
                                   os.path.join(FIGURES_DIR, filename))

    say(f"\nTotal de motores evaluados: {total_engines}   (el paper evalua 1338)")

    say("\nFiguras generadas en results/figures/:")
    for name in sorted(os.listdir(FIGURES_DIR)):
        say(f"  {name}")

    with open(os.path.join(RESULTS_DIR, "salida_replicacion.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(_output_lines))
    say(f"\nSalida completa guardada en results/salida_replicacion.txt")


if __name__ == "__main__":
    main()
