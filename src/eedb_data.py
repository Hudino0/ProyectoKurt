"""
Carga y preprocesamiento del ICAO Engine Emissions Databank (EEDB).

Replica el preprocesamiento del artículo de Kurt (2024). El artículo no describe
explícitamente cómo transformó el Excel crudo en su matriz de datos, pero la
Table 6 del paper ("Randomly selected sample values") permite deducirlo sin
ambigüedad. Ejemplo, motor 1AS001:

    paper:  engine type=1, bypass=2.64, press=13.9, rated=15.6,
            baro=97.4, temp=286.5, humidity=0.00765, fuel flow T/O=0.205
    excel:  Eng Type=TF, B/P=2.64, PR=13.9, Thrust=15.6,
            Baro Min=96.7 Max=98.1, Temp Min=285 Max=288,
            Hum Min=0.0066 Max=0.0087, Fuel Flow T/O=0.205

    (96.7+98.1)/2 = 97.4  ✓   (285+288)/2 = 286.5  ✓
    (0.0066+0.0087)/2 = 0.00765  ✓

De donde:
  * `Engine type`  : TF -> 1, MTF -> 2  (turbofan / mixed-turbofan)
  * ambiente       : punto medio de las columnas Min y Max
  * el resto       : se toman tal cual

Ediciones utilizadas (recuperadas del Internet Archive, ya que EASA sólo publica
la edición vigente):
  * Issue 26B, 2019-09-20  ->  "EEDB-09/2019" del paper
  * Issue 28C, 2021-07-20  ->  "EEDB-07/2021" del paper
"""

import os
import re

import numpy as np
import openpyxl

# Directorio raíz del proyecto (un nivel por encima de src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

# Nombres de las 8 columnas de la matriz final: 7 predictores + 1 objetivo.
# El orden es el de las Tables 4 y 5 del paper.
COLUMNS = [
    "engine_type",      # x1  TF=1, MTF=2
    "bypass_ratio",     # x2  adimensional
    "press_ratio",      # x3  adimensional
    "rated_output",     # x4  kN
    "ambient_baro",     # x5  kPa
    "ambient_temp",     # x6  K
    "ambient_humidity", # x7  kg/kg
    "fuel_flow_to",     # y   kg/s  <- objetivo
]
N_FEATURES = 7

# Mapeo del tipo de motor a código numérico (Table 4: min=1, max=2).
ENGINE_TYPE_CODE = {"TF": 1.0, "MTF": 2.0}


def _to_float(value):
    """Convierte una celda a float; devuelve None si no es un número."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _midpoint(value):
    """
    Punto medio de una celda de condición ambiental.

    El formato *nuevo* del EEDB guarda Min y Max en dos columnas separadas y esta
    función recibe cada una por separado. El formato *antiguo* (edición 2019)
    guarda el rango como texto en una sola celda, con formas como:
        "96.7-98.1"      "285 - 288"      ".0066-.0087"      "101.3"      "-"
    Se extraen todos los números presentes y se promedian, de modo que un rango
    da su punto medio y un valor suelto se devuelve intacto.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    numbers = re.findall(r"\d*\.?\d+", str(value))
    if not numbers:
        return None                      # celdas como "-" o vacías
    values = [float(n) for n in numbers]
    return sum(values) / len(values)


def _read_sheet(path, sheet_name):
    """Devuelve (cabecera, filas) de una hoja de cálculo."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[sheet_name]
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()
    return rows[0], rows[1:]


def _build(records):
    """
    Filtra y convierte la lista de registros crudos en una matriz (n, 8).

    Criterio de inclusión (el que reproduce los N del paper):
      * el motor es TF o MTF (los únicos tipos con código 1 y 2 en Table 4),
      * todas las condiciones ambientales y el fuel flow T/O están presentes.

    El bypass ratio recibe un trato especial, dictado por las propias tablas del
    paper: la Table 4 (2019) reporta un mínimo de 0.64, mientras que la Table 5
    (2021) reporta 0.00. La diferencia se explica por cómo cada edición codifica
    el dato ausente para los mismos 8 motores (1PW020, 1PW021, 1PW031, 1PW040,
    5PW076, 8PW086, 8PW087, 8PW088):

      * edición 2019 (formato antiguo): la celda contiene el marcador "-",
        es decir, el fabricante declara explícitamente que no lo reporta;
      * edición 2021 (formato nuevo): la celda está simplemente vacía.

    Se aplica por tanto una única regla sobre el contenido literal de la celda:
    marcador explícito -> fila inválida; celda vacía -> se imputa 0.0. Esto
    reproduce el mínimo de 0.64 en 2019 y el de 0.00 en 2021.
    """
    matrix, uids = [], []
    for rec in records:
        code = ENGINE_TYPE_CODE.get(rec["engine_type"])
        if code is None:
            continue
        if rec["bypass_missing"]:
            continue                                  # marcador "-" explícito
        bypass = 0.0 if rec["bypass_ratio"] is None else rec["bypass_ratio"]
        required = [rec[k] for k in
                    ("press_ratio", "rated_output", "ambient_baro",
                     "ambient_temp", "ambient_humidity", "fuel_flow_to")]
        if any(v is None for v in required):
            continue
        matrix.append([code, bypass] + required)
        uids.append(rec["uid"])
    return np.array(matrix, dtype=float), uids


def load_2019():
    """
    Carga EEDB Issue 26B (2019-09-20) = "EEDB-09/2019" del paper.

    Esta edición usa el formato antiguo, con cabeceras en varias filas y los
    rangos ambientales guardados como texto, por lo que se accede por índice de
    columna en lugar de por nombre. Los índices se verificaron leyendo las filas
    de cabecera (filas 0-3 de la hoja).
    """
    path = os.path.join(DATA_DIR, "EEDB_v26B_2019.xlsx")
    _, rows = _read_sheet(path, "ICAO databank")
    records = []
    # La hoja tiene 4 filas de cabecera; _read_sheet ya consumió la primera,
    # de modo que aquí quedan 3 por descartar.
    for row in rows[3:]:
        records.append({
            "uid":              row[0],
            "engine_type":      row[3],
            "bypass_ratio":     _to_float(row[4]),
            # celda no vacía pero no numérica => marcador "-" de "no reportado"
            "bypass_missing":   row[4] is not None and _to_float(row[4]) is None,
            "press_ratio":      _to_float(row[5]),
            "rated_output":     _to_float(row[6]),
            "fuel_flow_to":     _to_float(row[80]),
            "ambient_baro":     _midpoint(row[90]),
            "ambient_temp":     _midpoint(row[91]),
            "ambient_humidity": _midpoint(row[92]),
        })
    return _build(records)


def load_2021():
    """
    Carga EEDB Issue 28C (2021-07-20) = "EEDB-07/2021" del paper.

    Esta edición ya usa el formato tabular plano, con una columna por campo y
    Min/Max en columnas separadas, de modo que se accede por nombre de cabecera.
    """
    path = os.path.join(DATA_DIR, "EEDB_v28C_2021.xlsx")
    header, rows = _read_sheet(path, "Gaseous Emissions and Smoke")
    col = {name: i for i, name in enumerate(header) if name}

    def pair(row, lo, hi):
        """Punto medio de un par de columnas Min/Max."""
        a, b = _to_float(row[col[lo]]), _to_float(row[col[hi]])
        return None if a is None or b is None else (a + b) / 2.0

    records = []
    for row in rows:
        records.append({
            "uid":              row[col["UID No"]],
            "engine_type":      row[col["Eng Type"]],
            "bypass_ratio":     _to_float(row[col["B/P Ratio"]]),
            "bypass_missing":   (row[col["B/P Ratio"]] is not None
                                 and _to_float(row[col["B/P Ratio"]]) is None),
            "press_ratio":      _to_float(row[col["Pressure Ratio"]]),
            "rated_output":     _to_float(row[col["Rated Thrust (kN)"]]),
            "fuel_flow_to":     _to_float(row[col["Fuel Flow T/O (kg/sec)"]]),
            "ambient_baro":     pair(row, "Ambient Baro Min (kPa)",
                                          "Ambient Baro Max (kPa)"),
            "ambient_temp":     pair(row, "Ambient Temp Min (K)",
                                          "Ambient Temp Max (K)"),
            "ambient_humidity": pair(row, "Humidity Min (kg/kg)",
                                          "Humidity Max (kg/kg)"),
        })
    return _build(records)


def load_current(filename="EEDB_v32_2026.xlsx", sheet_name="Gaseous Emissions and Smoke"):
    """
    Carga cualquier edición del EEDB en el formato tabular plano (el que usan todas
    las ediciones desde 26B/2019 en adelante, incluida la vigente).

    `filename` es el nombre del fichero dentro de `data/` (o una ruta completa). Por
    defecto apunta a Issue 32 (03/2026), la vigente en julio de 2026, pero se puede
    pasar cualquier otra edición descargada (p. ej. las de 2022-2025) sin tocar esta
    función ni ninguna otra: basta con guardar el .xlsx en `data/` y pasar su nombre.
    La hoja se llama igual en todas ellas ("Gaseous Emissions and Smoke") y tiene las
    mismas cabeceras, por lo que la lógica de columnas es idéntica a `load_2021()`.
    Se mantiene como función separada (en vez de generalizar `load_2021`) para no
    tocar código ya verificado contra las Tables 4/5 del paper.

    Sirve para generar evidencia adicional de generalización más allá de 2021,
    replicando el mismo análisis de las Tables 17-18 sobre datos más recientes.
    """
    path = filename if os.path.isabs(filename) else os.path.join(DATA_DIR, filename)
    header, rows = _read_sheet(path, sheet_name)
    col = {name: i for i, name in enumerate(header) if name}

    def pair(row, lo, hi):
        """Punto medio de un par de columnas Min/Max."""
        a, b = _to_float(row[col[lo]]), _to_float(row[col[hi]])
        return None if a is None or b is None else (a + b) / 2.0

    records = []
    for row in rows:
        records.append({
            "uid":              row[col["UID No"]],
            "engine_type":      row[col["Eng Type"]],
            "bypass_ratio":     _to_float(row[col["B/P Ratio"]]),
            "bypass_missing":   (row[col["B/P Ratio"]] is not None
                                 and _to_float(row[col["B/P Ratio"]]) is None),
            "press_ratio":      _to_float(row[col["Pressure Ratio"]]),
            "rated_output":     _to_float(row[col["Rated Thrust (kN)"]]),
            "fuel_flow_to":     _to_float(row[col["Fuel Flow T/O (kg/sec)"]]),
            "ambient_baro":     pair(row, "Ambient Baro Min (kPa)",
                                          "Ambient Baro Max (kPa)"),
            "ambient_temp":     pair(row, "Ambient Temp Min (K)",
                                          "Ambient Temp Max (K)"),
            "ambient_humidity": pair(row, "Humidity Min (kg/kg)",
                                          "Humidity Max (kg/kg)"),
        })
    return _build(records)


def split_xy(matrix):
    """Separa la matriz en predictores X (n, 7) y objetivo y (n,)."""
    return matrix[:, :N_FEATURES], matrix[:, N_FEATURES]


def describe(matrix):
    """
    Estadística descriptiva por columna, en el formato de las Tables 4 y 5.

    Devuelve una lista de tuplas (nombre, n, min, max, media, desviación típica).
    La desviación típica es la muestral (ddof=1), que es la que reporta SPSS y la
    que usa el paper.
    """
    stats = []
    for i, name in enumerate(COLUMNS):
        column = matrix[:, i]
        stats.append((name, len(column), column.min(), column.max(),
                      column.mean(), column.std(ddof=1)))
    return stats


def save_csv(matrix, path):
    """Guarda la matriz preprocesada como CSV con cabecera."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", header=",".join(COLUMNS),
               comments="", fmt="%.6g")
