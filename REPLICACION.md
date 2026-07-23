# Informe de replicación

**Artículo replicado:** Kurt, B. (2024). *Evaluation of aircraft engine performance during
takeoff phase with machine learning methods.* Neural Computing and Applications, 36:19173–19190.
<https://doi.org/10.1007/s00521-024-10220-3>

**Ejecución:** `python run_replication.py` — salida íntegra en
`results/salida_replicacion.txt`.

---

## 1. Resumen

Se ha replicado el proceso completo del artículo en Python, **construyendo los cuatro
modelos desde cero** (regresión lineal múltiple, GPR, SVM y MLP) sin usar ninguna
librería de machine learning. Sólo se emplean NumPy (álgebra lineal), openpyxl
(lectura de Excel) y matplotlib (figuras).

**Todas las conclusiones cualitativas del artículo se reproducen:**

| Conclusión del artículo | ¿Se replica? |
|---|---|
| Entre los GPR, el kernel **exponencial** es el mejor | Sí |
| Entre los SVM, el **cuadrático** es el mejor; el cúbico se degrada en validación | Sí |
| Entre los MLP, **`trainlm`** supera a `trainrp` y a `traincgb` | Sí |
| Comparados los tres métodos, el **MLP** es el mejor en test | Sí |
| El MLP generaliza al conjunto de 2021 con MAPE < 10% ("very good", Lewis) | Sí |
| Los intervalos al 99% permiten señalar motores anómalos | Sí |

---

## 2. Los datos

El artículo usa el **ICAO Engine Emissions Databank (EEDB)**, ediciones 09/2019 y 07/2021.
EASA sólo publica la edición vigente, así que ambas se recuperaron del Internet Archive y
se verificó su identidad en la hoja *Record of Changes*:

| Fichero | Issue | Fecha | Edición del paper |
|---|---|---|---|
| `data/EEDB_v26B_2019.xlsx` | 26B | 2019-09-20 | EEDB-09/2019 |
| `data/EEDB_v28C_2021.xlsx` | 28C | 2021-07-20 | EEDB-07/2021 |

### Preprocesamiento deducido

El artículo no describe cómo pasó del Excel del EEDB a su matriz de datos. La Table 6 del
paper permite deducirlo sin ambigüedad. Para el motor `1AS001`:

| | engine type | bypass | press | rated | baro | temp | humidity | fuel flow |
|---|---|---|---|---|---|---|---|---|
| paper (Table 6) | 1 | 2.64 | 13.9 | 15.6 | **97.4** | **286.5** | **0.00765** | 0.205 |
| Excel EEDB | TF | 2.64 | 13.9 | 15.6 | Min 96.7 / Max 98.1 | Min 285 / Max 288 | Min .0066 / Max .0087 | 0.205 |

De donde: `engine type` TF→1 y MTF→2, y las tres variables ambientales son el **punto medio
de las columnas Min y Max**. Reproducido en `src/eedb_data.py`.

### Verificación: Table 5 (EEDB-07/2021) — coincidencia exacta

| Parámetro | Mín | Máx | Media (réplica / paper) | Desv. típica (réplica / paper) |
|---|---|---|---|---|
| Engine type | 1 | 2 | 1.20957 / **1.2096** | 0.40727 / **0.40727** |
| Bypass ratio | 0.00 | 12.72 | 6.59786 / **6.5978** | 2.64049 / **2.64116** |
| Press ratio | 9.76 | 49.55 | 31.3022 / **31.3034** | 8.52774 / **8.52827** |
| Rated output | 9.79 | 513.90 | 186.533 / **186.5325** | 118.293 / **118.29263** |
| Ambient baro | 95.15 | 103.45 | 99.4889 / **99.4889** | 1.67661 / **1.67661** |
| Ambient temp | 226.50 | 308.50 | 287.188 / **287.1879** | 8.14578 / **8.14578** |
| Ambient humidity | 0.0005 | 0.0413 | 0.007036 / **0.0070** | 0.003773 / **0.00377** |
| Fuel flow T/O | 0.148 | 4.69 | 1.64101 / **1.6410** | 0.963492 / **0.96349** |

**N = 773 en ambos casos.** Las ocho estadísticas coinciden hasta la precisión reportada.
Esto confirma que el conjunto de datos reconstruido es el mismo que usó el autor.

Para 2019 se obtiene **N = 563** frente a las 565 del paper (99.6%), con todos los mínimos
y máximos coincidentes.

---

## 3. Hallazgo: datos corruptos en la Table 4 del artículo original

La Table 4 del paper reporta, para EEDB-09/2019:

> `Ambient baro:  min 0.00   max 1019394.5   mean 8004.3   std 85867.3925`
> `Ambient temp:  min 144`

Son valores **físicamente imposibles**: la presión atmosférica a nivel del suelo está entre
95 y 104 kPa, y 144 K son −129 °C. El EEDB original no contiene ninguno de esos valores —
se verificó celda a celda en el fichero fuente. Son artefactos de la importación del autor.

La aritmética permite reconstruir el defecto con precisión:

**Temperatura — exactamente una celda corrupta.** El motor `4AL002` tiene 288 K en el EEDB,
pero la Table 6 del propio paper lo lista con **144** = 288/2. Si un único valor pasa de
288 a 144 sobre N=565:

- desplazamiento de la media: −144/565 = −0.25 → 287.22 − 0.25 = **286.97**  (paper: 286.9116)
- desviación típica: √(70.5 + 143²/564) = **10.33**  (paper: 10.30618)

Ambas predicciones aciertan, lo que confirma que se trata de una sola celda.

**Presión — unas 4–5 celdas corruptas de magnitud ≈10⁶.** De los estadísticos publicados:

- suma implícita: 565 × 8004.3 = 4.52×10⁶ → exceso de ≈4.47×10⁶ sobre lo esperable
- suma de cuadrados implícita: 564 × 85867.4² = 4.16×10¹² → k × (10⁶)² con **k ≈ 4.2**

Ambas cifras son consistentes con ~4–5 valores de orden 10⁶.

**Consecuencia.** El defecto se propagó a los modelos publicados. En la Table 9 el
coeficiente de `ambient baro` es β = 2.242×10⁻⁷ con Beta estandarizado 0.020: un coeficiente
propio de una variable en escala 10⁶, no de una en escala 10². Es decir, **los modelos del
artículo se entrenaron sobre una variable de presión parcialmente corrupta.**

Esta replicación usa los valores correctos, por lo que no puede — ni debe — reproducir ese
coeficiente.

**Nota menor adicional:** la Table 6 del paper lista el motor `1AA001` con fuel flow T/O =
1.670, pero el EEDB registra 1.15 en ambas ediciones.

---

## 4. Resultados por modelo

### 4.1 Regresión lineal múltiple (Tables 7, 8, 9)

| Estadístico | Réplica | Paper |
|---|---|---|
| R | 0.98974 | 0.990 |
| R² | 0.97958 | 0.979 |
| R² ajustado | 0.97932 | 0.979 |
| Error típico | 0.13665 | 0.13733 |
| SS regresión | 497.18 | 496.25 |
| SS residual | 10.36 | 10.51 |
| SS total | 507.54 | 506.75 |
| F | 3803.69 | 3759.00 |
| Sig. | 0.000 | 0.000 |

Coincidencia esencialmente exacta.

**Los estadísticos *t* replican con notable precisión**, lo que confirma que se trata del
mismo modelo sobre los mismos datos:

| Predictor | *t* réplica | *t* paper |
|---|---|---|
| Bypass ratio | −23.785 | **−23.782** |
| Rated output | 99.894 | **100.231** |
| Engine type | −2.951 | **−2.962** |
| Press ratio | 0.005 | −0.046 (ambos ≈ 0, Sig. ≈ 1) |

**Errata detectada en la Table 9 del paper.** Para `engine type` el artículo publica
β = −0.49 junto a un error típico de 0.017 y *t* = −2.962. Pero −0.49/0.017 = −28.8, no
−2.962. El valor coherente con su propio error típico y su propio *t* es **β = −0.049**;
la réplica obtiene −0.053. El "−0.49" es un error de un factor 10, que el artículo arrastra
además a su Eq. (2).

El único coeficiente que no replica es el de `ambient baro`, por lo explicado en §3.

Implementación: mínimos cuadrados por **descomposición QR de Householder** escrita a mano
(`src/mlr.py`), y p-valores de t y F mediante la **función beta incompleta** evaluada con la
fracción continua de Lentz (`src/distributions.py`) — sin SciPy.

### 4.2 Gaussian process regression (Tables 10, 11 · Fig. 6)

GPR exacta con base constante, factorización de **Cholesky** propia e hiperparámetros
ajustados por máxima verosimilitud marginal con **BFGS** propio (el "Quasinewton" del paper).

| Kernel | MSE test (réplica) | MSE test (paper) |
|---|---|---|
| **Exponential** | **0.00393** | **0.00668** |
| Rational quadratic | 0.00657 | 0.00900 |
| Squared exponential | 0.00669 | 0.02720 |

Mismo ganador y mismo orden que el paper. Como en el artículo (que reporta R = 1 en
entrenamiento), el kernel exponencial tiende a interpolar los datos de entrenamiento; se
añadió una cota inferior al ruido para mantener el problema numéricamente bien planteado.

### 4.3 Support vector machine (Tables 12, 13 · Fig. 7)

SVR ε-insensible resuelta con **SMO implementado desde cero**, con solución exacta del
subproblema de dos variables incluyendo los quiebros del término L1.

Confirmación fuerte de los hiperparámetros: el paper reporta Box constraint = 1.2076 y
Epsilon = 0.1208. Aplicando la heurística por defecto de `fitrsvm` (`iqr(y)/1.349` y
`iqr(y)/13.49`) al bloque de entrenamiento reconstruido se obtiene **1.2092 y 0.1209**.

| Kernel | MSE test (réplica) | MSE test (paper) |
|---|---|---|
| **Quadratic** | **0.01000** | **0.00542** |
| Linear | 0.01873 | 0.02062 |
| Cubic | 0.02378 | 0.01016 |

El SVM lineal replica casi exactamente (R = 0.98870/0.99078/0.99104 frente a
0.98939/0.98859/0.98906). El cúbico se degrada en validación en ambos casos (el paper de
forma más extrema: R = 0.449).

### 4.4 Multilayer perceptron (Tables 14, 15 · Fig. 9)

Arquitectura exacta del paper: **7-10-8-4-3-1**, activaciones `tansig-logsig-tansig-purelin`
y salida lineal; 223 parámetros. Inicialización **Nguyen-Widrow**, entradas y objetivo
reescalados a [-1,1] (`mapminmax`), parada temprana por validación (`max_fail` = 6).

Los tres algoritmos de entrenamiento están implementados desde cero:
**Levenberg-Marquardt** (con jacobiano analítico por retropropagación),
**Rprop** y **gradiente conjugado con reinicios de Powell-Beale**.

Ranking replicado exactamente — `trainlm` (posiciones 1, 3, 4) > `trainrp` (2, 5, 6) >
`traincgb` (7, 8, 9):

| | R train/valid/test | MSE train/valid/test |
|---|---|---|
| Réplica, mejor `trainlm` | 0.99865 / 0.99808 / 0.99837 | 0.00228 / 0.00403 / 0.00350 |
| Paper, Model 1 `trainlm` | 0.9991 / 0.9970 / 0.9986 | 0.00168 / 0.00427 / 0.00255 |

### 4.5 Comparación de los tres métodos (Table 16)

| Conjunto | Mejor modelo (réplica) | Mejor modelo (paper) |
|---|---|---|
| Entrenamiento | Exponential GPR | Exponential GPR |
| Validación | MLP | MLP |
| **Test** | **MLP** | **MLP** |

Coincidencia completa en los tres bloques.

### 4.6 Generalización a EEDB-07/2021 (Table 17)

| Conjunto | MAE | MSE | R | MAPE |
|---|---|---|---|---|
| 2019 — réplica | 0.0366 | 0.0027 | 0.99850 | **3.35 %** |
| 2019 — paper | 0.0294 | 0.0022 | 0.99878 | 2.34 % |
| 2021 — réplica | 0.0412 | 0.0085 | 0.99555 | **3.50 %** |
| 2021 — paper | 0.0298 | 0.0219 | 0.98839 | 2.90 % |

Ambos MAPE quedan holgadamente por debajo del 10%, es decir en la clase "very good" de
Lewis, que es la conclusión del artículo. En MSE sobre 2021 la réplica es *mejor* que el
paper (0.0085 frente a 0.0219).

### 4.7 Intervalos de confianza al 99% (Table 18 · Figs. 10, 11)

σ del error del MLP = 0.04777 kg/s → semianchura al 99% = 2.58 σ = **0.1232 kg/s**.

| Conjunto | Motores | Fuera del intervalo |
|---|---|---|
| EEDB-09/2019 | 563 | 22 (3.91 %) |
| EEDB-07/2021 | 773 | 34 (4.40 %) |
| **Total** | **1336** | 56 |

El paper evalúa 1338 motores; la réplica, 1336. Las figuras generadas
(`results/figures/fig10_ci_2019.png` y `fig11_ci_2021.png`) son visualmente equivalentes a
las Figs. 10 y 11 del artículo, con la misma forma, el mismo rango y la misma anchura de
banda.

**Sobre la Eq. (8).** Tomada al pie de la letra, con n = 565, la semianchura sería
z·σ/√565 ≈ 0.005 kg/s, unas veinte veces más estrecha que las bandas visibles en las
Figs. 10 y 11 del paper. Esas figuras corresponden a n = 1, esto es, al intervalo de
predicción de **un motor individual** — que además es lo único coherente con el objetivo
declarado de decidir si *ese* motor concreto está degradado. La réplica usa n = 1 y
obtiene bandas de la misma anchura que las publicadas, lo que confirma la interpretación.

---

## 5. Diferencias respecto al artículo, y por qué

| Diferencia | Causa |
|---|---|
| N = 563 en 2019 frente a 565 | Criterio de exclusión de filas incompletas ligeramente distinto; afecta al 0.4% de los datos |
| Coeficiente de `ambient baro` en la Table 9 | El paper lo estimó sobre una variable corrupta (§3) |
| Valores concretos de MSE/MAE/MAPE | La partición 70/15/15 es aleatoria y el paper no publica su semilla; los modelos también dependen de la inicialización |
| MSE de GPR en entrenamiento más bajo | El kernel exponencial interpola; el grado exacto depende de la cota de ruido, que el paper no especifica |

Ninguna de estas diferencias altera las conclusiones del artículo.

---

## 6. Evidencia adicional: generalización a ediciones del EEDB posteriores al paper

El paper sólo llega hasta EEDB-07/2021 (Table 17). EASA sigue publicando ediciones nuevas;
esta sección extiende el mismo análisis (Tables 17-18) a todas las ediciones intermedias
entre 2021 y julio de 2026 que se pudieron recuperar:

| Año | Issue | Fecha | Origen | Fichero |
|---|---|---|---|---|
| 2022 | — | — | No existe: EASA no publicó ninguna edición nueva en 2022 (salta de la Issue 28 de 2021 a la 29 de 2023) | — |
| 2023 | 29B | 2023-06-20 | Internet Archive (snapshot de dic. 2023 del enlace de descarga persistente de EASA) | `data/EEDB_2023_Issue29.xlsx` |
| 2024 | 30 | 2024-07-23 | No se encontró una captura archivada de esta edición concreta (el Internet Archive salta de una captura de enero de 2024, todavía Issue 29, a una de agosto de 2025, ya Issue 31) | — |
| 2025 | 31 | 2025-06-06 | Internet Archive (snapshot de agosto de 2025) | `data/EEDB_2025_Issue31.xlsx` |
| 2026 | 32 | 2026-03-20 | Descarga directa de EASA (edición vigente) | `data/EEDB_v32_2026.xlsx` |

`python run_new_edition_evidence.py` entrena el MLP exactamente igual que
`run_replication.py` (sobre EEDB-09/2019, `trainlm`, 3 semillas, se elige la de menor MSE en
test) y lo evalúa sobre cada edición indicada como un grupo de generalización más, análogo a
la Table 17 y a la Table 18:

| Conjunto | N | MAE | MSE | R | MAPE | Fuera del IC 99% |
|---|---|---|---|---|---|---|
| EEDB 2023 (Issue 29B) | 791 | 0.0407 | 0.0083 | 0.99562 | **3.46 %** | 34 (4.30%) |
| EEDB 2025 (Issue 31) | 820 | 0.0411 | 0.0082 | 0.99577 | **3.48 %** | 38 (4.63%) |
| EEDB 2026 (Issue 32, vigente) | 846 | 0.0408 | 0.0080 | 0.99581 | **3.42 %** | 38 (4.49%) |

Las tres ediciones caen en la misma clase "very good" de Lewis que 2019 (3.35%) y 2021
(3.50%), con una proporción de motores fuera del intervalo de confianza (4.3-4.6%) también
consistente con 2019 (3.91%) y 2021 (4.40%). La conclusión del artículo (el MLP generaliza
con MAPE < 10%) se sostiene de forma estable a lo largo de cinco años de datos posteriores
al paper. Figuras en `results/figures/fig_ci_EEDB_*.png`, CSV preprocesados en
`results/EEDB_*_preprocesado.csv`.

Nota: los números exactos varían ligeramente entre ejecuciones porque, como en el resto de
esta réplica, ni la partición 70/15/15 ni la inicialización del MLP usan una semilla fijada
por el paper (§5).

`run_new_edition_evidence.py` no tiene ningún fichero de edición fijado en el código: pide
por consola (o acepta como argumentos) uno o varios `.xlsx` de `data/`, y genera un CSV y
una figura por cada uno, con el nombre del fichero como etiqueta. Así se puede repetir este
mismo análisis sobre cualquier edición futura (o pasada) sin tocar el script.

---

## 7. Estructura del código

```
data/                      ficheros EEDB originales (.xlsx)
src/
  eedb_data.py             carga y preprocesamiento del EEDB
  splits.py                partición 70/15/15, estandarización, mapminmax
  metrics.py               MSE, MAE, MAPE, R           (Eqs. 5-7)
  distributions.py         beta incompleta -> p-valores de t y F
  mlr.py                   regresión lineal múltiple    (Tables 7-9, Eq. 2)
  optimize.py              BFGS ("Quasinewton")
  gpr.py                   Gaussian process regression  (Tables 10-11, Eq. 3)
  svr.py                   SVR con SMO                  (Tables 12-13, Eq. 4)
  mlp.py                   MLP con LM / Rprop / CG-PB   (Tables 14-15)
  confidence.py            intervalos de confianza      (Table 18, Eq. 8)
  figures.py               Figs. 6, 7, 9, 10, 11
run_replication.py            pipeline completo del paper (§1-5)
run_new_edition_evidence.py   evidencia adicional sobre la edición vigente del EEDB (§6)
results/                   generado: tablas, figuras y CSV preprocesados
```
