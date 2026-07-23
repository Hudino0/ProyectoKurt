# Resumen de orientación — qué es esta carpeta y cómo encaja

Mapa rápido para retomar el proyecto sin releer todo `REPLICACION.md`. Para el detalle
numérico completo, ese fichero sigue siendo la fuente autoritativa.

## Qué es esto

Réplica en Python puro (sin scikit-learn/TensorFlow/PyTorch) del artículo de Kurt (2024)
sobre predicción del **Fuel Flow T/O** de motores de avión con datos del **ICAO Engine
Emissions Databank (EEDB)**. Se comparan 4 modelos (regresión lineal, GPR, SVM, MLP), todos
construidos a mano, y se usa el mejor (MLP) para detectar motores con performance anómala
mediante intervalos de confianza al 99%.

## Cómo correrlo

```bash
python run_replication.py              # réplica completa del paper (~4 min)
python run_new_edition_evidence.py      # evidencia extra sobre el EEDB vigente (~1 min)
```

## Qué hace cada pieza (`src/`)

| Fichero | Qué hace |
|---|---|
| `eedb_data.py` | Carga los `.xlsx` del EEDB y arma la matriz de 8 columnas. Tiene `load_2019()`, `load_2021()` (los del paper) y `load_current()` (edición vigente, añadida esta sesión) |
| `splits.py` | Partición 70/15/15, estandarización, escalado mapminmax |
| `metrics.py` | MSE, MAE, MAPE, R |
| `distributions.py` | Beta incompleta, para los p-valores de la regresión lineal |
| `mlr.py` | Regresión lineal múltiple (QR de Householder) |
| `optimize.py` | BFGS ("Quasinewton" del paper) |
| `gpr.py` | Gaussian process regression (Cholesky propia) |
| `svr.py` | Support vector regression (SMO propio) |
| `mlp.py` | Red neuronal con Levenberg-Marquardt, Rprop y gradiente conjugado |
| `confidence.py` | Intervalos de confianza al 99% |
| `figures.py` | Generación de las figuras |

**Ninguno de estos ficheros de modelo (`mlr.py`, `gpr.py`, `svr.py`, `mlp.py`,
`optimize.py`, `distributions.py`) se ha tocado en ninguna sesión posterior a su
verificación inicial** — están ya comprobados número a número contra el paper.

## El hallazgo de la sesión anterior

La **Table 4 del paper original** (estadística descriptiva de EEDB-09/2019) contiene datos
corruptos: presión barométrica máxima de 1 019 394.5 kPa y temperatura mínima de 144 K,
físicamente imposibles. Se reconstruyó por aritmética que es 1 celda de temperatura y unas
4-5 de presión mal importadas por el autor, y que el defecto se propaga al coeficiente de
`ambient baro` en su Table 9. Detalle completo en `REPLICACION.md` §3.

## Qué se hizo en esta sesión

1. Se comprobó que EASA ya va por la **Issue 32 (edición 03/2026)** del EEDB — muy posterior
   a la 07/2021 que usa el paper — y se descargó a `data/EEDB_v32_2026.xlsx`.
2. Se añadió `eedb_data.load_current()` (sin tocar `load_2021()` ni ningún modelo) para leer
   cualquier edición en el formato tabular de 2021 en adelante, pasando el nombre del
   fichero como parámetro.
3. Se creó `run_new_edition_evidence.py`, que entrena el MLP igual que
   `run_replication.py` y lo evalúa sobre la(s) edición(es) que se le indiquen (por consola
   o por argumento) como grupo(s) de generalización adicionales (análogo a las Tables 17-18
   del paper). No tiene ningún fichero fijado en el código.
4. Se recuperaron del Internet Archive dos ediciones intermedias que EASA ya no publica:
   **2023 (Issue 29B)** y **2025 (Issue 31)** — no existe edición propia de 2022 (EASA no
   publicó ninguna ese año) y no se pudo localizar un archivo de la edición 2024 (Issue 30).
5. Resultado sobre 2023, 2025 y 2026: **MAPE entre 3.42% y 3.48%**, siempre clase "very
   good" de Lewis — la conclusión del artículo se sostiene de forma estable a lo largo de
   cinco años de datos posteriores a su publicación. Detalle en `REPLICACION.md` §6.

## Dónde mirar según lo que necesites

- **Resultado numérico tabla a tabla vs. el paper** → `REPLICACION.md`
- **Evidencia con la edición vigente del EEDB (2026)** → `REPLICACION.md` §6
- **Plan y checklist original** → `tasks/todo.md`
- **Bugs encontrados durante el desarrollo y la regla aprendida de cada uno** →
  `tasks/lessons.md`
- **Salida completa de la última ejecución** → `results/salida_replicacion.txt`
