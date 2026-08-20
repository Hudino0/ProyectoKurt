# ProyectoKurt — Jet-Engine Fuel-Flow Prediction, ML From Scratch

**I rebuilt four machine-learning models by hand — no scikit-learn, no
TensorFlow — to predict jet-engine fuel flow, and reproduced every result of a
2024 research paper.**

A full Python replication of:

> Kurt, B. (2024). *Evaluation of aircraft engine performance during takeoff
> phase with machine learning methods.* Neural Computing and Applications,
> 36:19173–19190. <https://doi.org/10.1007/s00521-024-10220-3>

The paper predicts an engine's **takeoff fuel flow (Fuel Flow T/O)** from 7
variables in the ICAO Engine Emissions Databank. It compares multiple linear
regression, Gaussian process regression (GPR), support vector machines (SVM),
and a neural network (MLP), then uses the best model to flag engine performance
loss with 99% confidence intervals.

## Built from scratch

The only library is NumPy, for arrays and basic linear algebra. Every model is
my own code:

| Component | My implementation |
|---|---|
| Multiple linear regression | Householder QR decomposition |
| t and F p-values | Incomplete beta function, Lentz continued fraction |
| Gaussian process regression | Right-looking Cholesky + BFGS on the marginal likelihood |
| Support vector regression | SMO with an exact 2-variable subproblem solution |
| Neural network (MLP) | Levenberg–Marquardt (analytic Jacobian), Rprop, Powell–Beale conjugate gradient |

## Run it

```bash
python -m pip install numpy openpyxl matplotlib
python run_replication.py        # ~4 minutes
```

It prints every table from the paper next to my replication, and writes figures
and CSVs to `results/`.

## Results

I reproduce every conclusion in the paper. The full table-by-table report is in
[REPLICACION.md](REPLICACION.md) (Spanish).

I also found a flaw in the original paper: Table 4 lists physically impossible
values for barometric pressure and temperature. I trace where they come from by
rebuilding them, by arithmetic, from the paper's own published statistics.

## Extra: data the paper never saw (2023–2026)

The paper stops at the 07/2021 databank. `run_new_edition_evidence.py` applies
the same MLP to any later edition. On the 2023, 2025, and current 2026 editions
it scores **3.42–3.48% MAPE** — "very good" on Lewis's scale, the same grade as
2019 and 2021. Details in [REPLICACION.md §6](REPLICACION.md).

## What's inside

```
data/                         ICAO databank: paper editions (2019, 2021) + later (2023, 2025, 2026)
src/                          modules: data loading, metrics, and the four models
run_replication.py            the full paper pipeline
run_new_edition_evidence.py   extra test on the current databank edition
results/                      tables, figures, preprocessed CSVs
tasks/                        development notes and project map
```
