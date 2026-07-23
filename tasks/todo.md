# Replicación: "Evaluation of aircraft engine performance during takeoff phase with machine learning methods" (Kurt, 2024)

Neural Computing and Applications (2024) 36:19173–19190 · https://doi.org/10.1007/s00521-024-10220-3

## Objetivo
Replicar en Python puro (sin librerías de ML) todo el proceso del artículo:
predecir el parámetro **Fuel Flow T/O** de motores aeronáuticos a partir de 7
parámetros de entrada, con MLR, GPR, SVM y MLP, y construir intervalos de
confianza al 99% para detectar degradación de performance.

---

## Fase 0 — Datos  ✅
- [x] Localizar el artículo (`paperEvaluationKurt - copia.pdf`) y leerlo completo
- [x] Identificar la base de datos: ICAO Engine Emissions Databank (EEDB), hospedada por EASA
- [x] Recuperar las **ediciones históricas exactas** (EASA sólo publica la vigente)
      vía Internet Archive:
      - `EEDB_v26B_2019.xlsx` → Issue **26B, 2019-09-20** = *EEDB-09/2019* del paper
      - `EEDB_v28C_2021.xlsx` → Issue **28C, 2021-07-20** = *EEDB-07/2021* del paper
- [x] Ingeniería inversa del preprocesamiento del autor (validada contra Table 6):
      - `Engine type`: TF → 1, MTF → 2
      - `Ambient baro/temp/humidity` = **media de las columnas Min y Max**
      - Filtro: sólo TF/MTF, con Fuel Flow T/O y los 7 predictores presentes
- [x] Verificar contra Table 5 (2021): **N=773 y las 8 estadísticas coinciden exactamente**
- [x] Verificar contra Table 4 (2019): N=563 vs 565 del paper; min/max de todas las
      variables limpias coinciden

### Hallazgo: datos corruptos en el artículo original
La Table 4 del paper reporta para EEDB-09/2019:
`Ambient baro: min 0.00, max 1019394.5, mean 8004.3, std 85867.39` y
`Ambient temp: min 144`. Son valores físicamente imposibles (la presión
barométrica atmosférica es ~95–104 kPa; 144 K = −129 °C).

Reconstrucción aritmética del defecto:
- **Temperatura**: una única celda vale 144 en vez de 288 (motor `4AL002`, cuyo
  valor real en el EEDB es 288 K → fue dividido por 2 al importar). Verificación:
  desplazamiento de media esperado −144/565 = −0.25 → 287.22−0.25 = **286.97**
  (paper: 286.9116) y std esperada √(70.5 + 143²/564) = **10.33** (paper: 10.30618).
- **Presión**: ~4–5 celdas valen ≈1×10⁶ en vez de ≈101. Verificación: la suma
  implícita (565 × 8004.3 = 4.52×10⁶) y la suma de cuadrados implícita
  (564 × 85867.4² = 4.16×10¹²) son ambas consistentes con exactamente ~4.2
  valores de magnitud 10⁶.

Consecuencia: el coeficiente de `ambient baro` del paper (Table 9,
β = 2.242×10⁻⁷) está en la escala de una variable de ~10⁶, no de ~10². El defecto
se propagó a todos los modelos publicados. **La replicación usa datos correctos**
y por tanto no puede reproducir ese coeficiente; se documenta la diferencia.

---

## Fase 1 — Infraestructura  ✅
- [x] `src/eedb_data.py` — parseo de ambos .xlsx, construcción de la matriz 8 columnas,
      exportación a CSV, reproducción de Tables 4 y 5
- [x] `src/metrics.py` — MSE, MAE, MAPE, R (Eqs. 5, 6, 7 del paper)
- [x] `src/splits.py` — partición aleatoria 70/15/15 (equivalente a `dividerand`)
- [x] `src/distributions.py` — beta incompleta (fracción continua de Lentz) para los
      p-valores de t y F, sin SciPy

## Fase 2 — Modelos desde cero (sin scikit-learn / TF / PyTorch)  ✅
- [x] `src/mlr.py` — regresión lineal múltiple por **QR de Householder** (más estable
      que las ecuaciones normales con predictores de escalas tan dispares):
      coeficientes, R, R², R² ajustado, error estándar, tabla ANOVA, t y Sig.
      → replica Tables 7, 8, 9 y Eq. 2
- [x] `src/optimize.py` — BFGS con búsqueda lineal de Wolfe (el "Quasinewton" del paper)
- [x] `src/gpr.py` — GPR exacta con **Cholesky right-looking** propia, kernels
      exponential / rational quadratic / squared exponential, hiperparámetros por
      máxima verosimilitud marginal → replica Tables 10, 11 y Fig. 6
- [x] `src/svr.py` — SVR ε-insensitiva resuelta con **SMO** propio, con solución exacta
      del subproblema de 2 variables incluyendo los quiebros del término L1;
      kernels polinómicos lineal / cuadrático / cúbico → replica Tables 12, 13 y Fig. 7
- [x] `src/mlp.py` — MLP **7-10-8-4-3-1**, `tansig-logsig-tansig-purelin`, inicialización
      Nguyen-Widrow, entrenado con **Levenberg–Marquardt** (jacobiano analítico),
      **Rprop** y **gradiente conjugado Powell–Beale** → replica Tables 14, 15 y Fig. 9

## Fase 3 — Resultados  ✅
- [x] Comparación de los 3 métodos en train/validation/test → replica Table 16
- [x] Aplicar el mejor MLP a EEDB-07/2021 (773 muestras) → replica Table 17
- [x] Intervalos de confianza al 99% (Eq. 8, z=2.58) → replica Table 18, Figs. 10 y 11
- [x] Detección de anomalías: 22/563 (2019) y 34/773 (2021) motores fuera del intervalo

## Fase 4 — Verificación  ✅
- [x] Comparar cada número producido contra la tabla correspondiente del paper
      (el pipeline imprime réplica y paper lado a lado en cada tabla)
- [x] Informe `REPLICACION.md` con tabla paper-vs-réplica y discusión de
      las diferencias

---

## Review

### Qué se hizo
Replicación completa y ejecutable del artículo. `python run_replication.py` corre el
proceso de principio a fin en ~4 minutos e imprime cada tabla junto a la del paper.

### Resultado: todas las conclusiones del artículo se reproducen
| Conclusión | ¿Replicada? |
|---|---|
| Mejor GPR = kernel exponencial | Sí |
| Mejor SVM = cuadrático; el cúbico se degrada en validación | Sí |
| Mejor MLP = `trainlm` (por delante de `trainrp` y `traincgb`) | Sí |
| Mejor modelo global en test = MLP | Sí |
| MLP generaliza a 2021 con MAPE < 10% ("very good") | Sí |

Coincidencias numéricas más fuertes:
- **Table 5 (2021): exacta.** N=773 y las 8 estadísticas descriptivas.
- **Tables 7-8 (MLR):** R 0.98974 vs 0.990; R² 0.97958 vs 0.979; SE 0.13665 vs 0.13733.
- **Table 13 (SVM):** la heurística por defecto de `fitrsvm` sobre el bloque de
  entrenamiento reconstruido da C=1.2092 y ε=0.1209, frente a 1.2076 y 0.1208 del paper.
- **Figs. 10 y 11:** misma forma, mismo rango y misma anchura de banda.

### Contribución más allá de la réplica
Se detectó que la **Table 4 del artículo contiene datos corruptos** (presión barométrica
máx = 1 019 394.5 kPa, temperatura mín = 144 K). Se reconstruyó el defecto por aritmética:
1 celda de temperatura y ~4-5 de presión. El error se propagó a los modelos publicados
(explica el coeficiente β = 2.242×10⁻⁷ de la Table 9). Detalle en `REPLICACION.md`.

### Decisiones de diseño que merecieron pausa
- **QR de Householder en vez de ecuaciones normales** en la MLR: los predictores van de
  7×10⁻³ (humedad) a 5×10² (empuje), y (X'X) elevaría al cuadrado el número de condición.
- **Cholesky "right-looking"** en vez de elemento a elemento: BFGS evalúa la verosimilitud
  marginal miles de veces; la versión vectorizada por columnas la hace viable.
- **Solución exacta del subproblema SMO** incluyendo los quiebros de ||β||₁, en vez de
  ignorar la no suavidad: es lo que hace que el algoritmo converja de verdad.

### Bugs encontrados y corregidos durante el desarrollo
Registrados en `tasks/lessons.md`.

### Limitaciones honestas
- N=563 frente a 565 en 2019 (99.6%): el criterio exacto de exclusión de filas incompletas
  del autor no es recuperable a partir del artículo.
- Los valores puntuales de MSE/MAE/MAPE difieren porque el paper no publica la semilla de
  su partición aleatoria ni de la inicialización de las redes. El orden entre modelos, que
  es lo que sostiene las conclusiones, sí se reproduce.
