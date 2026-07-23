"""
Regresión lineal múltiple (Section 2.2 del paper).

Replica el análisis que el autor hizo en SPSS y que aparece en:
    Table 7  - resumen del modelo (R, R², R² ajustado, error típico)
    Table 8  - tabla ANOVA (sumas de cuadrados, gl, F, Sig.)
    Table 9  - coeficientes (β, error típico, Beta estandarizado, t, Sig.)
    Eq. (2)  - ecuación de regresión resultante

Modelo, Eq. (1) del paper:

    y = β0 + β1*p1 + β2*p2 + ... + βn*pn + ε

La estimación es por mínimos cuadrados ordinarios. En lugar de invertir la
matriz (X'X) explícitamente —numéricamente inestable cuando los predictores
tienen escalas tan dispares como aquí (humedad ~1e-3 frente a empuje ~5e2)— se
resuelve el sistema mediante descomposición QR con reflexiones de Householder,
implementada más abajo desde cero.
"""

import numpy as np

from distributions import f_upper_p, t_two_sided_p


def qr_householder(A):
    """
    Descomposición QR reducida de A (m x n, m >= n) por reflexiones de
    Householder: A = Q R con Q ortonormal (m x n) y R triangular superior (n x n).

    Cada reflexión anula los elementos por debajo de la diagonal de una columna.
    Es el método estándar por su estabilidad numérica: el número de condición del
    problema de mínimos cuadrados no se eleva al cuadrado, como sí ocurre al
    formar las ecuaciones normales X'X.
    """
    R = A.astype(float).copy()
    m, n = R.shape
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0.0:
            continue
        # Vector de Householder v = x + sign(x0)*||x||*e1: refleja x sobre el
        # eje e1. El signo se elige igual al de x0 para evitar cancelación
        # catastrófica cuando x ya está casi alineado con e1.
        v = x.copy()
        sign = 1.0 if x[0] >= 0 else -1.0
        v[0] += sign * norm_x
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            continue
        v = v / norm_v
        # Aplicar H = I - 2vv' por la izquierda a R y acumular en Q.
        R[k:, :] -= 2.0 * np.outer(v, v @ R[k:, :])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q[:, :n], R[:n, :n]


def solve_triangular_upper(R, b):
    """Sustitución hacia atrás para R x = b con R triangular superior."""
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


class MultipleLinearRegression:
    """Ajuste OLS con el juego completo de estadísticos de SPSS."""

    def __init__(self):
        self.coefficients = None       # β0 incluido, en la posición 0
        self.summary = {}              # Table 7
        self.anova = {}                # Table 8
        self.coefficient_table = []    # Table 9

    def fit(self, X, y, feature_names=None):
        n, k = X.shape                 # n observaciones, k predictores
        design = np.hstack([np.ones((n, 1)), X])   # columna de unos = intercepto

        Q, R = qr_householder(design)
        self.coefficients = solve_triangular_upper(R, Q.T @ y)

        predictions = design @ self.coefficients
        residuals = y - predictions

        # --- Sumas de cuadrados (Table 8) ---------------------------------
        ss_total = float(np.sum((y - y.mean()) ** 2))
        ss_residual = float(np.sum(residuals ** 2))
        ss_regression = ss_total - ss_residual

        df_regression = k
        df_residual = n - k - 1
        df_total = n - 1

        ms_regression = ss_regression / df_regression
        ms_residual = ss_residual / df_residual
        f_statistic = ms_regression / ms_residual

        self.anova = {
            "ss_regression": ss_regression, "ss_residual": ss_residual,
            "ss_total": ss_total,
            "df_regression": df_regression, "df_residual": df_residual,
            "df_total": df_total,
            "ms_regression": ms_regression, "ms_residual": ms_residual,
            "F": f_statistic,
            "Sig": f_upper_p(f_statistic, df_regression, df_residual),
        }

        # --- Resumen del modelo (Table 7) ---------------------------------
        r_squared = 1.0 - ss_residual / ss_total
        adjusted = 1.0 - (1.0 - r_squared) * df_total / df_residual
        self.summary = {
            "R": float(np.sqrt(max(r_squared, 0.0))),
            "R_square": r_squared,
            "adjusted_R_square": adjusted,
            # El "Std. Error of the Estimate" de SPSS es la raíz del cuadrado
            # medio residual, no la desviación típica de los residuos.
            "std_error": float(np.sqrt(ms_residual)),
        }

        # --- Coeficientes (Table 9) ---------------------------------------
        # var(β) = σ² (X'X)^-1, y con X = QR se tiene (X'X)^-1 = R^-1 R^-T,
        # de modo que basta invertir la triangular R.
        R_inverse = np.linalg.inv(R)
        covariance = ms_residual * (R_inverse @ R_inverse.T)
        standard_errors = np.sqrt(np.diag(covariance))

        # El coeficiente estandarizado (Beta) reescala β por las desviaciones
        # típicas, permitiendo comparar la importancia relativa de predictores
        # medidos en unidades distintas.
        y_std = y.std(ddof=1)
        x_std = X.std(axis=0, ddof=1)

        names = feature_names or [f"x{i + 1}" for i in range(k)]
        self.coefficient_table = []
        for i, label in enumerate(["(Constant)"] + list(names)):
            beta = float(self.coefficients[i])
            se = float(standard_errors[i])
            t = beta / se if se > 0 else float("inf")
            standardized = (0.0 if i == 0
                            else beta * x_std[i - 1] / y_std)
            self.coefficient_table.append({
                "name": label, "beta": beta, "std_error": se,
                "standardized": float(standardized), "t": float(t),
                "Sig": t_two_sided_p(t, df_residual),
            })
        return self

    def predict(self, X):
        return self.coefficients[0] + X @ self.coefficients[1:]

    def equation(self, feature_names):
        """Devuelve la ecuación en texto, al estilo de la Eq. (2) del paper."""
        parts = [f"Fuel Flow T/O = {self.coefficients[0]:.6g}"]
        for name, coefficient in zip(feature_names, self.coefficients[1:]):
            sign = "-" if coefficient < 0 else "+"
            parts.append(f" {sign} {abs(coefficient):.4g} x {name}")
        return "".join(parts)
