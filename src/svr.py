"""
Máquina de vectores soporte para regresión (Section 2.3.2 del paper), desde cero.

El paper compara kernels lineal, cuadrático y cúbico (Table 12) y reporta la
estructura del mejor modelo en la Table 13 (Function: Polynomial, Box constraint
1.2076, Bias 1.7458, Epsilon 0.1208).

Formulación. La SVR epsilon-insensible tiene el dual

    min_{a,a*}  1/2 sum_ij (a_i-a_i*)(a_j-a_j*) K_ij
                + eps sum_i (a_i+a_i*) - sum_i y_i (a_i-a_i*)
    s.a.        sum_i (a_i - a_i*) = 0,     0 <= a_i, a_i* <= C

Como en el óptimo nunca son ambos no nulos, conviene la variable única
beta_i = a_i - a_i*, con |beta_i| <= C. El problema queda

    min_beta  1/2 beta' K beta + eps ||beta||_1 - y'beta
    s.a.      sum_i beta_i = 0,     -C <= beta_i <= C

que es la forma que resuelve LIBSVM. La predicción es, como en la Eq. (4) del
paper,  y = (K_xi * W_jk) + b, es decir  f(x) = sum_j beta_j K(x, x_j) + b.

El optimizador es SMO (Sequential Minimal Optimization) implementado a mano: la
restricción de igualdad obliga a mover dos variables a la vez, y cada
subproblema de dos variables se resuelve de forma **exacta**, incluyendo los
quiebros que introduce el término ||beta||_1.
"""

import numpy as np

# Valores por defecto de `fitrsvm` de MATLAB, que son los que usó el paper:
#   BoxConstraint = iqr(y) / 1.349     Epsilon = iqr(y) / 13.49
# (1.349 es el rango intercuartílico de una normal estándar, de modo que el
# cociente estima su desviación típica de forma robusta.)
BOX_CONSTRAINT_DIVISOR = 1.349
EPSILON_DIVISOR = 13.49


def default_hyperparameters(y):
    """Devuelve (C, epsilon) con la heurística por defecto de MATLAB."""
    q75, q25 = np.percentile(y, [75, 25])
    interquartile_range = q75 - q25
    return (interquartile_range / BOX_CONSTRAINT_DIVISOR,
            interquartile_range / EPSILON_DIVISOR)


def polynomial_kernel(A, B, degree, kernel_scale=1.0):
    """
    Kernel polinómico de MATLAB:  K(u,v) = (1 + (u·v)/s)^d

    d=1 da el modelo "Linear SVM" del paper, d=2 el "Quadratic SVM" y d=3 el
    "Cubic SVM" (Table 12).
    """
    return (1.0 + (A @ B.T) / kernel_scale) ** degree


class SupportVectorRegressor:
    """SVR epsilon-insensible con kernel polinómico, entrenada con SMO."""

    def __init__(self, degree=2, C=None, epsilon=None, kernel_scale=1.0,
                 tolerance=1e-3, max_iterations=200000):
        self.degree = degree
        self.C = C                    # box constraint; None -> heurística
        self.epsilon = epsilon        # anchura del tubo; None -> heurística
        self.kernel_scale = kernel_scale
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.beta = None
        self.bias = None
        self._X = None
        self.n_iterations = 0

    # -- subproblema de dos variables --------------------------------------

    def _solve_pair(self, i, j, beta, gradient, K, C, epsilon):
        """
        Optimiza exactamente el par (beta_i, beta_j) manteniendo su suma.

        Con beta_i = t y beta_j = s - t, la parte suave del objetivo es una
        parábola en t y la parte no suave es eps*(|t| + |s-t|), lineal a trozos
        con quiebros en t=0 y t=s. La función total es convexa y lineal a trozos
        + cuadrática, así que basta minimizar en cada uno de los (como mucho)
        tres tramos y quedarse con el mejor candidato.
        """
        eta = K[i, i] - 2.0 * K[i, j] + K[j, j]
        if eta <= 1e-12:
            return None                       # puntos idénticos: nada que hacer

        t0 = beta[i]
        s = beta[i] + beta[j]
        # Derivada de la parte suave en t0 (gradient_k = (K beta)_k - y_k).
        slope = gradient[i] - gradient[j]

        # Intervalo admisible para t, dado que |beta_i|<=C y |beta_j|<=C.
        low = max(-C, s - C)
        high = min(C, s + C)
        if high - low < 1e-12:
            return None

        # Los quiebros parten el intervalo en tramos con signos constantes.
        breakpoints = sorted({low, high} | {b for b in (0.0, s) if low < b < high})

        def objective(t):
            """Objetivo relativo (se omiten las constantes en t)."""
            delta = t - t0
            return (0.5 * eta * delta ** 2 + slope * delta
                    + epsilon * (abs(t) + abs(s - t)))

        candidates = list(breakpoints)
        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            middle = 0.5 * (left + right)
            # Subgradiente constante del término L1 dentro del tramo.
            sign_t = np.sign(middle)
            sign_st = np.sign(s - middle)
            unconstrained = t0 - (slope + epsilon * (sign_t - sign_st)) / eta
            if left < unconstrained < right:
                candidates.append(unconstrained)

        best_t = min(candidates, key=objective)
        delta = best_t - t0
        if abs(delta) < 1e-12:
            return None
        return delta

    # -- entrenamiento ------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        C, epsilon = default_hyperparameters(y)
        if self.C is not None:
            C = self.C
        if self.epsilon is not None:
            epsilon = self.epsilon
        self.C, self.epsilon = float(C), float(epsilon)

        K = polynomial_kernel(X, X, self.degree, self.kernel_scale)

        beta = np.zeros(n)
        # gradient_k = d/dbeta_k de la parte suave = (K beta)_k - y_k
        gradient = -y.copy()

        for iteration in range(self.max_iterations):
            # Selección del par que más viola las condiciones KKT.
            #
            # Mover beta_i hacia arriba y beta_j hacia abajo en delta (lo único
            # que permite la restricción de igualdad) cambia el objetivo en
            # delta * (up_i - down_j), donde up y down son las derivadas
            # direccionales, que dependen del subgradiente de |beta_k| en el
            # sentido del movimiento:
            #
            #   subir  beta_k:  +1 si beta_k >= 0, -1 si beta_k < 0
            #   bajar  beta_k:  +1 si beta_k >  0, -1 si beta_k <= 0
            #
            # El par óptimo es entonces el que minimiza up_i y maximiza down_j.
            up = gradient + epsilon * np.where(beta >= 0, 1.0, -1.0)
            down = gradient + epsilon * np.where(beta > 0, 1.0, -1.0)

            # Sólo se puede subir beta_k si no está ya en la cota +C, y bajarlo
            # si no está en -C.
            can_increase = beta < C - 1e-12
            can_decrease = beta > -C + 1e-12

            i = int(np.argmin(np.where(can_increase, up, np.inf)))
            j = int(np.argmax(np.where(can_decrease, down, -np.inf)))
            violation = down[j] - up[i]
            if violation < self.tolerance or i == j:
                self.n_iterations = iteration
                break

            delta = self._solve_pair(i, j, beta, gradient, K, C, epsilon)
            if delta is None:
                # El par elegido no admite mejora; se prueba con otro al azar
                # para no bloquear el algoritmo.
                j = int(np.random.default_rng(iteration).integers(n))
                delta = self._solve_pair(i, j, beta, gradient, K, C, epsilon)
                if delta is None:
                    self.n_iterations = iteration
                    break

            beta[i] += delta
            beta[j] -= delta
            # Actualización de rango 2 del gradiente: O(n) en vez de O(n^2).
            gradient += delta * (K[:, i] - K[:, j])
        else:
            self.n_iterations = self.max_iterations

        self.beta = beta
        self._X = X
        self.bias = self._compute_bias(beta, gradient, epsilon, C)
        return self

    def _compute_bias(self, beta, gradient, epsilon, C):
        """
        Sesgo b a partir de las condiciones KKT de los vectores soporte libres.

        Para un punto con 0 < beta_i < C se cumple  y_i - f(x_i) = eps, y para
        uno con -C < beta_i < 0,  y_i - f(x_i) = -eps. Como
        gradient_i = (K beta)_i - y_i, en ambos casos  b = -gradient_i -+ eps.
        Se promedia sobre todos los vectores soporte libres para reducir el
        efecto del ruido numérico.
        """
        free_positive = (beta > 1e-8) & (beta < C - 1e-8)
        free_negative = (beta < -1e-8) & (beta > -C + 1e-8)
        estimates = np.concatenate([
            -gradient[free_positive] - epsilon,
            -gradient[free_negative] + epsilon,
        ])
        if estimates.size > 0:
            return float(estimates.mean())
        # Sin vectores soporte libres se toma el punto medio del intervalo
        # compatible con las condiciones KKT.
        return float(np.mean(-gradient))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        K = polynomial_kernel(X, self._X, self.degree, self.kernel_scale)
        return K @ self.beta + self.bias

    def structure(self):
        """Información equivalente a la Table 13 del paper."""
        return {
            "Model type": {1: "Linear SVM", 2: "Quadratic SVM",
                           3: "Cubic SVM"}[self.degree],
            "Function": "Polynomial",
            "Kernel scale": self.kernel_scale,
            "Box constraint": self.C,
            "Bias": self.bias,
            "Epsilon": self.epsilon,
            "Support vectors": int(np.sum(np.abs(self.beta) > 1e-8)),
        }
