"""
Regresión por procesos gaussianos (Section 2.3.1 del paper), desde cero.

Modelo, Eq. (3) del paper:

    f(x) ~ GPR(m(x), k(x, x'))

con función media constante ("Basis function: Constant" en la Table 11) y una de
las tres funciones de covarianza que el autor compara en la Table 10:
exponencial, cuadrática racional y exponencial cuadrática.

Formulación. Con base constante h(x)=1 y coeficiente beta, las observaciones son

    y = beta + f + ruido,    f ~ N(0, K),    ruido ~ N(0, sigma_n^2 I)

de modo que  y ~ N(beta*1, K + sigma_n^2 I).  Escribiendo A = K + sigma_n^2 I:

  * beta se estima por mínimos cuadrados generalizados:  beta = (1'A^-1 y)/(1'A^-1 1)
  * la predicción en x* es  mu* = beta + k*' A^-1 (y - beta*1)
  * la varianza predictiva es  s*^2 = k(x*,x*) - k*' A^-1 k*

Los hiperparámetros (escala de longitud, amplitud de señal, ruido y, para la
cuadrática racional, el parámetro alpha) se ajustan maximizando la
log-verosimilitud marginal con el BFGS de `optimize.py`, que es el "Quasinewton"
que reporta la Table 11.

Todos los sistemas lineales se resuelven con una factorización de Cholesky
escrita a mano: A es simétrica definida positiva, así que Cholesky es a la vez
el método más rápido y el más estable, y además da el log-determinante gratis.
"""

import numpy as np

from optimize import bfgs_minimize

# Sustituto de sigma_n^2 mínimo, para que A nunca deje de ser definida positiva.
JITTER = 1e-10


def cholesky(A):
    """
    Factorización de Cholesky A = L L' con L triangular inferior.

    Variante "right-looking": en cada paso k se calcula la columna k de L y se
    actualiza el bloque restante con un producto exterior. Frente a la versión
    elemento a elemento, sustituye el bucle doble por un único bucle de n pasos
    con operaciones vectorizadas, lo que la hace unas mil veces más rápida — algo
    imprescindible aquí, porque BFGS evalúa la verosimilitud marginal (y por
    tanto esta factorización) miles de veces.

    Lanza ValueError si A no es definida positiva, lo que durante la
    optimización señala una combinación de hiperparámetros inadmisible.
    """
    R = np.array(A, dtype=float, copy=True)
    n = R.shape[0]
    L = np.zeros((n, n))
    for k in range(n):
        pivot = R[k, k]
        if pivot <= 0.0:
            raise ValueError("la matriz no es definida positiva")
        root = np.sqrt(pivot)
        L[k, k] = root
        if k + 1 < n:
            column = R[k + 1:, k] / root
            L[k + 1:, k] = column
            # Complemento de Schur del bloque inferior derecho.
            R[k + 1:, k + 1:] -= np.outer(column, column)
    return L


def _forward_substitution(L, b):
    """
    Resuelve L x = b con L triangular inferior.

    Acepta b unidimensional o una matriz de varios lados derechos en columnas.
    """
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b)
    n = L.shape[0]
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def _backward_substitution(U, b):
    """Resuelve U x = b con U triangular superior (b vector o matriz)."""
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b)
    n = U.shape[0]
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def cholesky_solve(L, b):
    """Resuelve A x = b dada la factorización A = L L'."""
    return _backward_substitution(L.T, _forward_substitution(L, b))


def _pairwise_distances(A, B):
    """Matriz de distancias euclídeas entre las filas de A y las de B."""
    squared = (np.sum(A ** 2, axis=1)[:, None]
               + np.sum(B ** 2, axis=1)[None, :]
               - 2.0 * A @ B.T)
    # Los errores de redondeo pueden dar cuadrados ligeramente negativos.
    return np.sqrt(np.maximum(squared, 0.0))


# --------------------------------------------------------------------------
# Funciones de covarianza (nomenclatura de MATLAB, Table 10 del paper)
# --------------------------------------------------------------------------

def kernel_exponential(A, B, parameters):
    """k(r) = sigma_f^2 * exp(-r / l)   ('exponential')"""
    sigma_f, length_scale = parameters[0], parameters[1]
    r = _pairwise_distances(A, B)
    return sigma_f ** 2 * np.exp(-r / length_scale)


def kernel_squared_exponential(A, B, parameters):
    """k(r) = sigma_f^2 * exp(-r^2 / (2 l^2))   ('squaredexponential')"""
    sigma_f, length_scale = parameters[0], parameters[1]
    r = _pairwise_distances(A, B)
    return sigma_f ** 2 * np.exp(-(r ** 2) / (2.0 * length_scale ** 2))


def kernel_rational_quadratic(A, B, parameters):
    """
    k(r) = sigma_f^2 * (1 + r^2 / (2 alpha l^2))^(-alpha)  ('rationalquadratic')

    Equivale a una mezcla infinita de kernels exponenciales cuadráticos con
    distintas escalas de longitud, lo que le permite capturar a la vez
    variaciones suaves y bruscas.
    """
    sigma_f, length_scale, alpha = parameters[0], parameters[1], parameters[2]
    r = _pairwise_distances(A, B)
    return sigma_f ** 2 * (1.0 + (r ** 2) / (2.0 * alpha * length_scale ** 2)) ** (-alpha)


KERNELS = {
    "exponential":         (kernel_exponential, 2),
    "squaredexponential":  (kernel_squared_exponential, 2),
    "rationalquadratic":   (kernel_rational_quadratic, 3),
}


class GaussianProcessRegressor:
    """GPR con base constante y kernel seleccionable."""

    def __init__(self, kernel="exponential"):
        if kernel not in KERNELS:
            raise ValueError(f"kernel desconocido: {kernel}")
        self.kernel_name = kernel
        self.kernel_function, self._n_kernel_parameters = KERNELS[kernel]
        self.kernel_parameters = None    # [sigma_f, length_scale, (alpha)]
        self.sigma_n = None              # desviación típica del ruido
        self.beta = None                 # coeficiente de la base constante
        self._X = None
        self._L = None                   # Cholesky de A
        self._alpha_vector = None        # A^-1 (y - beta)
        self._sigma_floor = 0.0          # se fija en fit(), relativo a std(y)

    # -- verosimilitud ------------------------------------------------------

    def _negative_log_marginal_likelihood(self, log_theta, X, y):
        """
        -log p(y | X, theta), con beta concentrada por mínimos cuadrados
        generalizados. Se parametriza en logaritmos para que la optimización sea
        libre y los hiperparámetros queden garantizadamente positivos.
        """
        theta = np.exp(log_theta)
        kernel_parameters = theta[:-1]
        # Cota inferior del ruido. Sin ella el kernel exponencial converge a
        # sigma_n = 0, es decir, a interpolar exactamente los datos de
        # entrenamiento: la verosimilitud marginal crece sin límite y la matriz
        # A queda casi singular. La cota mantiene el problema bien planteado.
        sigma_n = max(theta[-1], self._sigma_floor)
        n = len(y)
        try:
            K = self.kernel_function(X, X, kernel_parameters)
            A = K + (sigma_n ** 2 + JITTER) * np.eye(n)
            L = cholesky(A)
        except (ValueError, FloatingPointError):
            return 1e10                   # hiperparámetros inadmisibles

        ones = np.ones(n)
        A_inverse_ones = cholesky_solve(L, ones)
        A_inverse_y = cholesky_solve(L, y)
        denominator = ones @ A_inverse_ones
        beta = (ones @ A_inverse_y) / denominator

        residual = y - beta
        A_inverse_residual = cholesky_solve(L, residual)
        log_determinant = 2.0 * np.sum(np.log(np.diag(L)))

        nlml = 0.5 * (residual @ A_inverse_residual
                      + log_determinant
                      + n * np.log(2.0 * np.pi))
        return float(nlml) if np.isfinite(nlml) else 1e10

    # -- ajuste y predicción ------------------------------------------------

    def fit(self, X, y, restarts=3, seed=0, verbose=False):
        """
        Ajusta los hiperparámetros maximizando la verosimilitud marginal.

        Se hacen varios reinicios desde puntos de partida distintos porque la
        verosimilitud marginal no es convexa y BFGS puede quedar atrapado en un
        óptimo local.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        generator = np.random.default_rng(seed)
        # Ruido mínimo admisible: una milésima de la dispersión del objetivo.
        self._sigma_floor = 1e-3 * float(y.std())

        # Punto de partida con sentido físico: escala de longitud del orden de
        # la distancia media entre puntos, amplitud del orden de la desviación
        # típica del objetivo, y ruido un orden de magnitud menor.
        median_distance = np.median(_pairwise_distances(X, X))
        base = [np.log(y.std() + 1e-8), np.log(median_distance + 1e-8)]
        if self._n_kernel_parameters == 3:
            base.append(np.log(1.0))      # alpha inicial
        base.append(np.log(0.1 * y.std() + 1e-8))
        base = np.array(base)

        best_theta, best_value = None, np.inf
        for restart in range(restarts):
            start = base if restart == 0 else base + generator.normal(0, 0.7, len(base))
            theta, value = bfgs_minimize(
                lambda t: self._negative_log_marginal_likelihood(t, X, y),
                start, max_iterations=150, verbose=verbose)
            if value < best_value:
                best_theta, best_value = theta, value

        theta = np.exp(best_theta)
        self.kernel_parameters = theta[:-1]
        self.sigma_n = float(max(theta[-1], self._sigma_floor))

        # Precomputación para predecir en O(n) por punto.
        n = len(y)
        K = self.kernel_function(X, X, self.kernel_parameters)
        A = K + (self.sigma_n ** 2 + JITTER) * np.eye(n)
        self._L = cholesky(A)
        ones = np.ones(n)
        self.beta = float((ones @ cholesky_solve(self._L, y))
                          / (ones @ cholesky_solve(self._L, ones)))
        self._alpha_vector = cholesky_solve(self._L, y - self.beta)
        self._X = X
        self.log_marginal_likelihood = -best_value
        return self

    def predict(self, X, return_std=False):
        """Media predictiva (y opcionalmente desviación típica) en X."""
        X = np.asarray(X, dtype=float)
        K_star = self.kernel_function(X, self._X, self.kernel_parameters)
        mean = self.beta + K_star @ self._alpha_vector
        if not return_std:
            return mean
        # var = k** - k*' A^-1 k*.  Con A = L L' basta resolver L V = K*' una
        # sola vez para todos los puntos y sumar V columna a columna.
        V = _forward_substitution(self._L, K_star.T)
        prior_variance = self.kernel_function(
            X[:1], X[:1], self.kernel_parameters)[0, 0]
        variance = np.maximum(prior_variance - np.sum(V ** 2, axis=0), 0.0)
        return mean, np.sqrt(variance)

    def structure(self):
        """Información equivalente a la Table 11 del paper."""
        info = {
            "Model type": f"{self.kernel_name} GPR",
            "Basis function": "Constant",
            "Sigma": self.sigma_n,
            "Beta": self.beta,
            "Optimizer": "Quasinewton (BFGS)",
            "Signal std (sigma_f)": float(self.kernel_parameters[0]),
            "Kernel scale (length scale)": float(self.kernel_parameters[1]),
        }
        if self._n_kernel_parameters == 3:
            info["Alpha"] = float(self.kernel_parameters[2])
        return info
