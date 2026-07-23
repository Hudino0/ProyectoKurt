"""
Optimizador cuasi-Newton BFGS, implementado desde cero.

La Table 11 del paper indica que el modelo GPR se ajustó con el optimizador
"Quasinewton" de MATLAB, que es exactamente BFGS. Se usa aquí para maximizar la
log-verosimilitud marginal del proceso gaussiano sobre sus hiperparámetros.

La aproximación de la inversa del hessiano se actualiza con la fórmula BFGS y la
longitud de paso se elige con una búsqueda lineal que satisface las condiciones
de Wolfe (retroceso con comprobación de curvatura).
"""

import numpy as np


def numerical_gradient(function, x, step=1e-5):
    """
    Gradiente por diferencias centradas.

    El paso es relativo a la magnitud de cada componente, lo que mantiene la
    precisión cuando los hiperparámetros viven en escalas distintas.
    """
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        h = step * max(abs(x[i]), 1.0)
        forward, backward = x.copy(), x.copy()
        forward[i] += h
        backward[i] -= h
        gradient[i] = (function(forward) - function(backward)) / (2.0 * h)
    return gradient


def _line_search(function, x, value, gradient, direction,
                 c1=1e-4, c2=0.9, max_steps=40):
    """
    Búsqueda lineal por retroceso con condiciones de Wolfe.

    c1 controla el descenso suficiente (Armijo) y c2 la condición de curvatura.
    Devuelve (alpha, x_nuevo, valor_nuevo) o None si no encuentra un paso válido.
    """
    slope = float(gradient @ direction)
    if slope >= 0:
        return None                       # la dirección no es de descenso
    alpha = 1.0
    for _ in range(max_steps):
        candidate = x + alpha * direction
        candidate_value = function(candidate)
        if np.isfinite(candidate_value) and \
                candidate_value <= value + c1 * alpha * slope:
            # Descenso suficiente conseguido; se acepta el paso.
            return alpha, candidate, candidate_value
        alpha *= 0.5
    return None


def bfgs_minimize(function, x0, max_iterations=200, tolerance=1e-6, verbose=False):
    """
    Minimiza `function` (escalar, de un vector) partiendo de `x0`.

    Devuelve (x_optimo, valor_optimo).
    """
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    value = function(x)
    gradient = numerical_gradient(function, x)
    inverse_hessian = np.eye(n)

    for iteration in range(max_iterations):
        if np.linalg.norm(gradient) < tolerance:
            break

        direction = -inverse_hessian @ gradient
        result = _line_search(function, x, value, gradient, direction)
        if result is None:
            # La dirección BFGS falló: se reinicia con descenso de gradiente,
            # que siempre es de descenso si el gradiente no es nulo.
            inverse_hessian = np.eye(n)
            direction = -gradient
            result = _line_search(function, x, value, gradient, direction)
            if result is None:
                break

        alpha, x_new, value_new = result
        gradient_new = numerical_gradient(function, x_new)

        s = x_new - x                      # desplazamiento
        y = gradient_new - gradient        # cambio del gradiente
        curvature = float(y @ s)

        # Sólo se actualiza si se cumple la condición de curvatura; en caso
        # contrario la inversa dejaría de ser definida positiva.
        if curvature > 1e-10:
            rho = 1.0 / curvature
            identity = np.eye(n)
            left = identity - rho * np.outer(s, y)
            right = identity - rho * np.outer(y, s)
            inverse_hessian = (left @ inverse_hessian @ right
                               + rho * np.outer(s, s))

        if verbose:
            print(f"  iter {iteration:3d}  f={value_new:.8g}  "
                  f"|g|={np.linalg.norm(gradient_new):.3g}")

        if abs(value - value_new) < tolerance * (1.0 + abs(value)):
            x, value = x_new, value_new
            break

        x, value, gradient = x_new, value_new, gradient_new

    return x, value
