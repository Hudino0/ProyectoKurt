"""
Distribuciones estadísticas implementadas desde cero.

Se necesitan los p-valores de la t de Student (columna "Sig." de la Table 9) y
de la F de Fisher (columna "Sig." de la tabla ANOVA, Table 8). Ambos se obtienen
de la función beta incompleta regularizada I_x(a, b), que aquí se evalúa con la
fracción continua de Lentz — el método clásico de *Numerical Recipes*.

Se evita SciPy deliberadamente: el encargo es construir todo desde cero.
"""

import math


def _log_gamma(x):
    """
    Logaritmo de la función gamma por la aproximación de Lanczos (g=7, n=9).
    Precisión relativa mejor que 1e-15 para x > 0, más que suficiente aquí.
    """
    coefficients = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    if x < 0.5:
        # Fórmula de reflexión, para mantener el argumento en la zona estable.
        return (math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1.0 - x))
    x -= 1.0
    a = coefficients[0]
    t = x + 7.5
    for i in range(1, 9):
        a += coefficients[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _beta_continued_fraction(a, b, x):
    """Fracción continua de I_x(a,b), evaluada con el algoritmo de Lentz."""
    tiny, epsilon, max_iterations = 1e-30, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        # Paso par de la fracción continua.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # Paso impar.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def incomplete_beta(a, b, x):
    """Función beta incompleta regularizada I_x(a, b), con 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(_log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    # La fracción continua sólo converge rápido en una mitad del dominio;
    # en la otra se usa la simetría I_x(a,b) = 1 - I_{1-x}(b,a).
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
        + b * math.log(1.0 - x) + a * math.log(x)
    ) * _beta_continued_fraction(b, a, 1.0 - x) / b


def t_two_sided_p(t_statistic, degrees_of_freedom):
    """
    p-valor bilateral de la t de Student. Es la columna "Sig." de la Table 9.

        p = I_{df/(df + t^2)}(df/2, 1/2)
    """
    df = float(degrees_of_freedom)
    t2 = float(t_statistic) ** 2
    return incomplete_beta(df / 2.0, 0.5, df / (df + t2))


def f_upper_p(f_statistic, df_numerator, df_denominator):
    """
    p-valor de cola superior de la F de Fisher. Es la columna "Sig." de la
    tabla ANOVA (Table 8), donde el paper reporta 0.000.

        p = I_{d2/(d2 + d1*F)}(d2/2, d1/2)
    """
    d1, d2 = float(df_numerator), float(df_denominator)
    f = float(f_statistic)
    if f <= 0.0:
        return 1.0
    return incomplete_beta(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))
