"""
Partición aleatoria del conjunto de datos en entrenamiento/validación/test.

El paper (Sections 2.1 y 3.1) usa el comando `dividerand` de MATLAB con las
proporciones 70% / 15% / 15%, lo que sobre las 565 muestras de EEDB-09/2019 da
395 de entrenamiento, 85 de validación y 85 de test (Table 15). Los mismos
índices se reutilizan para los tres métodos (GPR, SVM y MLP) para que la
comparación de la Table 16 sea justa.
"""

import numpy as np

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


def divide_random(n_samples, seed=0):
    """
    Reparte los índices 0..n-1 en tres bloques disjuntos.

    Los tamaños se calculan igual que `dividerand`: se redondea el bloque de
    entrenamiento y el de validación, y el resto va a test.
    """
    generator = np.random.default_rng(seed)
    order = generator.permutation(n_samples)

    n_train = int(round(TRAIN_FRACTION * n_samples))
    n_validation = int(round(VALIDATION_FRACTION * n_samples))

    return (order[:n_train],
            order[n_train:n_train + n_validation],
            order[n_train + n_validation:])


class Standardizer:
    """
    Normalización z-score, ajustada sólo con el bloque de entrenamiento.

    Tanto `fitrgp` como `fitrsvm` de MATLAB estandarizan los predictores por
    defecto en la Regression Learner App. Es imprescindible aquí: el empuje
    nominal ronda 5e2 y la humedad 7e-3, cinco órdenes de magnitud de
    diferencia, lo que haría inservible cualquier kernel basado en distancias.
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0, ddof=1)
        # Una columna constante tendría escala 0; se deja en 1 para no dividir
        # por cero (la columna resultante será toda ceros).
        self.scale[self.scale == 0.0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Reescalado lineal al intervalo [-1, 1], equivalente a `mapminmax` de MATLAB.

    Es el preprocesamiento que aplican por defecto las redes de la Neural Network
    Toolbox, tanto a las entradas como al objetivo, y por eso se usa en el MLP.
    """

    def __init__(self, lower=-1.0, upper=1.0):
        self.lower = lower
        self.upper = upper
        self.minimum = None
        self.range = None

    def fit(self, X):
        X = np.atleast_2d(X)
        self.minimum = X.min(axis=0)
        maximum = X.max(axis=0)
        self.range = maximum - self.minimum
        self.range[self.range == 0.0] = 1.0
        return self

    def transform(self, X):
        span = self.upper - self.lower
        return self.lower + span * (X - self.minimum) / self.range

    def inverse_transform(self, Z):
        span = self.upper - self.lower
        return self.minimum + self.range * (Z - self.lower) / span

    def fit_transform(self, X):
        return self.fit(X).transform(X)
