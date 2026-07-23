"""
Perceptrón multicapa (Section 2.3.3 del paper), implementado desde cero.

Arquitectura de la Table 15 y la Fig. 8 del paper:

    entrada(7) -> 10 -> 8 -> 4 -> 3 -> salida(1)
    activaciones:  tansig, logsig, tansig, purelin,  salida purelin

El paper entrena 9 modelos (Table 14) con tres algoritmos de la Neural Network
Toolbox de MATLAB, los tres implementados aquí:

    trainlm   - Levenberg-Marquardt (el que da el mejor modelo, "Model 1")
    trainrp   - retropropagación resiliente (Rprop)
    traincgb  - gradiente conjugado con reinicios de Powell-Beale

Los pesos se inicializan con el método de Nguyen-Widrow, que es el que usa
MATLAB por defecto (`initnw`), y las entradas y el objetivo se reescalan a
[-1, 1] con `mapminmax` (véase `splits.MinMaxScaler`).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Funciones de activación y sus derivadas, expresadas en función de la salida a,
# que es como se necesitan durante la retropropagación.
# ---------------------------------------------------------------------------

def tansig(x):
    """Tangente hiperbólica de MATLAB: 2/(1+exp(-2x)) - 1, idéntica a tanh(x)."""
    return np.tanh(x)


def tansig_derivative(a):
    """d/dx tanh(x) = 1 - tanh(x)^2"""
    return 1.0 - a ** 2


def logsig(x):
    """Sigmoide logística: 1/(1+exp(-x))."""
    # Formulación estable: para x muy negativo exp(-x) desbordaría.
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def logsig_derivative(a):
    """d/dx logsig(x) = a(1-a)"""
    return a * (1.0 - a)


def purelin(x):
    """Identidad."""
    return x


def purelin_derivative(a):
    """d/dx x = 1"""
    return np.ones_like(a)


ACTIVATIONS = {
    "tansig":  (tansig, tansig_derivative),
    "logsig":  (logsig, logsig_derivative),
    "purelin": (purelin, purelin_derivative),
}

# Arquitectura exacta del paper (Table 15).
PAPER_LAYERS = [10, 8, 4, 3]
PAPER_ACTIVATIONS = ["tansig", "logsig", "tansig", "purelin"]


class MultilayerPerceptron:
    """
    Red feed-forward completamente conectada.

    Los pesos se almacenan como lista de matrices (n_entradas, n_neuronas) y
    sesgos como vectores, pero para los algoritmos de segundo orden se
    aplanan en un único vector de parámetros.
    """

    def __init__(self, n_inputs=7, hidden_layers=None, activations=None,
                 seed=0):
        hidden_layers = PAPER_LAYERS if hidden_layers is None else hidden_layers
        activations = PAPER_ACTIVATIONS if activations is None else activations
        if len(activations) != len(hidden_layers):
            raise ValueError("una activación por capa oculta")

        self.sizes = [n_inputs] + list(hidden_layers) + [1]
        # La capa de salida es lineal ("Function: Linear, Output layer").
        self.activations = list(activations) + ["purelin"]
        self.rng = np.random.default_rng(seed)
        self.weights, self.biases = self._initialize_nguyen_widrow()

    # -- inicialización -----------------------------------------------------

    def _initialize_nguyen_widrow(self):
        """
        Inicialización de Nguyen-Widrow (`initnw` de MATLAB).

        Reparte las regiones activas de las neuronas de forma aproximadamente
        uniforme sobre el espacio de entrada [-1, 1], en lugar de dejarlas al
        azar. Esto acelera notablemente la convergencia frente a una
        inicialización puramente aleatoria.
        """
        weights, biases = [], []
        for layer in range(len(self.sizes) - 1):
            n_in, n_out = self.sizes[layer], self.sizes[layer + 1]

            W = self.rng.uniform(-1.0, 1.0, size=(n_in, n_out))
            if self.activations[layer] == "purelin":
                # Una capa lineal no tiene región activa que repartir.
                weights.append(W * 0.5)
                biases.append(np.zeros(n_out))
                continue

            # Magnitud objetivo del vector de pesos de cada neurona.
            beta = 0.7 * n_out ** (1.0 / n_in)
            norms = np.linalg.norm(W, axis=0)
            norms[norms == 0.0] = 1.0
            W = beta * W / norms

            # Sesgos escalonados: desplazan el punto de inflexión de cada
            # neurona a lo largo del intervalo de entrada.
            if n_out == 1:
                b = np.zeros(1)
            else:
                b = beta * np.linspace(-1.0, 1.0, n_out) * np.sign(W[0, :])
            weights.append(W)
            biases.append(b)
        return weights, biases

    # -- (des)empaquetado de parámetros ------------------------------------

    def get_parameters(self):
        """Aplana pesos y sesgos en un único vector."""
        pieces = []
        for W, b in zip(self.weights, self.biases):
            pieces.append(W.ravel())
            pieces.append(b)
        return np.concatenate(pieces)

    def set_parameters(self, vector):
        """Operación inversa de `get_parameters`."""
        offset = 0
        for layer in range(len(self.weights)):
            n_in, n_out = self.sizes[layer], self.sizes[layer + 1]
            size = n_in * n_out
            self.weights[layer] = vector[offset:offset + size].reshape(n_in, n_out)
            offset += size
            self.biases[layer] = vector[offset:offset + n_out]
            offset += n_out

    @property
    def n_parameters(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    # -- propagación --------------------------------------------------------

    def _forward(self, X):
        """
        Pasada hacia adelante que devuelve la lista de activaciones de todas
        las capas (incluida la entrada), necesaria para la retropropagación.
        """
        activations = [X]
        current = X
        for layer, (W, b) in enumerate(zip(self.weights, self.biases)):
            function, _ = ACTIVATIONS[self.activations[layer]]
            current = function(current @ W + b)
            activations.append(current)
        return activations

    def predict(self, X):
        """Salida de la red (en el espacio escalado [-1, 1])."""
        return self._forward(np.asarray(X, dtype=float))[-1].ravel()

    # -- jacobiano y gradiente ---------------------------------------------

    def _jacobian(self, X, y):
        """
        Jacobiano de los residuos e = salida - objetivo respecto a todos los
        parámetros, junto con el propio vector de residuos.

        Con una única neurona de salida, cada residuo depende sólo de su propia
        muestra, de modo que la retropropagación se vectoriza sobre las N
        muestras a la vez: `delta[l]` tiene forma (N, n_l) y contiene
        de_n / d(entrada neta de la capa l) para la muestra n.
        """
        activations = self._forward(X)
        residuals = activations[-1].ravel() - y
        n_samples = X.shape[0]

        n_layers = len(self.weights)
        deltas = [None] * n_layers

        # Capa de salida: de/da = 1, y a = f(net) con f la activación de salida.
        _, derivative = ACTIVATIONS[self.activations[-1]]
        deltas[-1] = derivative(activations[-1])          # (N, 1)

        # Retropropagación hacia las capas anteriores.
        for layer in range(n_layers - 2, -1, -1):
            _, derivative = ACTIVATIONS[self.activations[layer]]
            propagated = deltas[layer + 1] @ self.weights[layer + 1].T
            deltas[layer] = propagated * derivative(activations[layer + 1])

        # Montaje del jacobiano: de_n/dW[i,j] = a_entrada[n,i] * delta[n,j]
        columns = []
        for layer in range(n_layers):
            input_activation = activations[layer]          # (N, n_in)
            delta = deltas[layer]                          # (N, n_out)
            # Producto exterior por muestra, aplanado en (N, n_in*n_out).
            block = (input_activation[:, :, None] * delta[:, None, :])
            columns.append(block.reshape(n_samples, -1))
            columns.append(delta)                          # derivadas del sesgo
        return np.hstack(columns), residuals

    def _gradient(self, X, y):
        """Gradiente del error cuadrático medio y su valor."""
        J, residuals = self._jacobian(X, y)
        n = len(residuals)
        gradient = 2.0 * (J.T @ residuals) / n
        return gradient, float(np.mean(residuals ** 2))

    def mse(self, X, y):
        return float(np.mean((self.predict(X) - y) ** 2))

    # -- algoritmos de entrenamiento ---------------------------------------

    def train_lm(self, X, y, X_validation=None, y_validation=None,
                 epochs=1000, mu=1e-3, mu_increase=10.0, mu_decrease=0.1,
                 mu_max=1e10, max_fail=6):
        """
        Levenberg-Marquardt (`trainlm`).

        Resuelve en cada paso  (J'J + mu*I) dp = -J'e  y acepta el paso sólo si
        el error baja. mu interpola entre Gauss-Newton (mu pequeño, convergencia
        cuadrática cerca del óptimo) y descenso de gradiente (mu grande, robusto
        lejos de él). Los valores por defecto son los de MATLAB.

        La parada temprana por validación (`max_fail` = 6 épocas consecutivas
        sin mejora) es la que emplea la Neural Network Toolbox, y es la razón por
        la que el paper reserva un 15% de los datos para validación.
        """
        best_parameters = self.get_parameters().copy()
        best_validation = np.inf
        failures = 0
        history = []

        for epoch in range(epochs):
            J, residuals = self._jacobian(X, y)
            error = float(np.mean(residuals ** 2))
            history.append(error)

            JtJ = J.T @ J
            Jte = J.T @ residuals
            identity = np.eye(self.n_parameters)
            parameters = self.get_parameters()

            improved = False
            while mu <= mu_max:
                try:
                    step = np.linalg.solve(JtJ + mu * identity, -Jte)
                except np.linalg.LinAlgError:
                    mu *= mu_increase
                    continue
                self.set_parameters(parameters + step)
                new_error = self.mse(X, y)
                if new_error < error:
                    mu = max(mu * mu_decrease, 1e-20)
                    improved = True
                    break
                mu *= mu_increase
            if not improved:
                self.set_parameters(parameters)
                break                       # mu saturado: no se puede mejorar

            # Parada temprana sobre el conjunto de validación.
            if X_validation is not None:
                validation_error = self.mse(X_validation, y_validation)
                if validation_error < best_validation:
                    best_validation = validation_error
                    best_parameters = self.get_parameters().copy()
                    failures = 0
                else:
                    failures += 1
                    if failures >= max_fail:
                        break

        if X_validation is not None and np.isfinite(best_validation):
            self.set_parameters(best_parameters)
        return history

    def train_rp(self, X, y, X_validation=None, y_validation=None,
                 epochs=1000, delta0=0.07, delta_max=50.0,
                 increase=1.2, decrease=0.5, max_fail=6):
        """
        Retropropagación resiliente (`trainrp`).

        Usa sólo el **signo** del gradiente: cada parámetro tiene su propio paso,
        que crece mientras el signo se mantiene y se reduce cuando cambia. Al
        ignorar la magnitud del gradiente evita el estancamiento típico de las
        sigmoides saturadas. Los valores por defecto son los de MATLAB.
        """
        step_size = np.full(self.n_parameters, delta0)
        previous_gradient = np.zeros(self.n_parameters)
        best_parameters = self.get_parameters().copy()
        best_validation = np.inf
        failures = 0
        history = []

        for epoch in range(epochs):
            gradient, error = self._gradient(X, y)
            history.append(error)

            product = gradient * previous_gradient
            step_size = np.where(product > 0,
                                 np.minimum(step_size * increase, delta_max),
                                 np.where(product < 0,
                                          np.maximum(step_size * decrease, 1e-6),
                                          step_size))
            # Si el gradiente cambió de signo se anula el gradiente memorizado
            # para no volver a reducir el paso en la iteración siguiente.
            gradient = np.where(product < 0, 0.0, gradient)

            parameters = self.get_parameters()
            self.set_parameters(parameters - np.sign(gradient) * step_size)
            previous_gradient = gradient

            if X_validation is not None:
                validation_error = self.mse(X_validation, y_validation)
                if validation_error < best_validation:
                    best_validation = validation_error
                    best_parameters = self.get_parameters().copy()
                    failures = 0
                else:
                    failures += 1
                    if failures >= max_fail:
                        break

        if X_validation is not None and np.isfinite(best_validation):
            self.set_parameters(best_parameters)
        return history

    def train_cgb(self, X, y, X_validation=None, y_validation=None,
                  epochs=1000, max_fail=6):
        """
        Gradiente conjugado con reinicios de Powell-Beale (`traincgb`).

        La dirección de búsqueda combina el gradiente actual con la anterior
        mediante el coeficiente de Polak-Ribiere. El criterio de Powell-Beale
        reinicia la dirección cuando queda poca ortogonalidad entre gradientes
        consecutivos:

            |g_{k-1} · g_k|  >=  0.2 * ||g_k||^2

        La longitud de paso se elige por retroceso con la condición de Armijo.
        """
        gradient, error = self._gradient(X, y)
        direction = -gradient
        best_parameters = self.get_parameters().copy()
        best_validation = np.inf
        failures = 0
        history = []

        for epoch in range(epochs):
            history.append(error)
            parameters = self.get_parameters()
            slope = float(gradient @ direction)
            if slope >= 0:
                direction = -gradient
                slope = float(gradient @ direction)

            # Búsqueda lineal por retroceso (condición de Armijo).
            step, accepted = 1.0, False
            for _ in range(40):
                self.set_parameters(parameters + step * direction)
                new_error = self.mse(X, y)
                if new_error <= error + 1e-4 * step * slope:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                self.set_parameters(parameters)
                break

            new_gradient, new_error = self._gradient(X, y)

            # Reinicio de Powell-Beale.
            overlap = abs(float(new_gradient @ gradient))
            if overlap >= 0.2 * float(new_gradient @ new_gradient):
                direction = -new_gradient
            else:
                difference = new_gradient - gradient
                denominator = float(gradient @ gradient)
                beta = (float(new_gradient @ difference) / denominator
                        if denominator > 0 else 0.0)
                direction = -new_gradient + beta * direction

            gradient, error = new_gradient, new_error

            if X_validation is not None:
                validation_error = self.mse(X_validation, y_validation)
                if validation_error < best_validation:
                    best_validation = validation_error
                    best_parameters = self.get_parameters().copy()
                    failures = 0
                else:
                    failures += 1
                    if failures >= max_fail:
                        break

        if X_validation is not None and np.isfinite(best_validation):
            self.set_parameters(best_parameters)
        return history

    def structure(self):
        """Información equivalente a la Table 15 del paper."""
        return {
            "Layers": f"{self.sizes[0]} - "
                      + "-".join(str(s) for s in self.sizes[1:-1])
                      + f" - {self.sizes[-1]}",
            "Activation": " - ".join(f"'{a}'" for a in self.activations),
            "Parameters": self.n_parameters,
        }


TRAINERS = {
    "trainlm":  MultilayerPerceptron.train_lm,
    "trainrp":  MultilayerPerceptron.train_rp,
    "traincgb": MultilayerPerceptron.train_cgb,
}
