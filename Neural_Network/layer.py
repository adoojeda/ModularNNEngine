# layer.py
import numpy as np

class Layer:
    def __init__(self, num_inputs, num_units, activation_func, activation_derivate):
        """
        Inicializa la capa de la red neuronal.

        :param num_inputs: Número de neuronas en la capa anterior.
        :param num_units: Número de neuronas en esta capa.
        :param activation_func: Función de activación a utilizar.
        :param activation_derivate: Derivada de la función de activación.
        """
        self.num_units = num_units
        self.weights = np.random.randn(num_units, num_inputs)  # Inicialización de He
        self.bias = np.random.randn(num_units, 1)
        self.activation_func = activation_func
        self.activation_derivate = activation_derivate

    def forward(self, A_prev):
        """
        Realiza la propagación hacia adelante.

        :param A_prev: Salida de la capa anterior.
        :return: Salida de esta capa después de aplicar la activación.
        """
        self.Z = self.compute_z(A_prev)
        return self.activation_func(self.Z)

    def compute_z(self, A_prev):
        """
        Calcula el valor Z, que es la suma ponderada más el sesgo.

        :param A_prev: Salida de la capa anterior.
        :return: Valor Z de esta capa.
        """
        return np.dot(self.weights, A_prev).reshape(self.num_units,1) + self.bias
