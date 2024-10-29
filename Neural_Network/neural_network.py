# neural_network.py
import numpy as np

class NeuralNetwork:
    def __init__(self):
        """Inicializa la red neuronal."""
        self.layers = []

    def add_layer(self, layer):
        """
        Añade una capa a la red neuronal.

        :param layer: Instancia de la clase Layer a añadir.
        """
        self.layers.append(layer)

    def feedforward(self, X, n=None):
        """
        Realiza la propagación hacia adelante.

        :param X: Datos de entrada.
        :param n: Número de capas a considerar en la propagación.
        :return: Salida de la red neuronal.
        """
        n = len(self.layers) if n is None else n

        for layer in self.layers[:n]:
            X = layer.forward(X)
        return X
    
    def backpropagation(self, X, y):
        """
        Realiza la retropropagación para calcular los gradientes.

        :param X: Datos de entrada.
        :param y: Etiqueta verdadera.
        :return: Lista de gradientes para cada capa.
        """

        derivates = []

        A = self.feedforward(X)
        A = A.reshape(len(A),1)
        y = y.reshape(len(y),1)

        A_prev = self.feedforward(X, -1)
        A_prev = A_prev.reshape(len(A_prev),1)

        dz = self.layers[-1].activation_derivate(A, y)
        dW = dz.dot(A_prev.T)

        derivates.append((dz, dW))

        for k, layer in enumerate(self.layers[-2::-1]):
            A_prev = self.feedforward(X, -k-2)
            A_prev = A_prev.reshape(len(A_prev),1)

            derivate = layer.activation_derivate(layer.compute_z(A_prev))
            dz = derivate*self.layers[-k-1].weights.T.dot(dz)
            dW = dz.dot(A_prev.T)

            derivates.append((dz, dW))

        return derivates

    def compute_loss(self, y_true, y_pred):
        """
        Calcula la pérdida de la red neuronal usando la entropía cruzada.

        :param y_true: Etiquetas verdaderas.
        :param y_pred: Predicciones de la red.
        :return: Valor de la pérdida.
        """
        m = y_true.shape[0]
        logprobs = -np.log(y_pred + 1e-10)  # Evita log(0)
        loss = np.sum(y_true * logprobs) / m
        return loss
    
    def accuracy(Y_pred, Y):
        """
        Calcula la precisión de las predicciones.

        :param Y_pred: Predicciones de la red.
        :param Y: Etiquetas verdaderas.
        :return: Precisión como porcentaje.
        """
        acc = 0
        for y_pred, y in zip(Y_pred, Y):
            if np.argmax(y_pred) == np.argmax(y):
                acc += 1

        return acc / len(Y_pred)
