import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, X):
        self.Z = np.dot(X, self.weights) + self.biases
        return self.activation(self.Z)

    def backward(self, dA, A_prev, learning_rate):
        dZ = dA * self.activation_derivative(self.Z)
        dW = np.dot(A_prev.T, dZ) / A_prev.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / A_prev.shape[0]
        dA_prev = np.dot(dZ, self.weights.T)

        # Actualizar pesos y bias
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        return dA_prev
