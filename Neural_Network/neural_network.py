import numpy as np
import pickle  
from Neural_Network.layer import Layer

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation, activation_derivative):
        layer = Layer(input_size, output_size, activation, activation_derivative)
        self.layers.append(layer)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        logprobs = -np.log(y_pred + 1e-10)
        loss = np.sum(y_true * logprobs) / m
        return loss

    def backpropagation(self, X, y_true, learning_rate):
        m = y_true.shape[0]
        A = self.forward(X)
        dA = A - y_true

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = X if i == 0 else self.layers[i - 1].forward(X)
            dA = layer.backward(dA, A_prev, learning_rate)

    def train(self, X_train, y_train, learning_rate=0.01, epochs=10):
        for epoch in range(epochs):
            output = self.forward(X_train)
            loss = self.compute_loss(y_train, output)
            self.backpropagation(X_train, y_train, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def evaluate(self, X, y_true):
        y_pred = self.forward(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_pred_classes == y_true_classes)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)