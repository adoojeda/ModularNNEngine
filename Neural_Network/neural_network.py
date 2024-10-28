from Neural_Network.conv import conv2d, max_pooling
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.fc1_weights = np.random.randn(4, 64) * 0.1  # Iris tiene 4 características
        self.fc2_weights = np.random.randn(64, 32) * 0.1
        self.fc3_weights = np.random.randn(32, 3) * 0.1  # 3 clases de salida

    def forward(self, X):
        self.fc1_out = np.maximum(0, np.dot(X, self.fc1_weights))  # ReLU activations
        self.fc2_out = np.maximum(0, np.dot(self.fc1_out, self.fc2_weights))  # ReLU activations
        self.fc3_out = np.dot(self.fc2_out, self.fc3_weights)
        return self.softmax(self.fc3_out)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        logprobs = -np.log(y_pred + 1e-10)
        loss = np.sum(y_true * logprobs) / m
        return loss

    def backpropagation(self, X, y_true, learning_rate):
        m = y_true.shape[0]

        dZ3 = self.fc3_out - y_true
        dW3 = np.dot(self.fc2_out.T, dZ3) / m
        self.fc3_weights -= learning_rate * dW3

        dA2 = np.dot(dZ3, self.fc3_weights.T)
        dZ2 = dA2 * (self.fc2_out > 0)
        dW2 = np.dot(self.fc1_out.T, dZ2) / m
        self.fc2_weights -= learning_rate * dW2

        dA1 = np.dot(dZ2, self.fc2_weights.T)
        dZ1 = dA1 * (self.fc1_out > 0)
        dW1 = np.dot(X.T, dZ1) / m
        self.fc1_weights -= learning_rate * dW1
        
    def evaluate(self, X, y_true):
        y_pred = self.forward(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_pred_classes == y_true_classes)
    
    def train(self, X_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.forward(X_train)
            loss = self.compute_loss(y_train, output)
            self.backpropagation(X_train, y_train, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
            print(f"Accuracy: {self.evaluate(X_train, y_train)}")

    
