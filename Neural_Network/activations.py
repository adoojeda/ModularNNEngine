import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    """Derivative of the sigmoid function."""
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivate(x):
    """Derivative of the ReLU function."""
    return (x > 0).astype(float)

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def tanh_derivate(x):
    """Derivative of the Tanh function."""
    return 1 - np.tanh(x)**2

def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x, axis=0)

def softmax_derivate(y_pred, y_true):
    """Derivada de la función softmax."""
    return y_pred - y_true

