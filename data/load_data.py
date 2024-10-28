# load_data.py
from sklearn.datasets import load_iris, fetch_openml
import numpy as np

def load_iris_data():
    """
    Carga y devuelve el dataset de Iris.
    """
    iris = load_iris()
    X = iris['data']
    y = iris['target'].reshape(-1, 1)
    return X, y

def load_mnist_data():
    """
    Carga y devuelve el dataset de MNIST.
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist['data'].values.reshape(-1, 1, 28, 28) / 255.0  
    y = mnist['target'].astype(np.int64).values.reshape(-1, 1)
    
    return X, y
