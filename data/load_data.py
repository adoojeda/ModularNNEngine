# load_data.py
from sklearn.datasets import load_iris, load_digits

def load_iris_data():
    """
    Carga y devuelve el dataset de Iris.
    """
    iris = load_iris()
    X = iris['data']
    y = iris['target'].reshape(-1, 1)
    return X, y

def load_digits_data():
    """
    Carga y devuelve el dataset de dígitos.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    return X, y
