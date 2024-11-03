# load_data.py
from sklearn.datasets import load_iris, load_digits, load_wine

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

def load_wine_data():
    """
    Carga y devuelve el dataset de vinos.
    """
    data = load_wine()
    X, y = data.data, data.target.reshape(-1, 1)
    return X, y

