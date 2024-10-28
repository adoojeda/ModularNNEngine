# load_data.py
from sklearn.datasets import load_iris

def load_iris_data():
    """
    Carga y devuelve el dataset de Iris.
    """
    iris = load_iris()
    X = iris['data']
    y = iris['target'].reshape(-1, 1)
    return X, y
