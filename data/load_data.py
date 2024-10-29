# load_data.py
import gzip
import os
from sklearn.datasets import load_iris, fetch_openml, load_digits
import numpy as np
import urllib

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
    Carga y devuelve el dataset MNIST.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data
    y = mnist.target
    y = y.astype(int)
    return X, y

def load_digits_data():
    """
    Carga y devuelve el dataset de dígitos.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    return X, y
