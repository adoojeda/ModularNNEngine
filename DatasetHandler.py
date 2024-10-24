import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class DatasetHandler:
    def __init__(self):
        # Cargar el dataset Iris
        iris = load_iris()
        self.X = iris.data  # Features
        self.y = iris.target  # Labels

        # Normalizar el dataset
        self.X = self.normalize(self.X)

        # Dividir el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def normalize(self, X):
        # Normalización por escala entre 0 y 1
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return (X - X_min) / (X_max - X_min)


# Uso del DatasetHandler
#dataset = DatasetHandler()
# Imprimir los datos normalizados y sus dimensiones
#print("Características normalizadas (X_train):")
#print(dataset.X_train)
#print("Etiquetas (y_train):", dataset.y_train)
#print("Tamaño del conjunto de entrenamiento:", dataset.X_train.shape)
#print("Tamaño del conjunto de prueba:", dataset.X_test.shape)
