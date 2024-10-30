# visualizations.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def plot_confusion_matrix(y_true, y_pred_classes):
    """
    Muestra la matriz de confusión.

    :param y_true: Etiquetas verdaderas.
    :param y_pred_classes: Etiquetas predichas por el modelo.
    """
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Matriz de Confusión')
    plt.show()

def scatter_roc_curve(nn, X_test, y_test):
    """
    Función que grafica la curva ROC para un modelo de red neuronal.

    Parámetros:
    nn: instancia de la clase NeuralNetwork.
    X_test: matriz de características de prueba.
    y_test: matriz de etiquetas de prueba.
    """

    if len(y_test.shape) == 1:
        encoder = OneHotEncoder(sparse_output=False)
        y_test = encoder.fit_transform(y_test.reshape(-1, 1))

    y_probs = np.array([nn.feedforward(x) for x in X_test])

    plt.figure(figsize=(8, 6))
    
    for i in range(y_test.shape[1]):
        y_true_binary = y_test[:, i]
        y_probs_binary = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs_binary)
        
        plt.scatter(fpr, tpr, label=f'Clase {i}', s=10)  # Ajusta el tamaño de los puntos según prefieras

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Gráfica de Dispersión')
    plt.legend()
    plt.show()