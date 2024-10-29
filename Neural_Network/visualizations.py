# visualizations.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
