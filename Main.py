from data.load_data import load_iris_data
import numpy as np
from Neural_Network.neural_network import NeuralNetwork
from Neural_Network.utils import preprocess_data, train_test_split

X, y = load_iris_data() 
X, y = preprocess_data(X, y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

nn = NeuralNetwork()
nn.train(X_train, y_train, learning_rate=0.2, epochs=750)

y_pred_test = nn.forward(X_test)
y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_true_classes_test = np.argmax(y_test, axis=1)

accuracy_test = np.mean(y_pred_classes_test == y_true_classes_test)
print(f"Precisión en el conjunto de prueba: {accuracy_test * 100:.2f}%")