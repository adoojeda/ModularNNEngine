from Neural_Network.layer import Layer
from data.load_data import load_iris_data
from Neural_Network.neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from Neural_Network.preprocessing import preprocess_data
from Neural_Network.activations import softmax, sigmoid, softmax_derivate, sigmoid_derivate
from Neural_Network import optimizers

# Cargar y preprocesar los datos
X, y = load_iris_data() 
X, y = preprocess_data(X, y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nn = NeuralNetwork()
nn.set(Layer(4, 5, activation=sigmoid, 
                            activation_derivate=sigmoid_derivate))
nn.set(Layer(5, 3, activation=softmax,
                            activation_derivate=softmax_derivate))


print("accuracy before training: ", NeuralNetwork.accuracy(y_test, [nn.feedfoward(x) for x in X_test]))
(acc1, loss1) = optimizers.gradient_descent(nn, X_train, y_train, NeuralNetwork.accuracy, epochs=300, learning_rate=0.01)