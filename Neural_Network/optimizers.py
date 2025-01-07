# optimizers.py
import numpy as np

def cross_entropy(Y_pred, Y):
    """
    Calcula la pérdida de entropía cruzada.

    :param Y_pred: Predicciones de la red.
    :param Y: Etiquetas verdaderas.
    :return: Valor de la pérdida.
    """
    return -np.sum(Y * np.log(Y_pred.reshape(1, len(Y_pred)) + 1e-10))  

def gradient_descent(model, X, y, accuracy, epochs=100, learning_rate=0.01):
    """
    Optimiza los pesos y sesgos de la red usando el algoritmo de descenso de gradiente.

    :param model: Instancia del modelo de red neuronal.
    :param X: Datos de entrada.
    :param y: Etiquetas verdaderas.
    :param accuracy: Función para medir la precisión.
    :param epochs: Número de épocas para entrenar.
    :param learning_rate: Tasa de aprendizaje.
    :return: Listas de precisión y pérdida.
    """
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    acc_list = []
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X_train)):
            derivatives = model.backpropagation(X_train[i], y_train[i])
            for layer, derivative in zip(model.layers[::-1], derivatives):
                layer.weights -= learning_rate * derivative[1]
                layer.bias -= learning_rate * derivative[0]

        if epoch % 10 == 0:
            Y_pred = np.array([model.feedforward(x) for x in X_val])
            acc = accuracy(y_val, Y_pred)

            loss = cross_entropy(model.feedforward(X_train[i]), y_train[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)

    return acc_list, loss_list

def momentum_optimizer(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    """
    Optimiza los pesos y sesgos de la red usando el algoritmo de momentum.

    :param model: Instancia del modelo de red neuronal.
    :param X: Datos de entrada.
    :param y: Etiquetas verdaderas.
    :param measure_function: Función para medir la precisión.
    :param epochs: Número de épocas para entrenar.
    :param learning_rate: Tasa de aprendizaje.
    :return: Listas de precisión y pérdida.    
    """
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    velocities = [
        {
            'weights': np.zeros(layer.weights.shape),
            'biases': np.zeros(layer.bias.shape)
        }
        for layer in model.layers
    ]

    acc_list = []
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X_train)):
            derivates = model.backpropagation(X_train[i], y_train[i]) 

            pos = len(model.layers)-1
            for layer, derivate in zip(model.layers[::-1], derivates):
                velocities[pos]['weights'] = 0.9*velocities[pos]['weights'] + learning_rate*derivate[1]
                velocities[pos]['biases'] = 0.9*velocities[pos]['biases'] + learning_rate*derivate[0]

                layer.weights -= velocities[pos]['weights']
                layer.bias -= velocities[pos]['biases']

                pos -= 1

        if epoch % 10 == 0:
            Y_pred = [model.feedforward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)

            loss = cross_entropy(model.feedforward(X[i]), y[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)

    return acc_list, loss_list