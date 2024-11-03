# optimizers.py
import numpy as np

def cross_entropy(Y_pred, Y):
    """
    Calcula la pérdida de entropía cruzada.

    :param Y_pred: Predicciones de la red.
    :param Y: Etiquetas verdaderas.
    :return: Valor de la pérdida.
    """
    return -np.sum(Y * np.log(Y_pred.reshape(1, len(Y_pred)) + 1e-10))  # Evita log(0)

def adam_optimizer(model, X, y, accuracy, epochs=100, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Optimiza los pesos y sesgos de la red usando el algoritmo Adam.

    :param model: Instancia del modelo de red neuronal.
    :param X: Datos de entrada.
    :param y: Etiquetas verdaderas (debe estar en formato one-hot).
    :param accuracy: Función para medir la precisión.
    :param epochs: Número de épocas para entrenar.
    :param learning_rate: Tasa de aprendizaje.
    :param beta1: Parámetro beta1 para Adam.
    :param beta2: Parámetro beta2 para Adam.
    :param epsilon: Pequeño valor para evitar divisiones por cero.
    :return: Listas de precisión y pérdida.
    """
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    m_weights = [np.zeros_like(layer.weights) for layer in model.layers]
    v_weights = [np.zeros_like(layer.weights) for layer in model.layers]
    m_biases = [np.zeros_like(layer.bias) for layer in model.layers]
    v_biases = [np.zeros_like(layer.bias) for layer in model.layers]

    acc_list = []
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X_train)):
            
            derivatives = model.backpropagation(X_train[i], y_train[i])

            for layer_idx, layer in enumerate(model.layers):
                m_weights[layer_idx] = beta1 * m_weights[layer_idx] + (1 - beta1) * derivatives[layer_idx][1]
                v_weights[layer_idx] = beta2 * v_weights[layer_idx] + (1 - beta2) * (derivatives[layer_idx][1] ** 2)

                m_biases[layer_idx] = beta1 * m_biases[layer_idx] + (1 - beta1) * derivatives[layer_idx][0]
                v_biases[layer_idx] = beta2 * v_biases[layer_idx] + (1 - beta2) * (derivatives[layer_idx][0] ** 2)

                m_weights_corrected = m_weights[layer_idx] / (1 - beta1 ** (epoch + 1))
                v_weights_corrected = v_weights[layer_idx] / (1 - beta2 ** (epoch + 1))
                m_biases_corrected = m_biases[layer_idx] / (1 - beta1 ** (epoch + 1))
                v_biases_corrected = v_biases[layer_idx] / (1 - beta2 ** (epoch + 1))

                layer.weights -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + epsilon)
                layer.bias -= learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + epsilon)

        if epoch % 10 == 0:
            Y_pred = np.array([model.feedforward(x) for x in X_val])
            acc = accuracy(y_val, Y_pred)

            loss = model.compute_loss(y_train[i], model.feedforward(X_train[i]))

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)

    return acc_list, loss_list


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
