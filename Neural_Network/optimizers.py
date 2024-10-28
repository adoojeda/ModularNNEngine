import numpy as np

def cross_entropy(Y_pred, Y):
    return -np.sum(Y*np.log(Y_pred.reshape(1, len(Y_pred))))

def adam_optimizer(model, X, y, measure_function, epochs=100, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    X_train, X_val = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    y_train, y_val = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

    m_weights = [np.zeros_like(layer.weights) for layer in model.layers]
    v_weights = [np.zeros_like(layer.weights) for layer in model.layers]
    m_biases = [np.zeros_like(layer.bias) for layer in model.layers]
    v_biases = [np.zeros_like(layer.bias) for layer in model.layers]

    acc_list = []
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X_train)):
            derivates = model.backpropagation(X_train[i], y_train[i])
            for layer_idx, layer in enumerate(model.layers):
                m_weights[layer_idx] = beta1 * m_weights[layer_idx] + (1 - beta1) * derivates[layer_idx][1]
                v_weights[layer_idx] = beta2 * v_weights[layer_idx] + (1 - beta2) * (derivates[layer_idx][1] ** 2)

                m_biases[layer_idx] = beta1 * m_biases[layer_idx] + (1 - beta1) * derivates[layer_idx][0]
                v_biases[layer_idx] = beta2 * v_biases[layer_idx] + (1 - beta2) * (derivates[layer_idx][0] ** 2)

                m_weights_corrected = m_weights[layer_idx] / (1 - beta1 ** (epoch + 1))
                v_weights_corrected = v_weights[layer_idx] / (1 - beta2 ** (epoch + 1))
                m_biases_corrected = m_biases[layer_idx] / (1 - beta1 ** (epoch + 1))
                v_biases_corrected = v_biases[layer_idx] / (1 - beta2 ** (epoch + 1))

                layer.weights -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + epsilon)
                layer.bias -= learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + epsilon)

        if epoch % 10 == 0:
            Y_pred = [model.feedforward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)

            loss = cross_entropy(model.feedforward(X_train[i]), y_train[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)

    return acc_list, loss_list

def gradient_descent(model, X, y, measure_function, epochs=100, learning_rate=0.01):
    X, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    acc_list = []
    loss_list = []

    for epoch in range(epochs):
        for i in range(len(X)):
            derivates = model.backpropagation(X[i], y[i])                
            for layer, derivate in zip(model.layers[::-1], derivates):
                layer.weights -= learning_rate*derivate[1]
                layer.bias -= learning_rate*derivate[0]

        if epoch % 10 == 0:
            Y_pred = [model.feedfoward(x) for x in X_val]
            acc = measure_function(y_val, Y_pred)

            loss = cross_entropy(model.feedfoward(X[i]), y[i])

            print(f'epoch {epoch:3} - Loss {loss:.5f}, Accuracy {acc:.5f}')

            acc_list.append(acc)
            loss_list.append(loss)

    return acc_list, loss_list


