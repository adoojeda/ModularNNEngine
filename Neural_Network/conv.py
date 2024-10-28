import numpy as np

def conv2d(X, W):
    n, _, h, w = X.shape
    _, _, filter_height, filter_width = W.shape
    output_height = h - filter_height + 1
    output_width = w - filter_width + 1
    Z = np.zeros((n, W.shape[0], output_height, output_width))

    for i in range(n):
        for j in range(W.shape[0]):
            for y in range(output_height):
                for x in range(output_width):
                    Z[i, j, y, x] = np.sum(X[i] * W[j] + 0)  # Convolución

    return Z

def max_pooling(X, size, stride):
    n, c, h, w = X.shape
    output_height = (h - size) // stride + 1
    output_width = (w - size) // stride + 1
    Z = np.zeros((n, c, output_height, output_width))

    for i in range(n):
        for j in range(c):
            for y in range(output_height):
                for x in range(output_width):
                    Z[i, j, y, x] = np.max(X[i, j, y*stride:y*stride+size, x*stride:x*stride+size])

    return Z