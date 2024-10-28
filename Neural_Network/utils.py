# utils.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def preprocess_datairis(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded

def preprocess_datamnist(X, y):
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    y_encoded = np.zeros((y.size, y.max() + 1))
    y_encoded[np.arange(y.size), y.flatten()] = 1
    return X_scaled, y_encoded

def train_test_split(X, y, test_size=0.2, random_seed=42):
        
        np.random.seed(random_seed)
        
        indices = np.random.permutation(X.shape[0])  
        test_size = int(X.shape[0] * test_size)  
        
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
