# utils.py
import numpy as np

def train_test_split(X, y, test_size=0.2, random_seed=42):
        
        np.random.seed(random_seed)
        
        indices = np.random.permutation(X.shape[0])  
        test_size = int(X.shape[0] * test_size)  
        
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
