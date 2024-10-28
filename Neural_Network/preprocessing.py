#prepocessing.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_datairis(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded

def preprocess_datamnist(X,y):
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))  

    return X, y_encoded