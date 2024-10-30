#prepocessing.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(X, y):
    """"
    Preprocesa los datos de entrada y las etiquetas.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded
