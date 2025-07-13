from sklearn.neighbors import KNeighborsClassifier

def train(X_train, y_train, n_neighbors=3, **kwargs):
    """
    Entrena un modelo KNN con el número de vecinos especificado.

    Args:
        X_train (pd.DataFrame): features
        y_train (pd.Series): etiquetas
        n_neighbors (int): número de vecinos (por defecto 3)
        **kwargs: otros parámetros opcionales de KNeighborsClassifier

    Returns:
        Trained KNeighborsClassifier model
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    model.fit(X_train, y_train)
    return model