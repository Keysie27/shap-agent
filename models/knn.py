from sklearn.neighbors import KNeighborsClassifier

def train(X_train, y_train, n_neighbors, **kwargs):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    model.fit(X_train, y_train)
    return model
