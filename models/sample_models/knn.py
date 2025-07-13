from sklearn.neighbors import KNeighborsClassifier

def train(X, y, n_neighbors, **kwargs):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    model.fit(X, y)
    return model