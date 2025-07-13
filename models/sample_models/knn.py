import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def train(X, y, n_neighbors=3, **kwargs):
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    model.fit(X, y)
    return model