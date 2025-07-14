from sklearn.svm import SVC

def train(X, y, C=1.0, kernel='rbf', **kwargs):
    model = SVC(C=C, kernel=kernel, probability=True, **kwargs)
    model.fit(X, y)
    return model