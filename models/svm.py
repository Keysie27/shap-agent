from sklearn.svm import SVC

def train(X_train, y_train, kernel, C, **kwargs):
    model = SVC(kernel=kernel, C=C, probability=True, **kwargs)
    model.fit(X_train, y_train)
    return model
