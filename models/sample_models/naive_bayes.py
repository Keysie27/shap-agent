from sklearn.naive_bayes import GaussianNB

def train(X_train, y_train, **kwargs):
    model = GaussianNB(**kwargs)
    model.fit(X_train, y_train)
    return model
