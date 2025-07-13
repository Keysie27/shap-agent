from sklearn.naive_bayes import GaussianNB

def train(X, y, **kwargs):
    model = GaussianNB(**kwargs)
    model.fit(X, y)
    return model