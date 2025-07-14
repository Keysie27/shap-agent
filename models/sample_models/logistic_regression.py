from sklearn.linear_model import LogisticRegression

def train(X, y, **kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(X, y)
    return model