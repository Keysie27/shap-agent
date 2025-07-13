from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, **kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model
