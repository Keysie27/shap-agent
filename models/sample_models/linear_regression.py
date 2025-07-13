from sklearn.linear_model import LinearRegression

def train(X_train, y_train, **kwargs):
    model = LinearRegression(**kwargs)
    model.fit(X_train, y_train)
    return model
