from sklearn.linear_model import LinearRegression

def train(X, y, **kwargs):
    model = LinearRegression(**kwargs)
    model.fit(X, y)
    return model