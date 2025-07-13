from sklearn.tree import DecisionTreeClassifier

def train(X, y, max_depth=None, **kwargs):
    model = DecisionTreeClassifier(max_depth=max_depth, **kwargs)
    model.fit(X, y)
    return model