from sklearn.tree import DecisionTreeClassifier

def train(X_train, y_train, max_depth=None, **kwargs):
    model = DecisionTreeClassifier(max_depth=max_depth, **kwargs)
    model.fit(X_train, y_train)
    return model
