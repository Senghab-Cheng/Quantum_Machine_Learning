# phase 5 : Classical Baselline

from sklearn.linear_model import LogisticRegression

def build_model():
    return LogisticRegression(max_iter=1000, random_state=42)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)