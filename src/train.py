from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def save_model(model):
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)