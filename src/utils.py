import pickle

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
