from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open("outputs/models/best_model.pkl","rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
