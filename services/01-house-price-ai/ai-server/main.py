from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="AI Service Lab API",
    description="Master FastAPI template for AI services",
    version="1.0.0"
)

model = joblib.load("models/house_price_model.pkl")


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def root():
    return {
        "message": "AI Service Lab API is running",
        "status": "success"
    }

@app.get("/health")
def health():
    return {
        "status": "OK"
    }

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.model_dump()])
    predicted_price = model.predict(input_df)[0]

    return {
        "predicted_price": round(float(predicted_price), 3)
    }
