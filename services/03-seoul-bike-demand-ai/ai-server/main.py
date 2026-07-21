# ==========================================================
# Seoul Bike Demand AI - FastAPI
# Regression API
# ==========================================================

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================================================
# 1. FastAPI 앱 생성
# ==========================================================

app = FastAPI(
    title="Seoul Bike Demand AI",
    description="서울시 공공자전거 대여량 예측 API",
    version="1.0.0"
)


# ==========================================================
# 2. CORS 설정
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# 3. 모델 불러오기
# ==========================================================

model = joblib.load("models/bike_demand_model.pkl")


# ==========================================================
# 4. 입력 데이터 구조 정의
# ==========================================================

class BikeFeatures(BaseModel):
    hour: int
    temperature: float
    humidity: int
    wind_speed: float
    visibility: int
    dew_point: float
    solar_radiation: float
    rainfall: float
    snowfall: float
    season: str
    holiday: str
    functioning_day: str


# ==========================================================
# 5. 기본 라우터
# ==========================================================

@app.get("/")
def root():
    return {
        "message": "Seoul Bike Demand AI API",
        "status": "running"
    }


# ==========================================================
# 6. 예측 API
# ==========================================================

@app.post("/predict")
def predict(features: BikeFeatures):
    data = features.model_dump()

    input_df = pd.DataFrame([data])

    # train_model.py와 동일하게 One-Hot Encoding 적용
    input_df = pd.get_dummies(input_df)

    # 모델이 학습한 컬럼 순서/개수에 맞추기
    input_df = input_df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    return {
        "prediction": round(float(prediction), 2),
        "unit": "rented bikes"
    }