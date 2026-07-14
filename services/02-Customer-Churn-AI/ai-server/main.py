# main.py
# ---------------------------------------------------------------
#  Customer Churn AI - FastAPI 서버
#  고객 정보를 입력받아 '이탈(Churn)' 여부를 예측하는 API 입니다.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
#   1. 라이브러리 import
# ---------------------------------------------------------------
from fastapi import FastAPI          # 웹 API 서버를 만드는 라이브러리
from pydantic import BaseModel, Field  # 입력 데이터의 형태(타입)를 검증하는 라이브러리
import joblib                        # 학습된 AI 모델(.pkl)을 불러오는 라이브러리
import pandas as pd                  # 표(DataFrame) 형태로 데이터를 다루는 라이브러리


# ---------------------------------------------------------------
#   2. FastAPI 앱 생성
#   - title, description, version 은 Swagger 문서 상단에 표시됩니다.
# ---------------------------------------------------------------
app = FastAPI(
    title="Customer Churn AI API",
    description="고객 정보를 입력하면 이탈(Churn) 여부를 예측하는 AI 서버입니다.",
    version="1.0.0",
)


# ---------------------------------------------------------------
#   3. 학습된 모델 불러오기
#   - train_model.py 에서 저장한 모델 파일을 그대로 로드합니다.
#   - 서버가 켜질 때 딱 한 번만 로드해두고, 요청마다 재사용합니다.
# ---------------------------------------------------------------
model = joblib.load("models/customer_churn_model.pkl")


# ---------------------------------------------------------------
#   4. 입력 데이터 형태 정의 (Pydantic BaseModel)
#   - 학습에 사용한 Feature 와 '동일한' 컬럼을 입력받습니다. (Customer ID 는 제외)
#   - 원본 컬럼명에는 띄어쓰기가 있어서 파이썬 변수로 쓸 수 없습니다.
#     그래서 변수는 snake_case 로 만들고, Field(alias="원본 컬럼명") 으로 연결합니다.
#   - 즉, 실제 요청(JSON)의 key 는 alias 에 적힌 원본 컬럼명을 사용합니다.
# ---------------------------------------------------------------
class CustomerFeatures(BaseModel):
    gender: str = Field(alias="Gender")
    age: int = Field(alias="Age")
    under_30: str = Field(alias="Under 30")
    senior_citizen: str = Field(alias="Senior Citizen")
    married: str = Field(alias="Married")
    dependents: str = Field(alias="Dependents")
    number_of_dependents: int = Field(alias="Number of Dependents")
    country: str = Field(alias="Country")
    state: str = Field(alias="State")
    city: str = Field(alias="City")
    zip_code: int = Field(alias="Zip Code")
    latitude: float = Field(alias="Latitude")
    longitude: float = Field(alias="Longitude")
    population: int = Field(alias="Population")
    quarter: str = Field(alias="Quarter")
    referred_a_friend: str = Field(alias="Referred a Friend")
    number_of_referrals: int = Field(alias="Number of Referrals")
    tenure_in_months: int = Field(alias="Tenure in Months")
    offer: str = Field(alias="Offer")
    phone_service: str = Field(alias="Phone Service")
    avg_monthly_long_distance_charges: float = Field(alias="Avg Monthly Long Distance Charges")
    multiple_lines: str = Field(alias="Multiple Lines")
    internet_service: str = Field(alias="Internet Service")
    internet_type: str = Field(alias="Internet Type")
    avg_monthly_gb_download: int = Field(alias="Avg Monthly GB Download")
    online_security: str = Field(alias="Online Security")
    online_backup: str = Field(alias="Online Backup")
    device_protection_plan: str = Field(alias="Device Protection Plan")
    premium_tech_support: str = Field(alias="Premium Tech Support")
    streaming_tv: str = Field(alias="Streaming TV")
    streaming_movies: str = Field(alias="Streaming Movies")
    streaming_music: str = Field(alias="Streaming Music")
    unlimited_data: str = Field(alias="Unlimited Data")
    contract: str = Field(alias="Contract")
    paperless_billing: str = Field(alias="Paperless Billing")
    payment_method: str = Field(alias="Payment Method")
    monthly_charge: float = Field(alias="Monthly Charge")
    total_charges: float = Field(alias="Total Charges")
    total_refunds: float = Field(alias="Total Refunds")
    total_extra_data_charges: int = Field(alias="Total Extra Data Charges")
    total_long_distance_charges: float = Field(alias="Total Long Distance Charges")
    total_revenue: float = Field(alias="Total Revenue")
    satisfaction_score: int = Field(alias="Satisfaction Score")
    cltv: int = Field(alias="CLTV")

    # model_config : Pydantic 의 설정값
    #  - populate_by_name : 변수명(snake_case)으로도 값을 넣을 수 있게 허용
    #  - json_schema_extra : Swagger 에 미리 채워질 '예시(example)' 데이터
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Gender": "Female",
                "Age": 74,
                "Under 30": "No",
                "Senior Citizen": "Yes",
                "Married": "Yes",
                "Dependents": "Yes",
                "Number of Dependents": 1,
                "Country": "United States",
                "State": "California",
                "City": "Los Angeles",
                "Zip Code": 90063,
                "Latitude": 34.044271,
                "Longitude": -118.185237,
                "Population": 55668,
                "Quarter": "Q3",
                "Referred a Friend": "Yes",
                "Number of Referrals": 1,
                "Tenure in Months": 8,
                "Offer": "Offer E",
                "Phone Service": "Yes",
                "Avg Monthly Long Distance Charges": 48.85,
                "Multiple Lines": "Yes",
                "Internet Service": "Yes",
                "Internet Type": "Fiber Optic",
                "Avg Monthly GB Download": 17,
                "Online Security": "No",
                "Online Backup": "Yes",
                "Device Protection Plan": "No",
                "Premium Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Streaming Music": "No",
                "Unlimited Data": "Yes",
                "Contract": "Month-to-Month",
                "Paperless Billing": "Yes",
                "Payment Method": "Credit Card",
                "Monthly Charge": 80.65,
                "Total Charges": 633.3,
                "Total Refunds": 0.0,
                "Total Extra Data Charges": 0,
                "Total Long Distance Charges": 390.8,
                "Total Revenue": 1024.1,
                "Satisfaction Score": 3,
                "CLTV": 5302,
            }
        },
    }


# ---------------------------------------------------------------
#   5. GET / : 서버가 살아있는지 확인하는 기본 경로
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Customer Churn AI API is running",
        "status": "success",
    }


# ---------------------------------------------------------------
#   6. GET /health : 서버 상태 점검용(헬스 체크) 경로
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "OK",
    }


# ---------------------------------------------------------------
#   7. POST /predict : 고객 정보를 받아 이탈 여부를 예측
# ---------------------------------------------------------------
@app.post("/predict")
def predict(features: CustomerFeatures):
    # (1) 입력값(JSON)을 원본 컬럼명(alias) 기준의 딕셔너리로 변환합니다.
    #     by_alias=True 를 주면 key 가 "Number of Dependents" 처럼 원본 컬럼명이 됩니다.
    data = features.model_dump(by_alias=True)

    # (2) 한 줄짜리 표(DataFrame)로 만듭니다. (학습 때와 같은 컬럼 구조)
    input_df = pd.DataFrame([data])

    # (3) 문자형 컬럼을 숫자형(0/1)으로 변환합니다. (학습 때 사용한 pd.get_dummies 와 동일)
    input_df = pd.get_dummies(input_df)

    # (4) 모델이 학습한 컬럼 순서/개수에 정확히 맞춥니다.
    #     - 입력에 없는 컬럼은 0 으로 채우고(fill_value=0),
    #     - 학습에 없던 컬럼은 자동으로 버려서 형태를 일치시킵니다.
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # (5) 예측을 수행합니다. 결과는 0(잔류) 또는 1(이탈) 입니다.
    prediction = int(model.predict(input_df)[0])

    # (6) 숫자 예측값을 사람이 읽기 쉬운 문자로 변환합니다.
    result = "Churn" if prediction == 1 else "Stay"

    return {
        "prediction": prediction,
        "result": result,
    }
