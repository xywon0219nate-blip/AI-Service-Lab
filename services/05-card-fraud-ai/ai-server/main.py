# ==========================================================
# Card Fraud Detection AI - FastAPI
# PaySim 기반 금융거래 이상 탐지(사전 탐지) API
# ==========================================================

import json
import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from feature_engineering import TRANSACTION_TYPES, build_raw_input_frame
from train_model import CANDIDATE_DATA_FILES, find_data_file


# ==========================================================
# 1. 경로 / 상수
# ==========================================================

MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "fraud_detection_pipeline.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

DEFAULT_THRESHOLD = 0.5

# 위험 수준(Low/Medium/High) 기준 상수.
# 최종 Threshold를 기준으로 삼아, Threshold의 RISK_LEVEL_LOW_RATIO(50%) 미만이면 Low,
# 그 이상~Threshold 미만이면 Medium, Threshold 이상이면 High로 구분한다.
# (Threshold 자체가 "Suspicious로 분류하는 지점"이므로 High == Suspicious 판정 구간)
RISK_LEVEL_LOW_RATIO = 0.5

# 위험 요인 설명 규칙에서 사용하는 상수
NIGHT_HOURS = {0, 1, 2, 3, 4}
HIGH_BALANCE_USAGE_RATIO = 0.9
LOW_RECEIVER_BALANCE = 100.0

NORMAL_MESSAGE = "정상 가능성이 높은 거래로 분석되었습니다."
SUSPICIOUS_MESSAGE = "추가 확인이 필요한 거래로 탐지되었습니다."


# ==========================================================
# 2. FastAPI 앱 생성 및 CORS 설정
# ==========================================================

app = FastAPI(
    title="Card Fraud Detection AI",
    description="AI 기반 금융거래 이상 탐지 서비스 (PaySim 기반, 교육/포트폴리오 목적)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize_validation_errors(errors):
    """Pydantic 오류의 ctx에 담긴 예외 객체(JSON으로 직렬화 불가능)를 문자열로 바꾼다."""
    sanitized = []
    for error in errors:
        error = dict(error)
        ctx = error.get("ctx")
        if isinstance(ctx, dict):
            error["ctx"] = {key: str(value) for key, value in ctx.items()}
        sanitized.append(error)
    return sanitized


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {
                "error": "입력값을 확인해주세요.",
                "details": _sanitize_validation_errors(exc.errors()),
            }
        ),
    )


# ==========================================================
# 3. 모델 / 메타데이터 로딩
#    - 모델 파일이 없어도 서버 자체는 켜지도록 하고,
#      /predict 등에서 503과 함께 안내 메시지를 반환한다.
#    - "잘못된 상태로 서버를 실행하지 않는" 판단은 start.sh(컨테이너 진입점)에서
#      먼저 수행하며, main.py는 그 상황에서도 원인을 조회할 수 있도록 방어적으로 동작한다.
# ==========================================================

def load_model_and_metadata():
    if not os.path.exists(PIPELINE_PATH):
        print(f"[안내] 모델 파일이 없습니다: {PIPELINE_PATH}")
        print("       data/ 폴더에 PaySim 데이터를 추가한 뒤 train_model.py를 실행하세요.")
        return None, None

    try:
        pipeline = joblib.load(PIPELINE_PATH)
    except Exception as error:  # noqa: BLE001 - 모델 로딩 실패는 서버가 죽지 않고 안내만 한다.
        print(f"[경고] 모델 로딩 중 오류가 발생했습니다: {error}")
        return None, None

    metadata = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, metadata


model, model_metadata = load_model_and_metadata()


def get_threshold():
    if model_metadata and "threshold" in model_metadata:
        return float(model_metadata["threshold"])
    return DEFAULT_THRESHOLD


# ==========================================================
# 4. 샘플 거래 준비 (서버 시작 시 1회만 탐색)
#    - 실제 데이터가 있으면 그 안에서 정상/의심 거래를 하나씩 찾아 사용한다.
#    - 없으면 이해를 돕기 위한 데모 값을 사용하고, source 필드로 구분해준다.
# ==========================================================

DEMO_SAMPLES = {
    "normal": {
        "transaction_type": "PAYMENT",
        "amount": 9839.64,
        "sender_old_balance": 170136.0,
        "receiver_old_balance": 0.0,
        "transaction_hour": 10,
        "source": "demo",
    },
    "suspicious": {
        "transaction_type": "TRANSFER",
        "amount": 181000.0,
        "sender_old_balance": 181000.0,
        "receiver_old_balance": 0.0,
        "transaction_hour": 2,
        "source": "demo",
    },
}


def discover_sample_transactions():
    """data/ 폴더에 실제 PaySim 데이터가 있으면 정상 1건, 의심 1건을 찾아 반환한다.
    데이터가 없거나 탐색에 실패하면 None을 반환해 데모 값으로 대체한다.
    """
    data_path = find_data_file()
    if data_path is None:
        return None

    try:
        import pandas as pd

        found = {}
        for chunk in pd.read_csv(
            data_path,
            usecols=["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"],
            chunksize=200_000,
        ):
            if "normal" not in found:
                normal_rows = chunk[chunk["isFraud"] == 0]
                if len(normal_rows):
                    found["normal"] = normal_rows.iloc[0]
            if "suspicious" not in found:
                fraud_rows = chunk[chunk["isFraud"] == 1]
                if len(fraud_rows):
                    found["suspicious"] = fraud_rows.iloc[0]
            if len(found) == 2:
                break

        if not found:
            return None

        samples = {}
        for key, row in found.items():
            samples[key] = {
                "transaction_type": str(row["type"]),
                "amount": float(row["amount"]),
                "sender_old_balance": float(row["oldbalanceOrg"]),
                "receiver_old_balance": float(row["oldbalanceDest"]),
                "transaction_hour": int((int(row["step"]) - 1) % 24),
                "source": "real_data",
            }
        return samples
    except Exception as error:  # noqa: BLE001
        print(f"[안내] 샘플 거래 탐색 중 오류가 발생해 데모 값을 사용합니다: {error}")
        return None


_discovered_samples = discover_sample_transactions()
SAMPLE_TRANSACTIONS = {**DEMO_SAMPLES, **(_discovered_samples or {})}


# ==========================================================
# 5. 요청 스키마
# ==========================================================

class TransactionInput(BaseModel):
    transaction_type: str = Field(..., description="거래 유형 (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER)")
    amount: float = Field(..., ge=0, description="거래 금액")
    sender_old_balance: float = Field(..., ge=0, description="송금자 거래 전 잔액")
    receiver_old_balance: float = Field(..., ge=0, description="수취인 거래 전 잔액")
    transaction_hour: int = Field(..., ge=0, le=23, description="거래 발생 시간(0~23시)")

    @field_validator("transaction_type")
    @classmethod
    def validate_transaction_type(cls, value):
        if value not in TRANSACTION_TYPES:
            raise ValueError(
                f"transaction_type은 다음 중 하나여야 합니다: {', '.join(TRANSACTION_TYPES)}"
            )
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_type": "TRANSFER",
                "amount": 181000.0,
                "sender_old_balance": 181000.0,
                "receiver_old_balance": 0.0,
                "transaction_hour": 2,
            }
        }
    }


# ==========================================================
# 6. 위험 수준 / 위험 요인 계산
# ==========================================================

def compute_risk_level(fraud_probability, threshold):
    if fraud_probability < threshold * RISK_LEVEL_LOW_RATIO:
        return "Low"
    if fraud_probability < threshold:
        return "Medium"
    return "High"


def compute_risk_factors(payload: TransactionInput, pipeline):
    """모델 내부 값을 그대로 노출하지 않고, 입력값 기반의 이해하기 쉬운 참고 설명을 만든다.
    실제 모델의 정확한 인과관계가 아니라 참고용 설명임에 유의한다.
    """
    factors = []

    if payload.amount > payload.sender_old_balance:
        factors.append("거래 금액이 송금자 잔액보다 큽니다.")
    elif payload.sender_old_balance > 0 and (
        payload.amount / payload.sender_old_balance
    ) >= HIGH_BALANCE_USAGE_RATIO:
        factors.append("거래 금액이 송금자 잔액의 대부분을 차지합니다.")

    if payload.transaction_type == "TRANSFER":
        factors.append("거래 유형이 TRANSFER입니다.")
    elif payload.transaction_type == "CASH_OUT":
        factors.append("거래 유형이 CASH_OUT입니다.")

    if payload.receiver_old_balance < LOW_RECEIVER_BALANCE:
        factors.append("수취인의 기존 잔액이 매우 낮습니다.")

    try:
        amount_threshold = pipeline.named_steps["feature_engineering"].amount_threshold_
        if payload.amount > amount_threshold:
            factors.append("거래 금액이 학습 데이터의 고액 거래 기준보다 큽니다.")
    except (AttributeError, KeyError):
        pass

    if payload.transaction_hour in NIGHT_HOURS:
        factors.append("심야 시간대 거래입니다.")

    return factors


# ==========================================================
# 7. Health Check
# ==========================================================

@app.get("/")
def root():
    return {
        "service": "Card Fraud Detection AI",
        "dataset": "PaySim",
        "status": "running",
        "model_loaded": model is not None,
    }


# ==========================================================
# 8. 모델 정보
# ==========================================================

@app.get("/model-info")
def model_info():
    if model_metadata is None:
        return {
            "model_loaded": model is not None,
            "message": (
                "학습된 모델 메타데이터가 없습니다. data/ 폴더에 PaySim 데이터를 추가한 뒤 "
                "train_model.py를 실행하세요. 필요한 데이터 경로는 " + ", ".join(CANDIDATE_DATA_FILES)
            ),
        }

    return {
        "model_loaded": model is not None,
        "model_name": model_metadata.get("model_name"),
        "model_version": model_metadata.get("model_version"),
        "threshold": model_metadata.get("threshold"),
        "input_features": model_metadata.get("input_features"),
        "engineered_features": model_metadata.get("engineered_features"),
        "metrics": model_metadata.get("metrics"),
    }


# ==========================================================
# 9. 샘플 거래
# ==========================================================

@app.get("/samples")
def get_samples():
    return SAMPLE_TRANSACTIONS


# ==========================================================
# 10. 예측 API
# ==========================================================

@app.post("/predict")
def predict(payload: TransactionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "모델이 아직 준비되지 않았습니다. data/ 폴더에 PaySim 데이터를 추가한 뒤 "
                "train_model.py를 실행하거나 서버를 다시 시작하세요."
            ),
        )

    input_df = build_raw_input_frame(
        transaction_type=payload.transaction_type,
        amount=payload.amount,
        sender_old_balance=payload.sender_old_balance,
        receiver_old_balance=payload.receiver_old_balance,
        transaction_hour=payload.transaction_hour,
    )

    fraud_probability = float(model.predict_proba(input_df)[0, 1])
    normal_probability = 1.0 - fraud_probability

    threshold = get_threshold()
    prediction = int(fraud_probability >= threshold)
    label = "Suspicious" if prediction == 1 else "Normal"
    risk_level = compute_risk_level(fraud_probability, threshold)
    risk_factors = compute_risk_factors(payload, model)
    message = SUSPICIOUS_MESSAGE if prediction == 1 else NORMAL_MESSAGE

    return {
        "prediction": prediction,
        "label": label,
        "fraud_probability": round(fraud_probability, 4),
        "normal_probability": round(normal_probability, 4),
        "threshold": threshold,
        "risk_level": risk_level,
        "message": message,
        "risk_factors": risk_factors,
        "disclaimer": (
            "본 결과는 AI 모델 기반 참고 정보이며 실제 금융기관의 최종 거래 판단을 대신하지 않습니다."
        ),
    }
