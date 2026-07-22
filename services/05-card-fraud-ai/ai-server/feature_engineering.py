# feature_engineering.py
# ---------------------------------------------------------------
# Card Fraud Detection AI - 공용 Feature Engineering 모듈
#
# 학습(train_model.py)과 예측(main.py)이 완전히 동일한 파생변수 로직을
# 사용해야 하므로, scikit-learn Pipeline에 그대로 끼워 넣을 수 있는
# Transformer 하나로 모든 Feature Engineering을 정의한다.
#
# 이렇게 하면 학습된 Pipeline(.pkl) 안에 Feature Engineering이 함께
# 저장되므로, main.py는 원본 5개 입력만 DataFrame으로 만들어
# pipeline.predict_proba()에 넘기면 된다. (로직 중복/누락 위험 없음)
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------
# 1. 도메인 상수
# ---------------------------------------------------------------

# 사용자가 예측 시 반드시 입력해야 하는 원본 컬럼 (Pipeline의 입력 컬럼 순서)
RAW_INPUT_COLUMNS = ["type", "amount", "oldbalanceOrg", "oldbalanceDest", "Hour"]

# PaySim에서 사용되는 거래 유형
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# 데이터 누수 위험 또는 예측 시점에 알 수 없어 기본 모델 입력에서 제외하는 컬럼
# (Notebook에서 별도 비교 실험 시 참고용으로만 사용)
LEAKAGE_COLUMNS = ["newbalanceOrig", "newbalanceDest", "isFlaggedFraud"]
IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]
TARGET_COLUMN = "isFraud"

# IsLargeTransaction 판단 기준(고액 거래 분위수). 실제 값은 학습 데이터에서 fit() 시 계산한다.
LARGE_TRANSACTION_QUANTILE = 0.95

# 하루 중 시간대 구간
HOUR_BINS = [-1, 5, 11, 17, 23]
HOUR_LABELS = ["Dawn", "Morning", "Afternoon", "Evening"]

# Feature Engineering 이후 최종적으로 모델(ColumnTransformer)에 들어가는 컬럼 목록
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "Hour",
    "AmountLog",
    "AmountToSenderBalance",
    "AmountToReceiverBalance",
    "SenderBalanceAfterEstimated",
    "ReceiverBalanceAfterEstimated",
    "IsSenderBalanceZero",
    "IsLargeTransaction",
    "IsTransfer",
    "IsCashOut",
    "BalanceDifference",
]
CATEGORICAL_FEATURES = ["type", "TimeOfDay"]
ENGINEERED_FEATURES = [
    "TimeOfDay",
    "AmountLog",
    "AmountToSenderBalance",
    "AmountToReceiverBalance",
    "SenderBalanceAfterEstimated",
    "ReceiverBalanceAfterEstimated",
    "IsSenderBalanceZero",
    "IsLargeTransaction",
    "IsTransfer",
    "IsCashOut",
    "BalanceDifference",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """분모가 0이어도 NaN/Infinity가 나오지 않도록 +1을 더해 나누는 안전한 비율 계산."""
    denom_safe = denominator.clip(lower=0).fillna(0) + 1.0
    numer_safe = numerator.clip(lower=0).fillna(0)
    ratio = numer_safe / denom_safe
    return ratio.replace([np.inf, -np.inf], 0).fillna(0)


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """PaySim 원본 5개 입력(type, amount, oldbalanceOrg, oldbalanceDest, Hour)으로부터
    파생변수를 생성하는 scikit-learn 호환 Transformer.

    - fit(): 학습 데이터 기준으로 IsLargeTransaction 임계값(고액 거래 분위수)을 계산해 저장한다.
    - transform(): 학습/예측 어디서 호출되든 완전히 동일한 규칙으로 파생변수를 만든다.
    """

    def __init__(self, large_transaction_quantile: float = LARGE_TRANSACTION_QUANTILE):
        self.large_transaction_quantile = large_transaction_quantile

    def fit(self, X: pd.DataFrame, y=None):
        amount = pd.to_numeric(X["amount"], errors="coerce").clip(lower=0).fillna(0)
        self.amount_threshold_ = float(amount.quantile(self.large_transaction_quantile))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "amount_threshold_"):
            raise RuntimeError("FraudFeatureEngineer는 transform() 이전에 fit()이 필요합니다.")

        df = X.copy()

        # ---- 타입 정리 및 안전한 숫자 변환 -----------------------------
        df["type"] = df["type"].astype(str)
        amount = pd.to_numeric(df["amount"], errors="coerce").clip(lower=0).fillna(0)
        old_sender = pd.to_numeric(df["oldbalanceOrg"], errors="coerce").clip(lower=0).fillna(0)
        old_receiver = pd.to_numeric(df["oldbalanceDest"], errors="coerce").clip(lower=0).fillna(0)
        hour = pd.to_numeric(df["Hour"], errors="coerce").fillna(0).clip(lower=0, upper=23).astype(int)

        df["amount"] = amount
        df["oldbalanceOrg"] = old_sender
        df["oldbalanceDest"] = old_receiver
        df["Hour"] = hour

        # ---- 파생변수 -----------------------------------------------
        df["TimeOfDay"] = pd.cut(
            hour, bins=HOUR_BINS, labels=HOUR_LABELS, include_lowest=True
        ).astype(str)

        df["AmountLog"] = np.log1p(amount)

        df["AmountToSenderBalance"] = _safe_ratio(amount, old_sender)
        df["AmountToReceiverBalance"] = _safe_ratio(amount, old_receiver)

        # newbalanceOrig/newbalanceDest(사후 정보)를 사용하지 않고,
        # 예측 시점에 알 수 있는 값(oldbalance, amount)만으로 "예상" 잔액을 추정한다.
        sender_after = old_sender - amount
        receiver_after = old_receiver + amount
        df["SenderBalanceAfterEstimated"] = sender_after
        df["ReceiverBalanceAfterEstimated"] = receiver_after

        df["IsSenderBalanceZero"] = (sender_after <= 0).astype(int)
        df["IsLargeTransaction"] = (amount > self.amount_threshold_).astype(int)
        df["IsTransfer"] = (df["type"] == "TRANSFER").astype(int)
        df["IsCashOut"] = (df["type"] == "CASH_OUT").astype(int)
        df["BalanceDifference"] = old_sender - old_receiver

        output_columns = ["type"] + NUMERIC_FEATURES + ["TimeOfDay"]
        # 중복 없이 순서 보장
        seen = set()
        ordered_columns = []
        for col in output_columns:
            if col not in seen:
                ordered_columns.append(col)
                seen.add(col)

        return df[ordered_columns]

    def get_feature_names_out(self, input_features=None):
        return np.array(["type"] + NUMERIC_FEATURES + ["TimeOfDay"])


def build_raw_input_frame(
    transaction_type: str,
    amount: float,
    sender_old_balance: float,
    receiver_old_balance: float,
    transaction_hour: int,
) -> pd.DataFrame:
    """FastAPI 요청 필드명을 Pipeline이 기대하는 원본 컬럼명으로 변환한다.

    API 필드명(transaction_type 등)과 모델 학습 컬럼명(type 등)이 다르므로
    이 함수 하나만 거치면 항상 동일한 변환 규칙이 적용된다.
    """
    return pd.DataFrame(
        [
            {
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": sender_old_balance,
                "oldbalanceDest": receiver_old_balance,
                "Hour": transaction_hour,
            }
        ],
        columns=RAW_INPUT_COLUMNS,
    )
