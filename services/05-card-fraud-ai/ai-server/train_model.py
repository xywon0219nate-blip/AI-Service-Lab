# train_model.py
# ---------------------------------------------------------------
# Card Fraud Detection AI - PaySim 기반 이진 분류 모델 학습
#
# 실행:
#   python train_model.py                 # 전체(또는 샘플) 데이터로 학습
#   python train_model.py --build-sample   # 대용량 원본에서 학습용 샘플 CSV만 생성
# ---------------------------------------------------------------

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import (
    ENGINEERED_FEATURES,
    FraudFeatureEngineer,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RAW_INPUT_COLUMNS,
    TARGET_COLUMN,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------
# 0. 경로 및 상수
# ---------------------------------------------------------------

DATA_DIR = "data"
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "fraud_detection_pipeline.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# data/ 폴더에서 이 순서로 데이터 파일을 탐색한다.
CANDIDATE_DATA_FILES = [
    "paysim.csv",
    "paysim_sample.csv",
    "PS_20174392719_1491204439457_log.csv",
    "PaySim.csv",
    "fraud.csv",
]

REQUIRED_RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# 대용량 CSV를 pandas.read_csv(chunksize=...)로 나눠 읽기 시작하는 파일 크기 기준(300MB)
CHUNK_THRESHOLD_BYTES = 300 * 1024 * 1024
CHUNK_SIZE = 500_000

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD_CANDIDATES = [0.50, 0.40, 0.30, 0.20]


# ---------------------------------------------------------------
# 1. 데이터 파일 탐색 / 로딩
# ---------------------------------------------------------------

def find_data_file():
    for filename in CANDIDATE_DATA_FILES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            return path
    return None


def print_missing_data_guide():
    print("=" * 70)
    print("[오류] PaySim 데이터 파일을 찾을 수 없습니다.")
    print("=" * 70)
    print("다음 경로 중 하나에 PaySim CSV 파일을 저장한 뒤 다시 실행하세요.")
    for filename in CANDIDATE_DATA_FILES:
        print(f"  - {os.path.join(DATA_DIR, filename)}")
    print()
    print("자세한 안내는 data/README.md 를 참고하세요.")
    print("=" * 70)


def _read_csv_columns(path):
    header = pd.read_csv(path, nrows=0)
    return list(header.columns)


def load_paysim_dataframe(path):
    """usecols/dtype을 지정해 메모리를 아끼면서 PaySim CSV를 로딩한다.
    파일이 매우 크면(300MB 이상) chunksize로 나누어 읽는다.
    """
    available_columns = _read_csv_columns(path)
    missing_required = [c for c in REQUIRED_RAW_COLUMNS if c not in available_columns]
    if missing_required:
        print(f"[경고] 예상 컬럼과 다릅니다. 누락된 컬럼: {missing_required}")
        print(f"       실제 컬럼: {available_columns}")
        print("       실제 파일의 컬럼을 기준으로 계속 진행합니다.")

    usecols = [c for c in REQUIRED_RAW_COLUMNS if c in available_columns]

    dtype_map = {
        "step": "int32",
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    dtype = {k: v for k, v in dtype_map.items() if k in usecols}

    file_size = os.path.getsize(path)
    print(f"데이터 파일: {path} ({file_size / (1024 ** 2):.1f} MB)")

    if file_size < CHUNK_THRESHOLD_BYTES:
        df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    else:
        print(f"파일이 커서 chunksize={CHUNK_SIZE:,} 로 나누어 읽습니다.")
        chunks = []
        for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=CHUNK_SIZE):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    return df


def build_training_sample(source_path, output_path, non_fraud_sample_size=200_000, random_state=RANDOM_STATE):
    """대용량 원본에서 학습/강의용 샘플을 만든다.
    - 사기 거래(isFraud == 1)는 전부 보존한다.
    - 정상 거래(isFraud == 0)는 지정한 개수만큼만 무작위 샘플링한다.
    이렇게 만든 샘플은 학습(train) 용도로만 사용하고, 실제 서비스 평가에는
    원본 데이터의 실제 클래스 분포를 사용해야 한다.
    """
    print(f"샘플 생성을 시작합니다: {source_path} -> {output_path}")
    df = load_paysim_dataframe(source_path)

    fraud_rows = df[df["isFraud"] == 1]
    non_fraud_rows = df[df["isFraud"] == 0]

    sample_size = min(non_fraud_sample_size, len(non_fraud_rows))
    non_fraud_sample = non_fraud_rows.sample(n=sample_size, random_state=random_state)

    sample_df = pd.concat([fraud_rows, non_fraud_sample], ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    print(f"샘플 저장 완료: {output_path}")
    print(f"  사기 거래: {len(fraud_rows):,}건 (전부 포함)")
    print(f"  정상 거래: {len(non_fraud_sample):,}건 (샘플링)")
    print(f"  전체: {len(sample_df):,}건")
    return output_path


# ---------------------------------------------------------------
# 2. 데이터 품질 점검 및 정리
# ---------------------------------------------------------------

def report_data_quality(df):
    print("-" * 70)
    print("데이터 품질 점검")
    print("-" * 70)
    print("shape:", df.shape)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("결측치:", dict(missing) if len(missing) else "없음")

    duplicate_count = df.duplicated().sum()
    print("완전 중복 행:", duplicate_count)

    numeric_cols = ["amount", "oldbalanceOrg", "oldbalanceDest"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    inf_count = np.isinf(df[numeric_cols].to_numpy(dtype="float64")).sum() if numeric_cols else 0
    print("무한값(Infinity) 개수:", inf_count)

    negative_amount = (df["amount"] < 0).sum() if "amount" in df.columns else 0
    print("음수 거래 금액:", negative_amount)

    if "isFraud" in df.columns:
        print("타깃(isFraud) 값 종류:", sorted(df["isFraud"].dropna().unique().tolist()))
    print("-" * 70)


def clean_data(df):
    df = df.drop_duplicates()

    numeric_cols = [c for c in ["amount", "oldbalanceOrg", "oldbalanceDest"] if c in df.columns]
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=[c for c in numeric_cols + ["type", "isFraud"] if c in df.columns])
    if "amount" in df.columns:
        df = df[df["amount"] >= 0]
    after = len(df)
    if before != after:
        print(f"정리 과정에서 {before - after:,}건의 행을 제거했습니다 (결측치/음수 금액).")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# 3. Feature / Target 준비 (데이터 누수 방지)
# ---------------------------------------------------------------

def prepare_features(df):
    df = df.copy()

    # step(1 = 1시간)에서 하루 중 시간(0~23)을 계산한다.
    df["Hour"] = ((df["step"].astype(int) - 1) % 24).astype(int)

    X = df[RAW_INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    # 아래 컬럼들은 사전 탐지 시점에 사용할 수 없거나(사후 정보) 데이터 누수를 유발하므로
    # 기본 모델 입력(RAW_INPUT_COLUMNS)에 애초에 포함하지 않는다.
    #   - newbalanceOrig, newbalanceDest : 거래 이후에만 알 수 있는 값
    #   - nameOrig, nameDest             : 고객 식별자
    #   - isFlaggedFraud                 : 기존 규칙 기반 탐지 결과(타깃과 강하게 연관)
    return X, y


# ---------------------------------------------------------------
# 4. 전처리 Pipeline
# ---------------------------------------------------------------

def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_pipeline(model):
    return Pipeline(steps=[
        ("feature_engineering", FraudFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


# ---------------------------------------------------------------
# 5. 평가
# ---------------------------------------------------------------

def evaluate_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def evaluate_model(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


# ---------------------------------------------------------------
# 6. 학습 메인 로직
# ---------------------------------------------------------------

def build_candidate_models(scale_pos_weight):
    # max_samples: 트리마다 학습 데이터 전체가 아닌 일부(30%)만 부트스트랩하여
    # PaySim처럼 수백만 행 규모인 데이터에서도 학습 시간이 지나치게 길어지지 않도록 한다.
    candidates = {
        "Baseline (Random Forest, 불균형 미처리)": RandomForestClassifier(
            n_estimators=200, max_depth=16, max_samples=0.3,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, max_depth=16, max_samples=0.3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost (balanced)"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        # XGBoost가 설치되지 않은 환경을 위한 대체 모델.
        # HistGradientBoostingClassifier는 scikit-learn 내장이라 별도 설치가 필요 없다.
        from sklearn.ensemble import HistGradientBoostingClassifier

        candidates["HistGradientBoosting (XGBoost 대체, balanced)"] = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
        print("[안내] xgboost 가 설치되어 있지 않아 HistGradientBoostingClassifier로 대체합니다.")

    return candidates


def main():
    build_sample_only = "--build-sample" in sys.argv

    os.makedirs(MODEL_DIR, exist_ok=True)

    data_path = find_data_file()
    if data_path is None:
        print_missing_data_guide()
        sys.exit(1)

    if build_sample_only:
        sample_path = os.path.join(DATA_DIR, "paysim_sample.csv")
        build_training_sample(data_path, sample_path)
        return

    # -----------------------------------------------------------
    # 1) 데이터 로딩 및 품질 점검
    # -----------------------------------------------------------
    df = load_paysim_dataframe(data_path)
    report_data_quality(df)
    df = clean_data(df)

    fraud_count = int((df["isFraud"] == 1).sum())
    normal_count = int((df["isFraud"] == 0).sum())
    print(f"정상 거래: {normal_count:,}건 / 사기 거래: {fraud_count:,}건 "
          f"(사기 비율 {fraud_count / len(df) * 100:.4f}%)")

    if fraud_count < 10:
        print("[오류] 사기 거래 표본이 너무 적어 안정적인 학습이 어렵습니다. 데이터를 확인하세요.")
        sys.exit(1)

    # -----------------------------------------------------------
    # 2) Feature / Target 준비 + Train/Test 분리 (Stratified)
    # -----------------------------------------------------------
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    scale_pos_weight = normal_count / max(fraud_count, 1)

    # -----------------------------------------------------------
    # 3) 모델 비교 (불균형 처리 포함)
    # -----------------------------------------------------------
    candidates = build_candidate_models(scale_pos_weight)
    comparison_results = []
    fitted_pipelines = {}
    test_probabilities = {}

    for name, model in candidates.items():
        print(f"\n=== {name} 학습 중 ===")
        start = time.time()
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_proba, threshold=0.5)
        metrics["train_seconds"] = round(elapsed, 2)

        print(f"  Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
              f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
              f"({elapsed:.1f}s)")

        comparison_results.append({"Model": name, **metrics})
        fitted_pipelines[name] = pipeline
        test_probabilities[name] = y_proba

    comparison_df = pd.DataFrame(comparison_results)
    print("\n=== 모델 비교 결과 ===")
    print(comparison_df.to_string(index=False))

    # -----------------------------------------------------------
    # 4) 최종 모델 선정 (Fraud Recall과 F1을 함께 고려해 PR-AUC 기준으로 선정)
    # -----------------------------------------------------------
    balanced_candidates = comparison_df[comparison_df["Model"] != "Baseline (Random Forest, 불균형 미처리)"]
    best_row = balanced_candidates.loc[balanced_candidates["pr_auc"].idxmax()]
    best_model_name = best_row["Model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_proba = test_probabilities[best_model_name]

    print(f"\n최종 모델 후보: {best_model_name} (PR-AUC={best_row['pr_auc']:.4f} 기준)")

    # -----------------------------------------------------------
    # 5) Threshold 비교 및 선정 (F1 최댓값 기준)
    # -----------------------------------------------------------
    print("\n=== Threshold 비교 ===")
    threshold_results = [evaluate_at_threshold(y_test, best_proba, t) for t in THRESHOLD_CANDIDATES]
    threshold_df = pd.DataFrame(threshold_results)
    print(threshold_df.to_string(index=False))

    best_threshold_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    final_threshold = float(best_threshold_row["threshold"])
    print(f"\n선정된 Threshold: {final_threshold} (F1={best_threshold_row['f1']:.4f} 기준 최댓값)")

    final_metrics = evaluate_model(y_test, best_proba, threshold=final_threshold)

    # -----------------------------------------------------------
    # 6) Pipeline + 메타데이터 저장
    # -----------------------------------------------------------
    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"\n모델 저장 완료: {PIPELINE_PATH}")

    metadata = {
        "project": "04-card-fraud-ai",
        "service_name": "Card Fraud Detection AI",
        "dataset": "PaySim",
        "data_file": os.path.basename(data_path),
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "target": TARGET_COLUMN,
        "input_features": RAW_INPUT_COLUMNS,
        "engineered_features": ENGINEERED_FEATURES,
        "threshold": final_threshold,
        "metrics": {
            "accuracy": round(final_metrics["accuracy"], 6),
            "precision": round(final_metrics["precision"], 6),
            "recall": round(final_metrics["recall"], 6),
            "f1": round(final_metrics["f1"], 6),
            "roc_auc": round(final_metrics["roc_auc"], 6),
            "pr_auc": round(final_metrics["pr_auc"], 6),
        },
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_ratio_percent": round(fraud_count / len(df) * 100, 6),
        "random_state": RANDOM_STATE,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"메타데이터 저장 완료: {METADATA_PATH}")

    print("\n=== 학습 완료 ===")
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
