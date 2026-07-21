# ==========================================================
# Seoul Bike Demand AI - Train Model
# RandomForest Regression
# ==========================================================
# ----------------------------------------------------------
#    1. 라이브러리 import
# ----------------------------------------------------------
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------------
#    2. 데이터 불러오기
# ----------------------------------------------------------
df = pd.read_csv(
    'data/seoul_bike_data.csv', 
    encoding='cp949'
)
print(df.head())


# ----------------------------------------------------------
#    3. 컬럼명 변경
# ----------------------------------------------------------
df.columns = [
    "date",
    "bike_count",
    "hour",
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point",
    "solar_radiation",
    "rainfall",
    "snowfall",
    "season",
    "holiday",
    "functioning_day"
]
print(df.columns.tolist())
print(df.head())


# ---------------------------------------------
#    ⭐ 테스트 JSON 생성 : 마지막에 추가
#    9. 시나리오 기반 테스트 JSON 생성
# ---------------------------------------------
import json
import os

os.makedirs("test_data", exist_ok=True)

# FastAPI 입력값으로 사용할 컬럼
input_columns = [
    "hour",
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point",
    "solar_radiation",
    "rainfall",
    "snowfall",
    "season",
    "holiday",
    "functioning_day"
]

test_scenarios = {
    "test_case_01_summer_evening.json": df[
        (df["season"] == "Summer") &
        (df["hour"] == 18) &
        (df["rainfall"] == 0) &
        (df["snowfall"] == 0)
    ],

    "test_case_02_winter_morning.json": df[
        (df["season"] == "Winter") &
        (df["hour"] == 8) &
        (df["rainfall"] == 0)
    ],

    "test_case_03_rainy_day.json": df[
        df["rainfall"] > 0
    ],

    "test_case_04_snowy_day.json": df[
        df["snowfall"] > 0
    ],

    "test_case_05_holiday_afternoon.json": df[
        (df["holiday"] == "Holiday") &
        (df["hour"] == 14)
    ]
}

for filename, scenario_df in test_scenarios.items():

    # 조건에 맞는 데이터가 없을 경우 건너뛰기
    if scenario_df.empty:
        print(f"조건에 맞는 데이터 없음: {filename}")
        continue

    sample = scenario_df.sample(
        1,
        random_state=42
    )[input_columns]

    sample_json = sample.iloc[0].to_dict()

    with open(f"test_data/{filename}", "w", encoding="utf-8") as f:
        json.dump(
            sample_json,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"생성 완료: test_data/{filename}")

    # 🔥🔥 for문안에서 작업 JSON 입력 데이터
    sample_json = sample.iloc[0].to_dict()

    # ---------------------------------------------
    # 입력(JSON) 저장
    # ---------------------------------------------
    with open(f"test_data/{filename}", "w", encoding="utf-8") as f:
        json.dump(
            sample_json,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------------
    # 정답(answer) 저장
    # ---------------------------------------------
    answer = {
        "bike_count": int(scenario_df.sample(1, random_state=42)["bike_count"].iloc[0])
    }

    answer_filename = filename.replace("test_case", "answer")

    with open(f"test_data/{answer_filename}", "w", encoding="utf-8") as f:
        json.dump(
            answer,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"생성 완료 : {filename}")
    print(f"생성 완료 : {answer_filename}")



# ----------------------------------------------------------
#    4. Feature / Target 준비
# ----------------------------------------------------------
# 날짜는 문자열이므로 이번 모델에서는 제외
df = df.drop(columns=["date"])

# 문자열 컬럼을 숫자형으로 변환
df = pd.get_dummies(df)

X = df.drop(columns=["bike_count"])
y = df["bike_count"]

# ----------------------------------------------------------
#    5. Train / Test 데이터 분리
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------------
#    6. RandomForest 회귀 모델 생성 및 학습
# ----------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ----------------------------------------------------------
#    7. 예측 및 간단 평가
# ----------------------------------------------------------
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# ----------------------------------------------------------
#    8. 모델 저장
# ----------------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/bike_demand_model.pkl"
)

print("모델 저장 완료: models/bike_demand_model.pkl")


