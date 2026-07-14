# train_model.py

# ---------------------------------------------
#    1. 라이브러리 import
# ---------------------------------------------
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 모델 성능 평가를 위한 라이브러리
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


import joblib

# ---------------------------------------------
#    2. 데이터 불러오기
# ---------------------------------------------
df = pd.read_csv("data/ibm_telco.csv")

print(df.head())
print(df.info())
print(df.columns)
print(df["Churn Label"].value_counts())     # Target(y) 확인


# ---------------------------------------------
#    3. 데이터 전처리
# ---------------------------------------------
# Target(Churn Label) 문자값('Yes','No')을 숫자로 변환
df['Churn Label'] = df['Churn Label'].map({
    'Yes': 1,
    'No': 0 
})
print(df["Churn Label"].value_counts())     # Target(y) 확인

drop_columns = [
    "Customer ID",
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason"
]

df = df.drop(columns=drop_columns)

# ---------------------------------------------
#    4. X, y 분리
#    Feature(X)-입력값, Target(y)-출력값,정답
# ---------------------------------------------
X = df.drop(columns=['Churn Label'])
y = df['Churn Label']

# 문자형 컬럼을 숫자형 컬럼으로 변환
X = pd.get_dummies(X)

# ---------------------------------------------
#    5. Train/Test
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


# ---------------------------------------------
#    6. 모델 생성
# ---------------------------------------------
model = RandomForestClassifier(
    random_state=42
)

# ---------------------------------------------
#    7. 학습
# ---------------------------------------------
# 학습시 문자(String)를 그대로 학습할 수 없다.
model.fit(X_train, y_train)
# print(X.dtypes)

# ---------------------------------------------
#    8. 예측
# ---------------------------------------------
pred = model.predict(X_test)

print(pred[:10])

# ---------------------------------------------
#    9. 모델 저장
# ---------------------------------------------
joblib.dump(
    model,
    "models/customer_churn_model.pkl"
)

# ---------------------------------------------
#    10. 성능 평가
# ---------------------------------------------
pred = model.predict(X_test)

# ---------------------------------------------
#    11. 정확도(Accuracy)
#   비유) 화살이 전반적으로 중심(정답)에 얼마나 가까이 모였는가?
#   전체 고객 중 AI가 맞춘 비율
# ---------------------------------------------
accuracy = accuracy_score(
    y_test,
    pred
)
print('Accuracy :', accuracy)

# ---------------------------------------------
#    12. 정밀도(Precision)
#   비유) 화살들이 자기들끼리 얼마나 뭉쳐있는가?
#   AI가 '이탈할 고객'이라고 예측한 사람 중, 실제로 이탈한 비율
# ---------------------------------------------
precision = precision_score(
    y_test,
    pred
)
print("Precision :", precision)

# ---------------------------------------------
#    13. Recall
# ---------------------------------------------
recall = recall_score(
    y_test,
    pred
)
print("Recall :", recall)

# ---------------------------------------------
#    14. F1 Score
# ---------------------------------------------
f1 = f1_score(
    y_test,
    pred
)
print("F1 Score :", f1)

# ---------------------------------------------
#    15. AI 예측 테스트
# ---------------------------------------------
# 샘플 데이터 준비
sample = X_test.iloc[[0]]
print(sample)

# 샘플 데이터 예측
pred = model.predict(sample)
print('예측 결과 : ', pred)
print('정답 : ', y_test.iloc[0])
# ---------------------------------------------
#    16. 샘플 데이터 추가
# ---------------------------------------------
import json
import os
import pandas as pd

os.makedirs("test_data", exist_ok=True)

# 테스트할 고객 인덱스
test_indices = [185, 2715, 3825, 1807, 132]

remove_columns = [
    "Customer ID",
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
    "Churn Label"
]

for i, idx in enumerate(test_indices, start=1):

    sample = df.loc[idx].copy()

    sample = sample.drop(remove_columns, errors="ignore")

    # NaN -> None
    sample = sample.where(pd.notnull(sample), "None")

    filename = f"test_data/test_case_{i:02d}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            sample.to_dict(),
            f,
            indent=4,
            ensure_ascii=False,
            sort_keys=True
        )

print("테스트 JSON 생성 완료")