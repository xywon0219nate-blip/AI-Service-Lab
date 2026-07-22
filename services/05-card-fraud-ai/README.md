# Card Fraud Detection AI

AI 기반 금융거래 이상 탐지 서비스 — PaySim 데이터셋을 기반으로 거래가 실행되기 **전**에
위험도를 사전 분석하는 이진 분류(Binary Classification) 서비스입니다.

> 이 서비스는 교육 및 포트폴리오 목적의 AI 서비스이며, 실제 금융기관의 거래 차단이나 최종 판정을
> 대신하지 않습니다.

---

## 1. 프로젝트 소개

전자결제/모바일 송금 서비스에서는 매초 수많은 거래가 발생하며, 이 중 극히 일부가 이상거래(사기)입니다.
이 서비스는 거래 유형·금액·잔액·시간 5가지 정보만으로 해당 거래의 이상거래 위험도를 예측하고,
사람이 참고할 수 있는 형태(위험 확률, 위험 수준, 주요 확인 요인)로 안내합니다.

## 2. 서비스 목표

- 거래가 **실행되기 전** 시점에 판단 가능한 정보만으로 위험도를 사전 탐지
- 정상/사기를 이진 확정하는 것이 아니라 "추가 검토가 필요한 거래"를 안내하는 참고 도구 제공
- 클래스 불균형이 매우 심한 금융 사기 탐지 문제에서 Accuracy가 아닌 올바른 지표로 모델을 검증

## 3. PaySim 데이터셋

[PaySim (Synthetic Financial Datasets For Fraud Detection)](https://www.kaggle.com/datasets/ealaxi/paysim1)은
실제 모바일 금융 서비스 로그의 통계적 특성을 바탕으로 만든 시뮬레이션 데이터셋입니다. 1개월(744 step,
1 step = 1시간)의 거래를 담고 있으며, 타깃 `isFraud`(0=정상, 1=사기)가 라벨링되어 있습니다.

**이 저장소에는 용량 문제로 실제 데이터가 포함되어 있지 않습니다.** 데이터 준비 방법은
[`ai-server/data/README.md`](./ai-server/data/README.md)를 참고하세요.

## 4. 주요 컬럼 설명

| 컬럼 | 의미 |
|---|---|
| `step` | 거래 발생 시간 단계 (1 step = 1시간) |
| `type` | 거래 유형: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | 거래 금액 |
| `nameOrig` | 송금자 식별자 (모델 입력에서 제외) |
| `oldbalanceOrg` | 송금자 거래 전 잔액 |
| `newbalanceOrig` | 송금자 거래 후 잔액 (모델 입력에서 제외 - 데이터 누수) |
| `nameDest` | 수취인 식별자 (모델 입력에서 제외) |
| `oldbalanceDest` | 수취인 거래 전 잔액 |
| `newbalanceDest` | 수취인 거래 후 잔액 (모델 입력에서 제외 - 데이터 누수) |
| `isFraud` | 사기 거래 여부 (Target) |
| `isFlaggedFraud` | PaySim의 기존 규칙 기반 탐지 결과 (모델 입력에서 제외) |

## 5. 사용자 입력값

사용자는 다음 5가지만 입력합니다.

| 화면 표시 | FastAPI 필드 | 모델 컬럼 |
|---|---|---|
| 거래 유형 | `transaction_type` | `type` |
| 거래 금액 | `amount` | `amount` |
| 송금자 거래 전 잔액 | `sender_old_balance` | `oldbalanceOrg` |
| 수취인 거래 전 잔액 | `receiver_old_balance` | `oldbalanceDest` |
| 거래 발생 시간(0~23시) | `transaction_hour` | `Hour` |

사용자는 PaySim의 `step` 값이나 파생변수를 직접 입력하지 않습니다. `Hour`는 학습 시
`(step - 1) % 24`로 계산하며, 예측 시에는 사용자가 0~23시 중 하나를 직접 선택/입력합니다.

## 6. 데이터 누수 방지 원칙

이 서비스는 거래 **사전** 탐지를 목표로 하므로, 거래 이후에만 알 수 있는 정보는 사용하지 않습니다.

- `newbalanceOrig`, `newbalanceDest` : 거래 완료 후에 결정되는 값이므로 기본 입력에서 제외합니다.
  대신 `oldbalanceOrg`/`oldbalanceDest`와 `amount`만으로 계산한 **추정치**
  (`SenderBalanceAfterEstimated`, `ReceiverBalanceAfterEstimated`)를 사용합니다.
- `nameOrig`, `nameDest` : 고객 식별자이므로 제외합니다.
- `isFlaggedFraud` : 기존 규칙 기반 탐지 결과로 타깃과 강하게 연관될 수 있어 제외합니다.

## 7. Feature Engineering

`ai-server/feature_engineering.py`의 `FraudFeatureEngineer`가 학습(`train_model.py`)과 예측
(`main.py`) 양쪽에서 공통으로 사용하는 파생변수를 생성합니다. 원본 5개 입력(`type`, `amount`,
`oldbalanceOrg`, `oldbalanceDest`, `Hour`)만으로 아래 파생변수를 자동 생성합니다.

```text
TimeOfDay                      - Hour를 새벽/오전/오후/저녁 구간으로 변환
AmountLog                      - log1p(amount)
AmountToSenderBalance          - amount / (oldbalanceOrg + 1)
AmountToReceiverBalance        - amount / (oldbalanceDest + 1)
SenderBalanceAfterEstimated    - oldbalanceOrg - amount
ReceiverBalanceAfterEstimated  - oldbalanceDest + amount
IsSenderBalanceZero            - 추정 잔액이 0 이하인지 여부
IsLargeTransaction             - 학습 데이터 기준 고액 거래(상위 5%) 여부
IsTransfer / IsCashOut         - 거래 유형이 TRANSFER / CASH_OUT 인지 여부
BalanceDifference              - oldbalanceOrg - oldbalanceDest
```

모든 파생변수는 나눗셈 분모에 +1을 더해 `NaN`/`Infinity`가 발생하지 않도록 안전하게 처리합니다.
파생변수가 실제로 유용한지는 `ai-server/notebooks/paysim_fraud_analysis.ipynb` 10장에서 그룹별
사기 비율 비교를 통해 검증하도록 구현되어 있으며, 데이터가 준비되는 대로 재검증이 필요합니다.

## 8. 클래스 불균형

PaySim은 사기 거래 비율이 1% 미만인 매우 불균형한 데이터셋입니다. 이 서비스는 다음을 비교합니다.

1. 기본 모델(불균형 미처리)
2. `class_weight="balanced"` (Logistic Regression, Random Forest) / `scale_pos_weight` (XGBoost)
3. Threshold 조정 (0.50 / 0.40 / 0.30 / 0.20)

SMOTE는 선택 항목이라 기본 파이프라인에는 포함하지 않았습니다(불필요한 의존성 추가를 피하기
위함). 필요하다면 `imbalanced-learn`을 추가해 **Train 데이터에만** 적용할 수 있습니다.

## 9. 모델 비교

`train_model.py`는 다음 3~4개 모델을 비교합니다.

- Logistic Regression (`class_weight="balanced"`)
- Random Forest (`class_weight="balanced"`)
- XGBoost (`scale_pos_weight` 적용) — 설치되어 있지 않으면 scikit-learn 내장
  `HistGradientBoostingClassifier`로 자동 대체합니다.
- (비교 기준용) Baseline Random Forest — 불균형 미처리

## 10. 평가 지표

Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC(Average Precision), Confusion Matrix를 계산합니다.
클래스 불균형이 심한 문제에서는 Accuracy만으로 모델을 평가하지 않으며, 특히 **PR-AUC**와
**Fraud Recall**을 중요하게 봅니다.

## 11. 최종 모델 선정 기준

PR-AUC가 가장 높은 모델을 1차 후보로 선정한 뒤, Fraud Recall/Precision/F1, 오탐(False Positive)·
미탐(False Negative) 비용, 추론 속도, Docker 배포 편의성, 모델 크기를 함께 고려해 최종 모델을
선정합니다. 자세한 내용은 `notebooks/paysim_fraud_analysis.ipynb` 16장을 참고하세요.

## 12. Threshold

기본값 0.5 대신 0.40 / 0.30 / 0.20을 함께 비교하고, **F1 Score가 가장 높은 Threshold**를 최종값으로
선택합니다. 선택된 Threshold는 `models/model_metadata.json`에 저장되며, `main.py`가 이 값을 그대로
불러와 예측에 사용합니다.

**위험 수준(Low/Medium/High) 기준** (`main.py`)

```text
fraud_probability <  threshold * 0.5       -> Low
threshold * 0.5 <= fraud_probability < threshold  -> Medium
fraud_probability >= threshold             -> High (Suspicious 판정 구간)
```

## 13. API 명세

베이스 URL: `http://localhost:8000`

| Method | Path | 설명 |
|---|---|---|
| GET | `/` | Health Check |
| GET | `/model-info` | 모델명/버전/Threshold/입력·파생 피처/평가지표 |
| GET | `/samples` | 정상/의심 샘플 거래 (실제 데이터 유무에 따라 `source`가 `real_data` 또는 `demo`) |
| POST | `/predict` | 거래 위험도 예측 |

`POST /predict` 요청 예시:

```json
{
  "transaction_type": "TRANSFER",
  "amount": 181000.0,
  "sender_old_balance": 181000.0,
  "receiver_old_balance": 0.0,
  "transaction_hour": 2
}
```

응답 예시:

```json
{
  "prediction": 1,
  "label": "Suspicious",
  "fraud_probability": 0.8732,
  "normal_probability": 0.1268,
  "threshold": 0.35,
  "risk_level": "High",
  "message": "추가 확인이 필요한 거래로 탐지되었습니다.",
  "risk_factors": ["거래 금액이 송금자 잔액보다 큽니다.", "거래 유형이 TRANSFER입니다."],
  "disclaimer": "본 결과는 AI 모델 기반 참고 정보이며 실제 금융기관의 최종 거래 판단을 대신하지 않습니다."
}
```

`transaction_type`이 허용되지 않은 값이거나 금액/잔액이 음수, 시간이 0~23을 벗어나면 422 응답과
함께 이해하기 쉬운 오류 메시지를 반환합니다. 모델이 아직 학습되지 않았다면 503을 반환합니다.

## 14. React 화면

```text
┌──────────────────────────────────────────────┐
│ Card Fraud Detection AI                      │
│ AI 기반 금융거래 이상 탐지 서비스            │
├────────────────────┬─────────────────────────┤
│ 거래 정보 입력     │ 분석 결과               │
│ 거래 유형          │ 초기 안내 / Loading /   │
│ 거래 금액          │ 오류 / 정상·의심 결과   │
│ 송금 전 잔액       │                         │
│ 수취 전 잔액       │ Fraud Probability Bar   │
│ 거래 시간          │ Risk Level / Threshold  │
│ 정상 샘플 / 의심 샘플 / 초기화 │ 주요 확인 요인          │
│ [위험도 분석하기]  │ 다시 분석하기           │
└────────────────────┴─────────────────────────┘
```

- 정상/의심 샘플 버튼은 `GET /samples` 응답을 그대로 입력폼에 채워 넣으며, 실제 PaySim 데이터에서
  가져온 값인지(`real_data`) 데모 값인지(`demo`) 화면에 표시합니다.
- API 주소는 컴포넌트에 하드코딩하지 않고 `frontend/src/services/api.js` 한 곳에서만 관리합니다.
  기본값은 상대 경로(`""`)이며 Vite Proxy(`vite.config.js`)를 통해 backend로 전달됩니다. 필요시
  `.env`의 `VITE_API_BASE_URL`로 오버라이드할 수 있습니다.

## 15. 로컬 실행

```bash
# Backend
cd services/04-card-fraud-ai/ai-server
pip install -r requirements.txt
python train_model.py        # data/ 에 PaySim CSV가 있을 때만
uvicorn main:app --reload

# Frontend
cd services/04-card-fraud-ai/frontend
npm install
npm run dev
```

## 16. Docker 실행

```bash
cd services/04-card-fraud-ai
docker compose up --build -d
docker compose ps
docker compose logs -f
docker compose down
```

`ai-server/start.sh`가 컨테이너 시작 시 다음 순서로 동작합니다.

```text
모델 파일 있음            -> 바로 FastAPI 실행
모델 없음 + 데이터 있음    -> train_model.py 실행 후 FastAPI 실행
모델 없음 + 데이터 없음    -> 안내 메시지 출력 후 서버 실행하지 않음(exit 1)
```

## 17. AWS EC2 배포

1. EC2 인스턴스에 Docker, Docker Compose를 설치합니다.
2. 이 저장소를 clone하고 `services/04-card-fraud-ai/ai-server/data/`에 PaySim CSV를 업로드합니다
   (모델 파일이 없다면 `.pkl`도 함께 업로드하거나, 최초 기동 시 `start.sh`가 자동으로 학습합니다).
3. `docker compose up --build -d`로 실행합니다.
4. 프론트엔드에서 EC2의 공인 IP로 직접 API를 호출해야 한다면, `frontend/.env`에
   `VITE_API_BASE_URL=http://<EC2-공인IP>:8000`을 설정한 뒤 다시 빌드합니다. (Vite Proxy는 같은
   Docker 네트워크 안에서만 동작하므로, 프록시 없이 외부에서 직접 접근할 때 필요합니다)
5. 보안 그룹에서 8000, 5173 포트를 허용합니다.

## 18. 포트 정보

| 서비스 | 포트 | 컨테이너명 |
|---|---|---|
| Backend (FastAPI) | 8000 | `card-fraud-api` |
| Frontend (React/Vite) | 5173 | `card-fraud-react` |

Project01~03과 동일한 포트(8000/5173)를 사용합니다. 이 저장소의 각 서비스는 독립적으로
한 번에 하나씩 실행하는 것을 전제로 하며(각 프로젝트가 동일한 포트 컨벤션을 공유), 컨테이너 이름만
프로젝트별로 구분됩니다. 여러 서비스를 **동시에** 띄워야 한다면 `docker-compose.yml`의 포트 매핑을
변경하세요.

## 19. 모델 파일 관리

- `ai-server/models/fraud_detection_pipeline.pkl` : Feature Engineering + 전처리 + 최종 모델이
  하나로 저장된 scikit-learn Pipeline. Git에는 포함하지 않습니다(`.gitignore`).
- `ai-server/models/model_metadata.json` : 모델명/버전/Threshold/피처 목록/평가지표. 용량이 작고
  민감정보가 없어 Git에 포함합니다.
- 모델 파일을 Git에서 제외했으므로, AWS 서버에는 (1) 학습된 `.pkl`을 직접 업로드하거나 (2) PaySim
  데이터를 업로드해 `start.sh`가 최초 기동 시 자동으로 학습하도록 합니다.

## 20. 데이터 파일 관리

- `ai-server/data/*.csv`는 용량 문제로 Git에서 제외합니다(`.gitignore`).
- 데이터 준비 방법은 [`ai-server/data/README.md`](./ai-server/data/README.md)를 참고하세요.
- 대용량 원본에서 학습/강의용 샘플만 만들고 싶다면 `python train_model.py --build-sample`을
  실행하세요. 사기 거래는 전부 보존하고 정상 거래만 샘플링하여 `data/paysim_sample.csv`를
  생성합니다(평가에는 사용하지 않고 학습용으로만 사용하는 것을 권장합니다).

## 21. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 서버가 시작되지 않고 바로 종료됨 | 모델과 데이터가 모두 없는 상태입니다. `data/README.md`에 안내된 경로에 CSV를 추가하세요. |
| `POST /predict`가 503을 반환 | 모델이 아직 학습되지 않았습니다. `python train_model.py`를 실행하거나 데이터 추가 후 컨테이너를 재시작하세요. |
| `POST /predict`가 422를 반환 | 거래 유형이 5가지 값 중 하나가 아니거나, 금액/잔액이 음수이거나, 시간이 0~23 범위를 벗어났습니다. |
| React에서 "서버에 연결할 수 없습니다" | FastAPI가 실행 중인지, Docker Compose라면 `backend` 컨테이너가 정상인지 확인하세요(`docker compose logs backend`). |
| 학습이 너무 오래 걸림/메모리 부족 | 원본 CSV가 매우 큰 경우입니다. `python train_model.py --build-sample`로 샘플을 만들어 `data/paysim_sample.csv`로 학습해보세요. |

## 22. 금융 서비스 주의사항

- 이 서비스는 **교육 및 포트폴리오 목적**으로 제작되었으며, 실제 금융기관의 거래 차단·승인·최종
  사기 판정을 대신하지 않습니다.
- 응답의 `label`은 `Normal` / `Suspicious`로만 표현하며, 사기를 확정하는 표현(`Fraud Confirmed` 등)은
  사용하지 않습니다.
- `risk_factors`는 모델 내부 값을 그대로 노출하지 않는 입력 기반 참고 설명이며, 모델의 정확한
  인과관계를 의미하지 않습니다.
- 실제 서비스에 적용하려면 최신 데이터로 주기적인 재학습, Threshold 재검토, 법률/컴플라이언스
  검토가 반드시 필요합니다.
