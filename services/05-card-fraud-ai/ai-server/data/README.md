# PaySim 데이터 파일 안내

이 폴더에는 실제 PaySim 데이터가 포함되어 있지 않습니다 (용량 문제로 Git에 포함하지 않음).

## 1. 데이터 준비 방법

Kaggle의 **PaySim (Synthetic Financial Datasets For Fraud Detection)** 데이터셋을 내려받아
다음 경로에 그대로 저장하십시오.

```text
services/04-card-fraud-ai/ai-server/data/paysim.csv
```

원본 Kaggle 파일명(`PS_20174392719_1491204439457_log.csv`)을 그대로 두어도 됩니다.
`train_model.py`와 `notebooks/paysim_fraud_analysis.ipynb`는 아래 순서로 파일을 자동으로 탐색합니다.

```text
1. data/paysim.csv
2. data/paysim_sample.csv
3. data/PS_20174392719_1491204439457_log.csv
4. data/PaySim.csv
5. data/fraud.csv
```

## 2. 필요한 컬럼

```text
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
```

실제 파일의 컬럼명이 다르면 코드가 실행 시점에 컬럼을 확인하고 오류 메시지로 알려줍니다.

## 3. 데이터가 없을 때 동작

- `train_model.py`: 데이터 파일을 찾지 못하면 **가짜 성능을 만들지 않고** 안내 메시지를 출력한 뒤 종료합니다.
- `ai-server/start.sh`: 모델과 데이터가 모두 없으면 FastAPI 서버를 실행하지 않고 동일한 안내를 출력합니다.
- `notebooks/paysim_fraud_analysis.ipynb`: 데이터 로딩 셀에서 파일 존재 여부를 확인하고, 없으면 이 안내를 다시 보여준 뒤 이후 셀 실행을 중단하도록 안내합니다.

## 4. 대용량 데이터 처리

PaySim 원본 CSV는 6백만 행 이상으로 매우 큽니다. `train_model.py`는 다음을 지원합니다.

- 필요한 컬럼만 `usecols`로 로딩
- 메모리 절약을 위한 `dtype` 지정 (`type`은 `category`, 금액/잔액은 `float32`, `step`은 `int32`)
- 파일이 매우 큰 경우 `chunksize` 기반 스트리밍 처리
- 개발/강의용 샘플 생성: 정상 거래는 무작위 샘플링하되 **사기 거래(`isFraud == 1`) 행은 전부 보존**하여
  `data/paysim_sample.csv`로 저장 (자세한 내용은 `train_model.py`의 `build_training_sample()` 참고)

샘플링은 **학습 데이터에만** 적용하며, 평가(Test) 데이터의 클래스 분포는 왜곡하지 않습니다.

## 5. 데이터 추가 후 실행 순서

```bash
cd services/04-card-fraud-ai/ai-server
python train_model.py
uvicorn main:app --reload
```

또는 Docker Compose로 실행하면 `start.sh`가 모델 존재 여부를 확인해 필요할 때만 자동으로 학습합니다.

```bash
cd services/04-card-fraud-ai
docker compose up --build -d
```
