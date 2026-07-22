#!/bin/bash

MODEL_PATH="models/fraud_detection_pipeline.pkl"

echo "Card Fraud Detection AI - 서버 시작 준비"

if [ -f "$MODEL_PATH" ]; then
    echo "학습된 모델을 발견했습니다: $MODEL_PATH"
else
    echo "학습된 모델이 없습니다. 데이터 파일을 확인합니다."

    DATA_FOUND=0
    for f in data/paysim.csv data/paysim_sample.csv data/PS_20174392719_1491204439457_log.csv data/PaySim.csv data/fraud.csv; do
        if [ -f "$f" ]; then
            DATA_FOUND=1
            break
        fi
    done

    if [ "$DATA_FOUND" -eq 1 ]; then
        echo "데이터 파일을 발견했습니다. 모델 학습을 시작합니다."
        python train_model.py

        if [ ! -f "$MODEL_PATH" ]; then
            echo "[오류] 모델 학습에 실패했습니다. 위 로그를 확인하세요."
            exit 1
        fi
    else
        echo "======================================================"
        echo "[오류] 모델과 데이터 파일이 모두 없어 서버를 시작할 수 없습니다."
        echo "다음 경로 중 하나에 PaySim CSV 파일을 추가한 뒤 다시 실행하세요."
        echo "  - data/paysim.csv"
        echo "  - data/paysim_sample.csv"
        echo "  - data/PS_20174392719_1491204439457_log.csv"
        echo "  - data/PaySim.csv"
        echo "  - data/fraud.csv"
        echo "자세한 내용은 data/README.md 를 참고하세요."
        echo "======================================================"
        exit 1
    fi
fi

echo "FastAPI 서버를 시작합니다."

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000
