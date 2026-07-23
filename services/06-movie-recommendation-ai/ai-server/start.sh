#!/bin/bash

SIMILARITY_PATH="models/similarity_matrix.pkl"
METADATA_PATH="models/movie_metadata.pkl"
FEATURES_PATH="models/movie_features.pkl"

echo "Movie Recommendation AI - 서버 시작 준비"

if [ -f "$SIMILARITY_PATH" ] && [ -f "$METADATA_PATH" ] && [ -f "$FEATURES_PATH" ]; then
    echo "학습된 추천 모델 산출물을 발견했습니다."
else
    echo "추천 모델 산출물이 없습니다. 데이터 폴더를 확인합니다."

    if [ -f "data/ml-100k/u.data" ] && [ -f "data/ml-100k/u.item" ] && [ -f "data/ml-100k/u.user" ]; then
        echo "MovieLens 100K 데이터를 발견했습니다. 모델 학습을 시작합니다."
        python train_model.py

        if [ ! -f "$SIMILARITY_PATH" ]; then
            echo "[오류] 모델 학습에 실패했습니다. 위 로그를 확인하세요."
            exit 1
        fi
    else
        echo "======================================================"
        echo "[오류] 모델과 데이터가 모두 없어 서버를 시작할 수 없습니다."
        echo "다음 경로에 MovieLens 100K 데이터를 추가한 뒤 다시 실행하세요."
        echo "  - data/ml-100k/u.data"
        echo "  - data/ml-100k/u.item"
        echo "  - data/ml-100k/u.user"
        echo "자세한 내용은 data/README.md 를 참고하세요."
        echo "======================================================"
        exit 1
    fi
fi

echo "FastAPI 서버를 시작합니다."

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000
