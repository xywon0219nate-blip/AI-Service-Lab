#!/bin/bash

echo "FastAPI 서버를 시작합니다."

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000