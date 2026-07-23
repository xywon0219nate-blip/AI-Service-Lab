# MovieLens 100K 데이터 파일 안내

이 폴더에는 실제 MovieLens 100K 데이터가 포함되어 있지 않습니다 (라이선스/용량 문제로 Git에 포함하지 않음).

## 1. 데이터 준비 방법

[GroupLens](https://grouplens.org/datasets/movielens/100k/)에서 제공하는 **MovieLens 100K Dataset**을
내려받아 압축을 풀고, 다음 경로에 그대로 저장하십시오.

```bash
cd services/05-movie-recommendation-ai/ai-server/data
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

압축을 풀면 다음과 같은 폴더 구조가 만들어져야 합니다.

```text
services/05-movie-recommendation-ai/ai-server/data/
└── ml-100k/
    ├── u.data
    ├── u.item
    ├── u.user
    ├── u.genre
    ├── u.occupation
    ├── ua.base / ua.test  (Notebook의 평가 절에서만 사용)
    └── ...
```

`train_model.py`와 `feature_engineering.py`는 `data/ml-100k/`, `data/ML-100k/` 등 몇 가지 후보 폴더명을
자동으로 탐색합니다. 원본 압축 파일의 폴더명(`ml-100k`)을 그대로 두면 별도 설정 없이 바로 동작합니다.

## 2. 필요한 파일

```text
u.data   - 사용자별 영화 평점 (user_id, movie_id, rating, timestamp)
u.item   - 영화 메타데이터 (movie_id, title, release_date, genre 19개 컬럼)
u.user   - 사용자 정보 (user_id, age, gender, occupation, zip_code)
u.genre  - 장르 코드 목록
u.occupation - 직업 코드 목록
```

## 3. 데이터가 없을 때 동작

- `train_model.py`: 데이터 폴더를 찾지 못하면 **가짜 성능을 만들지 않고** 안내 메시지를 출력한 뒤 종료합니다.
- `ai-server/start.sh`: 모델과 데이터가 모두 없으면 FastAPI 서버를 실행하지 않고 동일한 안내를 출력합니다.
- `fe/movie_recommendation_analysis.ipynb`: 데이터 로딩 셀에서 파일 존재 여부를 확인하고, 없으면 이 안내를
  다시 보여준 뒤 이후 셀 실행을 중단하도록 안내합니다.

## 4. 인코딩 / 구분자 주의사항

- `u.data`는 탭(`\t`) 구분자를 사용합니다.
- `u.item`, `u.user`는 파이프(`|`) 구분자를 사용하며 헤더가 없습니다.
- `u.item`은 Latin-1(`latin-1`) 인코딩입니다. UTF-8로 읽으면 일부 특수문자에서 오류가 발생합니다.
- `u.item`의 `video_release_date` 컬럼은 데이터셋 전체에서 항상 결측이라 사용하지 않습니다.

## 5. 데이터 추가 후 실행 순서

```bash
cd services/05-movie-recommendation-ai/ai-server
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

또는 Docker Compose로 실행하면 `start.sh`가 산출물 존재 여부를 확인해 필요할 때만 자동으로 학습합니다.

```bash
cd services/05-movie-recommendation-ai
docker compose up --build -d
```
