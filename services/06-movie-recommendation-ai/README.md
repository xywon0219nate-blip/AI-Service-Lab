# Movie Recommendation AI

MovieLens 100K 데이터셋을 기반으로, 사용자가 선택한 영화와 유사한 영화를 추천하는 AI
서비스입니다.

> 이 서비스는 교육 및 포트폴리오 목적으로 제작되었으며, 실제 상용 스트리밍 서비스의 추천 품질을
> 대신하지 않습니다.

---

## 1. 프로젝트 소개

사용자가 영화 제목을 검색해 한 편을 선택하면, 그 영화와 비슷한 영화 목록과 추천 이유를 함께
보여주는 서비스입니다. 로그인이나 시청 이력 없이 "영화 한 편 선택 -> 추천"만으로 동작합니다.

## 2. 주요 기능

- 영화 제목 검색 (부분 일치, 대소문자 무시)
- 선택한 영화 기준 Top-N 유사 영화 추천 (기본 10개, 5~30개 조정 가능)
- 추천 이유 표시 (협업 필터링 근거 또는 공통 장르)
- 평점 데이터가 부족한 영화(Cold Start)에 대한 자동 대체 추천

## 3. 기술 스택

**Backend**: Python, FastAPI, Uvicorn, Pandas, NumPy, scikit-learn, Joblib, Pydantic
**Frontend**: React, Vite, JavaScript, Axios
**실행 환경**: Docker, Docker Compose

## 4. 폴더 구조

```text
services/05-movie-recommendation-ai/
├── ai-server/
│   ├── data/
│   │   ├── README.md
│   │   └── ml-100k/              # MovieLens 100K 원본 (Git 제외)
│   ├── fe/
│   │   └── movie_recommendation_analysis.ipynb
│   ├── models/                   # 학습 산출물 (model_info.json만 Git 포함)
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── start.sh
├── frontend/
│   ├── public/favicon.svg
│   ├── src/
│   │   ├── components/
│   │   │   ├── MovieSearch.jsx
│   │   │   └── RecommendationPanel.jsx
│   │   ├── services/api.js
│   │   ├── App.jsx / App.css
│   │   ├── main.jsx / index.css
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
├── Research_Report.md
├── Instructor_Feature_Engineering_Guide.md
├── docker-compose.yml
├── .gitignore
└── README.md
```

## 5. MovieLens 100K 데이터 준비

이 저장소에는 라이선스/용량 문제로 실제 데이터가 포함되어 있지 않습니다.

```bash
cd services/05-movie-recommendation-ai/ai-server/data
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

저장 경로: `services/05-movie-recommendation-ai/ai-server/data/ml-100k/`. 자세한 내용은
[`ai-server/data/README.md`](./ai-server/data/README.md)를 참고하세요.

## 6. 로컬 실행

```bash
# Backend
cd services/05-movie-recommendation-ai/ai-server
pip install -r requirements.txt
python train_model.py        # data/ml-100k/ 에 데이터가 있을 때만
uvicorn main:app --reload

# Frontend
cd services/05-movie-recommendation-ai/frontend
npm install
npm run dev
```

## 7. Docker 실행

```bash
cd services/05-movie-recommendation-ai
docker compose up --build -d
docker compose ps
docker compose logs -f
docker compose down
```

`ai-server/start.sh`가 컨테이너 시작 시 다음 순서로 동작합니다.

```text
모델 산출물 있음               -> 바로 FastAPI 실행
모델 없음 + 데이터 있음         -> train_model.py 실행 후 FastAPI 실행
모델 없음 + 데이터 없음         -> 안내 메시지 출력 후 서버 실행하지 않음(exit 1)
```

## 8. 접속 주소

| 서비스 | 주소 |
|---|---|
| Backend (FastAPI) | http://localhost:8000 |
| Swagger 문서 | http://localhost:8000/docs |
| Frontend (React/Vite) | http://localhost:5173 |

## 9. API 목록

| Method | Path | 설명 |
|---|---|---|
| GET | `/` | 서비스 상태 |
| GET | `/health` | Health Check (모델 로딩 상태 포함) |
| GET | `/model-info` | 모델명/알고리즘/평점 통계/데이터 규모 |
| GET | `/movies?search=&limit=` | 영화 목록 조회/검색 |
| POST | `/recommend` | 추천 (`{"movie_id": 1, "top_n": 10}`) |
| GET | `/recommend/{movie_id}?top_n=` | 추천 (경로 파라미터 버전) |

`POST /recommend` 응답 예시:

```json
{
  "selected_movie": {
    "movie_id": 1,
    "title": "Toy Story (1995)",
    "clean_title": "Toy Story",
    "release_year": 1995,
    "genres": ["Animation", "Children's", "Comedy"]
  },
  "algorithm": "item_based_cf",
  "requested_top_n": 5,
  "returned_count": 5,
  "recommendations": [
    {
      "movie_id": 588,
      "title": "Beauty and the Beast (1991)",
      "release_year": 1991,
      "genres": ["Animation", "Children's", "Musical"],
      "similarity_score": 0.2291,
      "average_rating": 3.79,
      "rating_count": 202,
      "recommendation_reason": "이 영화를 좋아한 다른 사용자들이 함께 높게 평가한 영화입니다. (공통 장르: Animation, Children's)"
    }
  ]
}
```

존재하지 않는 `movie_id`는 404, `top_n`이 1~30 범위를 벗어나면 422, 모델이 아직 준비되지 않았으면
503을 반환합니다.

## 10. 모델 학습 / 재생성

```bash
cd services/05-movie-recommendation-ai/ai-server
python train_model.py
```

산출물: `models/similarity_matrix.pkl`, `models/movie_metadata.pkl`, `models/movie_features.pkl`,
`models/model_info.json`. `.pkl` 파일은 Git에서 제외되며, `model_info.json`만 Git에 포함됩니다.

## 11. 추천 알고리즘

**Item-Based Collaborative Filtering**(평점 20건 이상인 939개 영화 대상)을 기본으로 사용하고,
평점이 부족한 영화(Cold Start)는 **Content-Based(장르 유사도)**로 자동 대체합니다. 알고리즘 선정
과정과 정량/정성 비교는 [`Research_Report.md`](./Research_Report.md) 11.6~11.8절을 참고하세요.

## 12. 데이터 파일 / 모델 파일 관리

- `ai-server/data/ml-100k/`: 용량 문제로 Git에서 제외합니다. 준비 방법은
  [`ai-server/data/README.md`](./ai-server/data/README.md) 참고.
- `ai-server/models/*.pkl`: Git에서 제외합니다. `model_info.json`은 용량이 작고 통계 정보만 담고
  있어 Git에 유지합니다.
- 데이터를 추가하면 `start.sh`(Docker) 또는 `python train_model.py`(로컬)로 언제든 재생성할 수
  있습니다.

## 13. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 서버가 시작되지 않고 바로 종료됨 | 모델과 데이터가 모두 없는 상태입니다. `data/README.md`에 안내된 경로에 데이터를 추가하세요. |
| `POST /recommend`가 503을 반환 | 모델이 아직 준비되지 않았습니다. `python train_model.py`를 실행하거나 데이터 추가 후 컨테이너를 재시작하세요. |
| `POST /recommend`가 404를 반환 | 존재하지 않는 `movie_id`입니다. `GET /movies`로 유효한 ID를 확인하세요. |
| React에서 "서버에 연결할 수 없습니다" | FastAPI가 실행 중인지, Docker Compose라면 `backend` 컨테이너가 정상인지 확인하세요(`docker compose logs backend`). |
| 검색 결과에 같은 제목이 두 번 나옴 | MovieLens 100K에 존재하는 18쌍의 중복 제목 특성입니다(Research_Report.md 11.4절). |

## 14. Project04와 공통된 구조

- Backend/Frontend 분리, FastAPI + React(Vite) 구성
- `start.sh`가 모델 존재 여부를 확인해 필요할 때만 학습 후 서버를 실행하는 방식
- `feature_engineering.py`를 학습(`train_model.py`)과 서빙(`main.py`)이 공유하는 구조
- 데이터/모델 파일을 Git에서 제외하고 안내 메시지로 준비 방법을 알려주는 정책
- CORS, 422/503 오류 처리, Swagger 문서, React의 Loading/Error/초기 상태 UI 패턴

## 15. Project05에서 변경된 부분

- 문제 유형: 이진 분류(사기 탐지) -> 추천(유사 아이템 랭킹)
- 데이터셋: PaySim -> MovieLens 100K
- 알고리즘: 분류 모델 비교(Logistic Regression/Random Forest/XGBoost) -> 추천 알고리즘 비교
  (Popularity/Content-Based/Item-Based CF)
- 모델 산출물: 단일 `.pkl` 파이프라인 -> 유사도 행렬 + 메타데이터 + 장르 피처로 분리 저장
- React 화면: 거래 정보 입력 폼 -> 영화 검색/선택 UI
