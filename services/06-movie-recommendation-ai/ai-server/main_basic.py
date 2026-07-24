# ==========================================================
# Movie Recommendation AI - FastAPI (Basic)
# ST_movie_recommendation_basic.ipynb 기반 - 장르 콘텐츠(코사인 유사도) 추천
#
# train_model_basic.py로 학습한 산출물(models_basic/)을 읽어서 서빙한다.
# ==========================================================

import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering import DEFAULT_TOP_N, TOP_N_MAX, TOP_N_MIN


# ==========================================================
# 1. 경로 / 상수
# ==========================================================

MODEL_DIR = "models_basic"
METADATA_PATH = os.path.join(MODEL_DIR, "movie_metadata.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "movie_features.pkl")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# 평점 수가 이 값 미만인 영화는 장르만 비슷할 뿐 신뢰하기 어려워 추천에서 제외한다.
# (ST_movie_recommendation_basic.ipynb 10번 섹션 "최소 평점 수 적용" 참고)
MIN_RATING_COUNT = 10

MOVIE_LIST_DEFAULT_LIMIT = 20
MOVIE_LIST_MAX_LIMIT = 200


# ==========================================================
# 2. FastAPI 앱 생성 및 CORS 설정
# ==========================================================

app = FastAPI(
    title="Movie Recommendation AI (Basic)",
    description="장르 기반 콘텐츠 유사도 추천 서비스 - ST_movie_recommendation_basic.ipynb 기반 (교육 목적)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize_validation_errors(errors):
    """Pydantic 오류의 ctx에 담긴 예외 객체(JSON으로 직렬화 불가능)를 문자열로 바꾼다."""
    sanitized = []
    for error in errors:
        error = dict(error)
        ctx = error.get("ctx")
        if isinstance(ctx, dict):
            error["ctx"] = {key: str(value) for key, value in ctx.items()}
        sanitized.append(error)
    return sanitized


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {
                "error": "입력값을 확인해주세요.",
                "details": _sanitize_validation_errors(exc.errors()),
            }
        ),
    )


# ==========================================================
# 3. 모델(추천 산출물) 로딩
#    - 산출물이 없어도 서버 자체는 켜지도록 하고, /recommend 등에서
#      503과 함께 안내 메시지를 반환한다.
# ==========================================================

def load_basic_model_artifacts():
    required = [METADATA_PATH, FEATURES_PATH]
    if not all(os.path.exists(p) for p in required):
        print("[안내] Basic 추천 모델 산출물이 없습니다.")
        print("       data/ 폴더에 movies.csv, ratings.csv 를 추가한 뒤 train_model_basic.py를 실행하세요.")
        return None, None, None

    try:
        movie_metadata = joblib.load(METADATA_PATH)
        movie_features = joblib.load(FEATURES_PATH)
    except Exception as error:  # noqa: BLE001 - 로딩 실패 시 서버가 죽지 않고 안내만 한다.
        print(f"[경고] Basic 모델 산출물 로딩 중 오류가 발생했습니다: {error}")
        return None, None, None

    model_info = None
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            model_info = json.load(f)

    return movie_metadata, movie_features, model_info


movie_metadata, movie_features, model_info = load_basic_model_artifacts()
MODEL_LOADED = movie_metadata is not None


# ==========================================================
# 4. 요청 스키마
# ==========================================================

class RecommendRequest(BaseModel):
    movie_id: int = Field(..., description="추천의 기준이 되는 영화 ID (movieId)")
    top_n: int = Field(
        DEFAULT_TOP_N, ge=TOP_N_MIN, le=TOP_N_MAX, description=f"추천 개수 ({TOP_N_MIN}~{TOP_N_MAX})"
    )

    model_config = {
        "json_schema_extra": {"example": {"movie_id": 1, "top_n": 10}}
    }


# ==========================================================
# 5. 추천 로직
#    - ST_movie_recommendation_basic.ipynb의 recommend_similar_movies(마지막 버전,
#      "18. 추천 이유 추가")를 옮기되 두 가지를 바꿨다.
#      1) title 문자열 대신 movieId로 조회한다 (API 입력으로 더 안전함).
#      2) "rating_count >= MIN_RATING_COUNT 이면 continue(제외)"로 되어 있던
#         노트북의 조건 실수를 "rating_count >= MIN_RATING_COUNT 인 영화만 포함"
#         (원래 의도, 10번 섹션 참고)으로 바로잡았다.
#    - 전체 영화 간 유사도 행렬을 저장해두지 않고, 질의한 영화 1개와 나머지
#      영화 전체의 코사인 유사도를 요청마다 계산한다 (train_model_basic.py 주석 참고).
# ==========================================================

def get_common_genres(base_genres: str, other_genres: str):
    base_set = set(base_genres.split("|"))
    other_set = set(other_genres.split("|"))
    return sorted(base_set & other_set)


def build_recommendation_reason(common_genres):
    if common_genres:
        return f"{', '.join(common_genres)} 장르가 유사합니다."
    return "장르 구성이 유사합니다."


def serialize_movie(movie_id: int):
    row = movie_metadata.loc[movie_id]
    return {
        "movie_id": int(movie_id),
        "title": row["title"],
        "genres": row["genres"].split("|"),
    }


def recommend_similar_movies(movie_id: int, top_n: int):
    """movie_id와 장르가 유사한 영화 top_n개를 추천한다.

    Returns:
        recommendations: list[dict]
    Raises:
        KeyError: movie_id가 movie_metadata에 없는 경우
    """
    base_row = movie_metadata.loc[movie_id]  # KeyError -> 호출부에서 404 처리
    base_genres = base_row["genres"]

    target_vector = movie_features.loc[[movie_id]]
    candidate_features = movie_features.drop(index=movie_id, errors="ignore")
    raw_scores = cosine_similarity(target_vector, candidate_features)[0]
    sims = pd.Series(raw_scores, index=candidate_features.index).sort_values(ascending=False)

    recommendations = []
    for candidate_id, score in sims.items():
        candidate_row = movie_metadata.loc[candidate_id]

        if candidate_row["rating_count"] < MIN_RATING_COUNT:
            continue

        common_genres = get_common_genres(base_genres, candidate_row["genres"])

        recommendations.append(
            {
                "movie_id": int(candidate_id),
                "title": candidate_row["title"],
                "genres": candidate_row["genres"].split("|"),
                "similarity_score": round(float(score), 3),
                "rating_count": int(candidate_row["rating_count"]),
                "rating_mean": round(float(candidate_row["rating_mean"]), 2),
                "recommendation_reason": build_recommendation_reason(common_genres),
            }
        )

        if len(recommendations) == top_n:
            break

    return recommendations


# ==========================================================
# 6. Root / Health Check
# ==========================================================

@app.get("/")
def root():
    return {
        "service": "Movie Recommendation AI (Basic)",
        "algorithm": "content_based_genre_cosine_similarity",
        "status": "running",
        "model_loaded": MODEL_LOADED,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "num_movies": int(movie_metadata.shape[0]) if MODEL_LOADED else 0,
    }


@app.get("/model-info")
def model_info_endpoint():
    if model_info is None:
        return {
            "model_loaded": MODEL_LOADED,
            "message": (
                "학습된 Basic 추천 모델 정보가 없습니다. data/ 폴더에 movies.csv, ratings.csv 를 추가한 뒤 "
                "train_model_basic.py를 실행하세요."
            ),
        }
    return {"model_loaded": MODEL_LOADED, **model_info}


# ==========================================================
# 7. 영화 목록 / 검색 API
# ==========================================================

@app.get("/movies")
def list_movies(
    search: str = Query("", description="영화 제목 검색어 (부분 일치, 대소문자 무시)"),
    limit: int = Query(MOVIE_LIST_DEFAULT_LIMIT, ge=1, le=MOVIE_LIST_MAX_LIMIT),
):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="영화 목록이 아직 준비되지 않았습니다. data/ 폴더에 movies.csv, ratings.csv 를 추가한 뒤 다시 시도하세요.",
        )

    df = movie_metadata
    if search.strip():
        keyword = search.strip().lower()
        df = df[df["title"].str.lower().str.contains(keyword, regex=False)]

    df = df.sort_values("rating_count", ascending=False).head(limit)

    return [
        {
            "movie_id": int(movie_id),
            "title": row["title"],
            "genres": row["genres"].split("|"),
            "rating_count": int(row["rating_count"]),
            "rating_mean": round(float(row["rating_mean"]), 2),
        }
        for movie_id, row in df.iterrows()
    ]


# ==========================================================
# 8. 추천 API
# ==========================================================

def _recommend_response(movie_id: int, top_n: int):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=(
                "추천 모델이 아직 준비되지 않았습니다. data/ 폴더에 movies.csv, ratings.csv 를 추가한 뒤 "
                "train_model_basic.py를 실행하거나 서버를 다시 시작하세요."
            ),
        )

    if movie_id not in movie_metadata.index:
        raise HTTPException(status_code=404, detail=f"movie_id={movie_id} 에 해당하는 영화를 찾을 수 없습니다.")

    recommendations = recommend_similar_movies(movie_id, top_n)

    if not recommendations:
        raise HTTPException(status_code=404, detail="추천할 유사 영화를 찾지 못했습니다.")

    return {
        "selected_movie": serialize_movie(movie_id),
        "algorithm": "content_based_genre_cosine_similarity",
        "requested_top_n": top_n,
        "returned_count": len(recommendations),
        "recommendations": recommendations,
    }


@app.post("/recommend")
def recommend(payload: RecommendRequest):
    return _recommend_response(payload.movie_id, payload.top_n)


@app.get("/recommend/{movie_id}")
def recommend_by_path(movie_id: int, top_n: int = Query(DEFAULT_TOP_N, ge=TOP_N_MIN, le=TOP_N_MAX)):
    return _recommend_response(movie_id, top_n)
