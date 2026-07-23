# ==========================================================
# Movie Recommendation AI - FastAPI
# MovieLens 100K 기반 영화 추천 API
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

MODEL_DIR = "models"
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity_matrix.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "movie_metadata.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "movie_features.pkl")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

MOVIE_LIST_DEFAULT_LIMIT = 20
MOVIE_LIST_MAX_LIMIT = 200


# ==========================================================
# 2. FastAPI 앱 생성 및 CORS 설정
# ==========================================================

app = FastAPI(
    title="Movie Recommendation AI",
    description="MovieLens 100K 기반 영화 추천 서비스 (교육/포트폴리오 목적)",
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
#      503과 함께 안내 메시지를 반환한다. (start.sh가 컨테이너 진입 시
#      먼저 학습 여부를 판단하지만, main.py도 방어적으로 동작한다.)
# ==========================================================

def load_recommendation_artifacts():
    required = [SIMILARITY_PATH, METADATA_PATH, FEATURES_PATH]
    if not all(os.path.exists(p) for p in required):
        print("[안내] 추천 모델 산출물이 없습니다.")
        print("       data/ 폴더에 MovieLens 100K 데이터를 추가한 뒤 train_model.py를 실행하세요.")
        return None, None, None, None

    try:
        similarity_matrix = joblib.load(SIMILARITY_PATH)
        movie_metadata = joblib.load(METADATA_PATH)
        movie_features = joblib.load(FEATURES_PATH)
    except Exception as error:  # noqa: BLE001 - 로딩 실패 시 서버가 죽지 않고 안내만 한다.
        print(f"[경고] 모델 산출물 로딩 중 오류가 발생했습니다: {error}")
        return None, None, None, None

    model_info = None
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            model_info = json.load(f)

    return similarity_matrix, movie_metadata, movie_features, model_info


similarity_matrix, movie_metadata, movie_features, model_info = load_recommendation_artifacts()
MODEL_LOADED = movie_metadata is not None


# ==========================================================
# 4. 요청 / 응답 스키마
# ==========================================================

class RecommendRequest(BaseModel):
    movie_id: int = Field(..., description="추천의 기준이 되는 영화 ID")
    top_n: int = Field(
        DEFAULT_TOP_N, ge=TOP_N_MIN, le=TOP_N_MAX, description=f"추천 개수 ({TOP_N_MIN}~{TOP_N_MAX})"
    )

    model_config = {
        "json_schema_extra": {"example": {"movie_id": 1, "top_n": 10}}
    }


# ==========================================================
# 5. 추천 로직
#    - Item-Based Collaborative Filtering을 기본으로 사용한다.
#    - 평점 수가 적어 유사도 행렬에 포함되지 않은 영화(Cold Start)는
#      장르 기반 콘텐츠 유사도로 대체한다.
# ==========================================================

def get_common_genres(base_genres, other_genres):
    base_set = set(base_genres)
    return [g for g in other_genres if g in base_set]


def build_recommendation_reason(algorithm, common_genres):
    if algorithm == "item_based_cf":
        if common_genres:
            return f"이 영화를 좋아한 다른 사용자들이 함께 높게 평가한 영화입니다. (공통 장르: {', '.join(common_genres)})"
        return "이 영화를 좋아한 다른 사용자들이 함께 높게 평가한 영화입니다."

    # content_based (cold start fallback)
    if common_genres:
        return f"선택한 영화와 {', '.join(common_genres)} 장르가 유사합니다."
    return "선택한 영화와 장르 구성이 유사한 영화입니다."


def serialize_movie(movie_id: int):
    row = movie_metadata.loc[movie_id]
    return {
        "movie_id": int(movie_id),
        "title": row["title"],
        "clean_title": row["clean_title"],
        "release_year": None if pd.isna(row["release_year"]) else int(row["release_year"]),
        "genres": row["genres"],
    }


def recommend_similar_movies(movie_id: int, top_n: int):
    """movie_id와 유사한 영화 top_n개를 추천한다.

    Returns:
        (algorithm, recommendations: list[dict])
    Raises:
        KeyError: movie_id가 movie_metadata에 없는 경우
    """
    base_row = movie_metadata.loc[movie_id]  # KeyError -> 호출부에서 404 처리
    base_genres = base_row["genres"]

    if movie_id in similarity_matrix.index:
        algorithm = "item_based_cf"
        sims = similarity_matrix.loc[movie_id].drop(labels=[movie_id], errors="ignore")
        sims = sims.sort_values(ascending=False).head(top_n)
    else:
        # Cold Start: 평점이 부족해 협업 필터링 대상에서 제외된 영화 -> 장르 기반 대체
        algorithm = "content_based"
        target_vector = movie_features.loc[[movie_id]]
        candidate_features = movie_features.drop(index=movie_id, errors="ignore")
        raw_scores = cosine_similarity(target_vector, candidate_features)[0]
        sims = pd.Series(raw_scores, index=candidate_features.index)
        sims = sims.sort_values(ascending=False).head(top_n)

    recommendations = []
    for candidate_id, score in sims.items():
        candidate_row = movie_metadata.loc[candidate_id]
        common_genres = get_common_genres(base_genres, candidate_row["genres"])
        recommendations.append(
            {
                "movie_id": int(candidate_id),
                "title": candidate_row["title"],
                "release_year": None if pd.isna(candidate_row["release_year"]) else int(candidate_row["release_year"]),
                "genres": candidate_row["genres"],
                "similarity_score": round(float(score), 4),
                "average_rating": round(float(candidate_row["rating_mean"]), 2),
                "rating_count": int(candidate_row["rating_count"]),
                "recommendation_reason": build_recommendation_reason(algorithm, common_genres),
            }
        )

    return algorithm, recommendations


# ==========================================================
# 6. Root / Health Check
# ==========================================================

@app.get("/")
def root():
    return {
        "service": "Movie Recommendation AI",
        "dataset": "MovieLens 100K",
        "status": "running",
        "model_loaded": MODEL_LOADED,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "num_movies": int(movie_metadata.shape[0]) if MODEL_LOADED else 0,
        "num_movies_in_cf_matrix": int(similarity_matrix.shape[0]) if MODEL_LOADED else 0,
    }


@app.get("/model-info")
def model_info_endpoint():
    if model_info is None:
        return {
            "model_loaded": MODEL_LOADED,
            "message": (
                "학습된 추천 모델 정보가 없습니다. data/ 폴더에 MovieLens 100K 데이터를 추가한 뒤 "
                "train_model.py를 실행하세요."
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
            detail="영화 목록이 아직 준비되지 않았습니다. data/ 폴더에 데이터를 추가한 뒤 다시 시도하세요.",
        )

    df = movie_metadata
    if search.strip():
        keyword = search.strip().lower()
        df = df[df["search_title"].str.contains(keyword, regex=False)]
        df = df.sort_values("rating_count", ascending=False)
    else:
        df = df.sort_values("popularity_score", ascending=False)

    df = df.head(limit)

    return [
        {
            "movie_id": int(movie_id),
            "title": row["title"],
            "clean_title": row["clean_title"],
            "release_year": None if pd.isna(row["release_year"]) else int(row["release_year"]),
            "genres": row["genres"],
            "average_rating": round(float(row["rating_mean"]), 2),
            "rating_count": int(row["rating_count"]),
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
                "추천 모델이 아직 준비되지 않았습니다. data/ 폴더에 MovieLens 100K 데이터를 추가한 뒤 "
                "train_model.py를 실행하거나 서버를 다시 시작하세요."
            ),
        )

    if movie_id not in movie_metadata.index:
        raise HTTPException(status_code=404, detail=f"movie_id={movie_id} 에 해당하는 영화를 찾을 수 없습니다.")

    algorithm, recommendations = recommend_similar_movies(movie_id, top_n)

    if not recommendations:
        raise HTTPException(status_code=404, detail="추천할 유사 영화를 찾지 못했습니다.")

    return {
        "selected_movie": serialize_movie(movie_id),
        "algorithm": algorithm,
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
