# train_model_basic.py
# ---------------------------------------------------------------
# Movie Recommendation AI - Basic 모델 학습
#
# ai-server/fe/ST_movie_recommendation_basic.ipynb 에서 실험한 장르 콘텐츠 기반
# (Content-Based) 추천 방식을 그대로 운영용 산출물로 저장한다.
#
#   - 영화 장르(One-Hot)만을 이용한 코사인 유사도 추천
#   - 평점 수가 MIN_RATING_COUNT 미만인 영화는 추천 후보에서 제외한다
#     (노트북 10번 섹션 "최소 평점 수 적용" 참고)
#
# train_model.py(협업 필터링 + 콘텐츠 기반 폴백)와 달리, 유사도 행렬 자체는
# 저장하지 않는다. 전체 영화(약 1만 편) 간 NxN 코사인 유사도 행렬은 수백 MB에
# 달해 저장/로딩 비용이 크므로, 장르 Feature 행렬만 저장하고 추천 시점에
# 질의한 영화 1개에 대해서만 코사인 유사도를 계산한다
# (main.py의 Cold Start 콘텐츠 기반 폴백과 동일한 방식).
#
# 실행:
#   python train_model_basic.py
# ---------------------------------------------------------------

import json
import os
import sys

import joblib
import pandas as pd


# ---------------------------------------------------------------
# 0. 경로 / 상수
# ---------------------------------------------------------------

DATA_DIR = "data"
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

MODEL_DIR = "models_basic"
METADATA_PATH = os.path.join(MODEL_DIR, "movie_metadata.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "movie_features.pkl")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# 평점 수가 이 값 미만인 영화는 장르만 비슷할 뿐 신뢰하기 어려워 추천에서 제외한다.
MIN_RATING_COUNT = 10
DEFAULT_TOP_N = 10


# ---------------------------------------------------------------
# 1. 학습 메인 로직
# ---------------------------------------------------------------

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not (os.path.exists(MOVIES_PATH) and os.path.exists(RATINGS_PATH)):
        print("[오류] movies.csv / ratings.csv 를 찾을 수 없습니다.")
        print(f"       {DATA_DIR}/ 폴더에 movies.csv, ratings.csv 를 추가한 뒤 다시 실행하세요.")
        sys.exit(1)

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    print(f"movies: {movies.shape}, ratings: {ratings.shape}")

    # -----------------------------------------------------------
    # 1) 영화별 평점 통계 (평점 수 / 평균 평점)
    # -----------------------------------------------------------
    rating_stats = ratings.groupby("movieId")["rating"].agg(rating_count="count", rating_mean="mean")

    movie_metadata = movies.set_index("movieId").join(rating_stats, how="left")
    movie_metadata["rating_count"] = movie_metadata["rating_count"].fillna(0).astype(int)
    movie_metadata["rating_mean"] = movie_metadata["rating_mean"].fillna(0.0)

    eligible_count = (movie_metadata["rating_count"] >= MIN_RATING_COUNT).sum()
    print(f"평점 {MIN_RATING_COUNT}건 이상인 영화: {eligible_count:,} / {movie_metadata.shape[0]:,}편")

    # -----------------------------------------------------------
    # 2) 장르 One-Hot Feature (콘텐츠 기반 추천용)
    # -----------------------------------------------------------
    movie_features = movie_metadata["genres"].str.get_dummies(sep="|")
    print(f"장르 Feature 크기: {movie_features.shape}")

    # -----------------------------------------------------------
    # 3) 산출물 저장
    # -----------------------------------------------------------
    joblib.dump(movie_metadata, METADATA_PATH)
    joblib.dump(movie_features, FEATURES_PATH)
    print(f"저장 완료: {METADATA_PATH}")
    print(f"저장 완료: {FEATURES_PATH}")

    model_info = {
        "project": "06-movie-recommendation-ai",
        "service_name": "Movie Recommendation AI (Basic)",
        "dataset": "MovieLens (movies.csv / ratings.csv)",
        "model_name": "Content-Based Genre Cosine Similarity",
        "model_version": "1.0.0",
        "primary_algorithm": "content_based_genre_cosine_similarity",
        "min_rating_count": MIN_RATING_COUNT,
        "default_top_n": DEFAULT_TOP_N,
        "num_movies": int(movie_metadata.shape[0]),
        "num_ratings": int(ratings.shape[0]),
        "num_genre_features": int(movie_features.shape[1]),
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {MODEL_INFO_PATH}")

    print("\n=== 학습 완료 ===")
    print(json.dumps(model_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
