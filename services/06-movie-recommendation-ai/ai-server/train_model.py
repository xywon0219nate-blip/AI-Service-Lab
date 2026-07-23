# train_model.py
# ---------------------------------------------------------------
# Movie Recommendation AI - MovieLens 100K 기반 영화 추천 모델 학습
#
# Notebook(ai-server/fe/movie_recommendation_analysis.ipynb)에서 비교한 결과,
# 최종 추천 방식으로 다음 두 가지를 함께 사용하기로 결정했다.
#
#   1) Item-Based Collaborative Filtering (기본)
#      - 평점 데이터가 충분한 영화(rating_count >= MIN_RATING_COUNT_FOR_CF)에 대해
#        사용자 평점 패턴 기반 코사인 유사도로 추천한다.
#   2) Content-Based Recommendation (Cold Start 대체)
#      - 평점 수가 적어 협업 필터링 신호가 부족한 영화는 장르 기반 코사인 유사도로
#        대체 추천한다.
#
# 이 스크립트는 Notebook의 실험 코드를 그대로 복사하지 않고, 운영(FastAPI)에 필요한
# 최종 산출물만 만든다. 알고리즘 간 비교/평가는 Notebook에서만 수행한다.
#
# 실행:
#   python train_model.py
# ---------------------------------------------------------------

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering import (
    DEFAULT_TOP_N,
    MIN_RATING_COUNT_FOR_CF,
    build_genre_matrix,
    build_movie_metadata,
    build_user_item_matrix,
    compute_bayesian_popularity,
    find_data_dir,
    load_movies,
    load_ratings,
    load_users,
    mean_center_user_item_matrix,
    print_missing_data_guide,
)


# ---------------------------------------------------------------
# 0. 경로 상수
# ---------------------------------------------------------------

DATA_DIR = "data"
MODEL_DIR = "models"
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity_matrix.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "movie_metadata.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "movie_features.pkl")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")


# ---------------------------------------------------------------
# 1. 데이터 품질 점검 및 정리
# ---------------------------------------------------------------

def report_data_quality(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> None:
    print("-" * 70)
    print("데이터 품질 점검")
    print("-" * 70)
    print(f"ratings: {ratings.shape}, movies: {movies.shape}, users: {users.shape}")

    print("ratings 결측치:", dict(ratings.isnull().sum()[ratings.isnull().sum() > 0]) or "없음")
    print("(user_id, movie_id) 중복 평가:", ratings.duplicated(subset=["user_id", "movie_id"]).sum())

    duplicate_titles = movies["title"].duplicated().sum()
    print(f"제목이 중복된 영화: {duplicate_titles}건 "
          f"(서로 다른 movie_id에 동일 제목이 존재하는 MovieLens 100K의 알려진 특성. "
          f"별도 병합 없이 그대로 유지한다)")

    out_of_range = ((ratings["rating"] < 1) | (ratings["rating"] > 5)).sum()
    print("평점 범위(1~5) 벗어난 값:", out_of_range)
    print("-" * 70)


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    before = len(ratings)
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]
    after = len(ratings)
    if before != after:
        print(f"정리 과정에서 {before - after:,}건의 행을 제거했습니다 (중복 평가/범위 밖 평점).")
    return ratings.reset_index(drop=True)


# ---------------------------------------------------------------
# 2. Item-Based Collaborative Filtering 유사도 행렬
# ---------------------------------------------------------------

def build_item_similarity_matrix(ratings: pd.DataFrame, min_rating_count: int) -> pd.DataFrame:
    """평점 수가 min_rating_count 이상인 영화만 대상으로 아이템 기반 협업 필터링
    유사도 행렬을 계산한다.

    - 사용자마다 평점 성향이 다르므로 사용자 평균으로 중심화한 뒤 코사인 유사도를 계산한다.
    - 평가하지 않은 칸은 0으로 채우는데, 이는 "실제로 중간 평점을 준 것"과는 다르다는
      한계가 있다 (자세한 설명은 Research_Report.md 11.6절 참고).
    """
    rating_count = ratings.groupby("movie_id")["rating"].count()
    eligible_movie_ids = rating_count[rating_count >= min_rating_count].index

    user_item_matrix = build_user_item_matrix(ratings)
    user_item_matrix = user_item_matrix[[c for c in user_item_matrix.columns if c in eligible_movie_ids]]

    centered = mean_center_user_item_matrix(user_item_matrix)

    # 행: 영화, 열: 사용자 순서로 바꿔 영화 간 유사도를 계산한다.
    movie_vectors = centered.T
    similarity = cosine_similarity(movie_vectors)

    return pd.DataFrame(similarity, index=movie_vectors.index, columns=movie_vectors.index)


# ---------------------------------------------------------------
# 3. 학습 메인 로직
# ---------------------------------------------------------------

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_dir = find_data_dir(DATA_DIR)
    if data_dir is None:
        print_missing_data_guide(DATA_DIR)
        sys.exit(1)

    print(f"데이터 경로: {data_dir}")

    # -----------------------------------------------------------
    # 1) 데이터 로딩 및 정리
    # -----------------------------------------------------------
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users = load_users(data_dir)

    report_data_quality(ratings, movies, users)
    ratings = clean_ratings(ratings)

    print(f"사용자 수: {users.shape[0]:,} / 영화 수: {movies.shape[0]:,} / 평점 수: {ratings.shape[0]:,}")

    # -----------------------------------------------------------
    # 2) Feature Engineering
    # -----------------------------------------------------------
    movie_metadata = build_movie_metadata(movies, ratings)
    genre_matrix = build_genre_matrix(movies)

    popularity_score, m, C = compute_bayesian_popularity(
        movie_metadata["rating_count"], movie_metadata["rating_mean"]
    )
    movie_metadata["popularity_score"] = popularity_score
    print(f"인기도(Bayesian Weighted Rating) 기준: m={m:.2f}건, 전체 평균 C={C:.4f}")

    eligible_count = (movie_metadata["rating_count"] >= MIN_RATING_COUNT_FOR_CF).sum()
    print(f"협업 필터링 대상 영화(평점 {MIN_RATING_COUNT_FOR_CF}건 이상): "
          f"{eligible_count:,} / {movie_metadata.shape[0]:,}편")

    # -----------------------------------------------------------
    # 3) Item-Based Collaborative Filtering 유사도 행렬 생성
    # -----------------------------------------------------------
    similarity_matrix = build_item_similarity_matrix(ratings, MIN_RATING_COUNT_FOR_CF)
    print(f"협업 필터링 유사도 행렬 크기: {similarity_matrix.shape}")

    # -----------------------------------------------------------
    # 4) 산출물 저장
    # -----------------------------------------------------------
    joblib.dump(similarity_matrix, SIMILARITY_PATH)
    joblib.dump(movie_metadata, METADATA_PATH)
    joblib.dump(genre_matrix, FEATURES_PATH)
    print(f"저장 완료: {SIMILARITY_PATH}")
    print(f"저장 완료: {METADATA_PATH}")
    print(f"저장 완료: {FEATURES_PATH}")

    model_info = {
        "project": "05-movie-recommendation-ai",
        "service_name": "Movie Recommendation AI",
        "dataset": "MovieLens 100K",
        "model_name": "Item-Based Collaborative Filtering + Content-Based(Genre) Fallback",
        "model_version": "1.0.0",
        "primary_algorithm": "item_based_collaborative_filtering",
        "fallback_algorithm": "content_based_genre_similarity",
        "min_rating_count_for_cf": MIN_RATING_COUNT_FOR_CF,
        "popularity_min_votes_m": round(m, 4),
        "popularity_global_mean_C": round(C, 4),
        "default_top_n": DEFAULT_TOP_N,
        "num_users": int(users.shape[0]),
        "num_movies": int(movies.shape[0]),
        "num_ratings": int(ratings.shape[0]),
        "num_movies_in_cf_matrix": int(similarity_matrix.shape[0]),
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {MODEL_INFO_PATH}")

    print("\n=== 학습 완료 ===")
    print(json.dumps(model_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
