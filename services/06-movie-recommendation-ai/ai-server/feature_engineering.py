# feature_engineering.py
# ---------------------------------------------------------------
# Movie Recommendation AI - 공용 데이터 로딩 / Feature Engineering 모듈
#
# Notebook(EDA/알고리즘 비교)과 운영 코드(train_model.py, main.py)가 완전히
# 동일한 로딩·정제·파생변수 로직을 사용해야 하므로, 데이터 처리와 관련된 함수를
# 이 모듈 하나에 모아두고 양쪽에서 그대로 가져다 쓴다.
# ---------------------------------------------------------------

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# 1. 데이터 경로 / 원본 컬럼 상수
# ---------------------------------------------------------------

# data/ 폴더에서 이 순서로 MovieLens 100K 폴더를 탐색한다.
CANDIDATE_DATA_DIRS = ["ml-100k", "ML-100k", "movielens-100k"]

REQUIRED_FILES = ["u.data", "u.item", "u.user"]

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

ITEM_COLUMNS = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS
USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]

TARGET_COLUMN = "rating"

# 영화별 최소 평점 수 기준. EDA 결과(rating_count 중앙값 27, 전체 1682편 중
# 20건 미만인 영화가 743편) 를 근거로, 협업 필터링 유사도 계산에 포함할
# "충분히 평가된 영화"의 기준을 20건으로 결정한다. (자세한 근거는
# Research_Report.md 11.5 절 참고)
MIN_RATING_COUNT_FOR_CF = 20

# 인기도(Bayesian Weighted Rating) 계산 시 사용하는 최소 투표 수 m은
# train_model.py에서 rating_count의 60번째 백분위수로 데이터 기반 결정한다.
POPULARITY_QUANTILE = 0.60

DEFAULT_TOP_N = 10
TOP_N_MIN = 1
TOP_N_MAX = 30


# ---------------------------------------------------------------
# 2. 데이터 파일 탐색 / 로딩
# ---------------------------------------------------------------

def find_data_dir(base_dir: str = "data") -> Optional[Path]:
    """data/ 폴더 아래에서 MovieLens 100K 원본 파일이 있는 폴더를 찾는다.
    없으면 None을 반환한다 (자동 다운로드는 시도하지 않는다).
    """
    base = Path(base_dir)

    for candidate in CANDIDATE_DATA_DIRS:
        path = base / candidate
        if all((path / f).exists() for f in REQUIRED_FILES):
            return path

    # data/ 바로 아래에 압축을 풀어둔 경우도 지원한다.
    if all((base / f).exists() for f in REQUIRED_FILES):
        return base

    return None


def print_missing_data_guide(base_dir: str = "data") -> None:
    print("=" * 70)
    print("[오류] MovieLens 100K 데이터를 찾을 수 없습니다.")
    print("=" * 70)
    print("다음 경로에 MovieLens 100K 압축을 해제한 뒤 다시 실행하세요.")
    print(f"  - {Path(base_dir) / 'ml-100k'} /u.data, u.item, u.user 등")
    print()
    print("자세한 안내는 data/README.md 를 참고하세요.")
    print("=" * 70)


def load_ratings(data_dir: Path) -> pd.DataFrame:
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=RATING_COLUMNS,
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"},
        engine="python",
    )
    return ratings


def load_movies(data_dir: Path) -> pd.DataFrame:
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=ITEM_COLUMNS,
        encoding="latin-1",
        engine="python",
    )
    # video_release_date는 MovieLens 100K 전체에서 항상 결측이라 제거한다.
    return movies.drop(columns=["video_release_date"])


def load_users(data_dir: Path) -> pd.DataFrame:
    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=USER_COLUMNS,
        dtype={"user_id": "int32", "age": "int32"},
        engine="python",
    )
    return users


# ---------------------------------------------------------------
# 3. 영화 제목 정리 (개봉연도 분리)
# ---------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"\((\d{4})\)")


def split_title_and_year(raw_title: str) -> tuple:
    """"Toy Story (1995)" -> ("Toy Story", 1995)로 분리한다.

    "Land Before Time III: The Time of the Great Giving (1995) (V)" 처럼
    연도 뒤에 "(V)" 같은 부가 표기가 더 붙는 경우도 있어, 문자열 마지막이 아니라
    "(YYYY)" 패턴이 마지막으로 등장하는 위치를 기준으로 자른다.
    "unknown"처럼 연도 표기가 아예 없는 경우 release_year는 None이 된다.
    """
    title = str(raw_title).strip()
    matches = list(_YEAR_PATTERN.finditer(title))
    if not matches:
        return title, None

    last_match = matches[-1]
    year = int(last_match.group(1))
    clean = title[: last_match.start()].strip()
    return clean, year


def build_movie_metadata(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """u.item + u.data를 합쳐 서비스에 필요한 영화 메타데이터 테이블을 만든다.

    포함 컬럼: movie_id, title(원본), clean_title, release_year, search_title,
    genres(리스트), rating_count, rating_mean
    """
    movies = movies.copy()

    parsed = movies["title"].apply(split_title_and_year)
    movies["clean_title"] = [p[0] for p in parsed]
    movies["release_year"] = [p[1] for p in parsed]
    movies["search_title"] = movies["clean_title"].str.lower()

    genre_matrix = movies[GENRE_COLUMNS]
    movies["genres"] = genre_matrix.apply(
        lambda row: [genre for genre, flag in zip(GENRE_COLUMNS, row) if flag == 1], axis=1
    )

    rating_stats = ratings.groupby("movie_id")["rating"].agg(
        rating_count="count", rating_mean="mean"
    )

    metadata = movies.set_index("movie_id").join(rating_stats, how="left")
    metadata["rating_count"] = metadata["rating_count"].fillna(0).astype(int)
    metadata["rating_mean"] = metadata["rating_mean"].fillna(0.0)

    columns = [
        "title", "clean_title", "release_year", "search_title",
        "genres", "rating_count", "rating_mean",
    ]
    return metadata[columns]


def build_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """콘텐츠 기반 추천(장르 코사인 유사도)에 쓰이는 movie_id x genre 멀티-핫 행렬."""
    genre_matrix = movies.set_index("movie_id")[GENRE_COLUMNS].astype(float)
    return genre_matrix


# ---------------------------------------------------------------
# 4. 사용자-영화 평점 행렬 (협업 필터링용)
# ---------------------------------------------------------------

def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """행=user_id, 열=movie_id, 값=rating 인 평점 행렬. 평가하지 않은 칸은 NaN이다.

    NaN을 바로 0으로 채우지 않는 이유: 0은 "1점보다 낮은 평점"으로 오인될 수 있어,
    사용자별 평균으로 중심화(centering)한 뒤에 남는 결측만 0으로 채운다.
    """
    return ratings.pivot_table(index="user_id", columns="movie_id", values="rating")


def mean_center_user_item_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """사용자마다 후한/짠 평점 성향이 다르므로, 사용자 평균을 빼서 중심화한다.
    평가하지 않은 칸은 0으로 채워 "차이 없음"으로 취급한다.
    """
    user_mean = user_item_matrix.mean(axis=1)
    centered = user_item_matrix.sub(user_mean, axis=0)
    return centered.fillna(0.0)


def compute_bayesian_popularity(rating_count: pd.Series, rating_mean: pd.Series, quantile: float = POPULARITY_QUANTILE):
    """Bayesian Weighted Rating (IMDB 공식과 동일한 형태).

    weighted = (v / (v + m)) * R + (m / (v + m)) * C
      v: 해당 영화의 평점 수, R: 해당 영화의 평균 평점
      m: 최소 투표 수 기준(데이터의 rating_count 분포에서 quantile 백분위수로 결정)
      C: 전체 영화 평균 평점

    평점 수가 아주 적은 영화가 평균 평점만으로 최상위에 올라가는 문제를 완화한다.
    """
    m = float(rating_count.quantile(quantile))
    C = float(rating_mean[rating_count > 0].mean())
    v = rating_count.astype(float)
    weighted = (v / (v + m)) * rating_mean + (m / (v + m)) * C
    return weighted, m, C
