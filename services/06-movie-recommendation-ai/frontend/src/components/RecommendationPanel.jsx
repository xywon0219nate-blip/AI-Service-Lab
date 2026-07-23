// components/RecommendationPanel.jsx
// 추천 결과 영역 (초기 안내 / Loading / 오류 / 추천 결과 카드 목록)

export default function RecommendationPanel({ result, error, loading, onReanalyze }) {
  if (loading) {
    return (
      <div className="result-panel state-loading">
        <div className="spinner" />
        <p>비슷한 영화를 찾고 있습니다...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-panel state-error">
        <p className="state-title">추천에 실패했습니다</p>
        <p className="state-message">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-panel state-empty">
        <p className="state-title">추천받고 싶은 영화를 선택해주세요</p>
        <p className="state-message">
          왼쪽에서 영화를 검색하고 선택한 뒤 &quot;추천받기&quot; 버튼을 누르면
          비슷한 영화 목록과 추천 이유를 확인할 수 있습니다.
        </p>
      </div>
    );
  }

  const { selected_movie: selectedMovie, recommendations, algorithm } = result;

  return (
    <div className="result-panel state-result">
      <div className="selected-summary">
        <p className="selected-summary-label">선택한 영화</p>
        <p className="selected-summary-title">
          {selectedMovie.clean_title}
          {selectedMovie.release_year ? ` (${selectedMovie.release_year})` : ""}
        </p>
        <span className="algorithm-badge">
          {algorithm === "item_based_cf" ? "협업 필터링 기반 추천" : "콘텐츠(장르) 기반 추천"}
        </span>
      </div>

      <div className="recommendation-list">
        {recommendations.map((movie, index) => {
          const percent = Math.round(movie.similarity_score * 1000) / 10;
          return (
            <div className="recommendation-card" key={movie.movie_id}>
              <div className="poster-placeholder">
                <span>{index + 1}</span>
              </div>
              <div className="recommendation-body">
                <p className="recommendation-title">
                  {movie.title}
                  {movie.release_year ? ` (${movie.release_year})` : ""}
                </p>
                <p className="recommendation-genres">{movie.genres?.join(", ")}</p>
                <p className="recommendation-reason">{movie.recommendation_reason}</p>

                <div className="recommendation-meta">
                  <span>유사도 {percent}%</span>
                  <span>⭐ {movie.average_rating}</span>
                  <span>평가 {movie.rating_count}건</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <button type="button" className="secondary-button reanalyze-button" onClick={onReanalyze}>
        다른 영화로 다시 추천받기
      </button>
    </div>
  );
}
