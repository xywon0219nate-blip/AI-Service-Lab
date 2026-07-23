// components/MovieSearch.jsx
// 영화 검색 + 선택 + 추천 개수 선택 + 추천받기 버튼

const TOP_N_OPTIONS = [5, 10, 15, 20];

export default function MovieSearch({
  query,
  onQueryChange,
  searchResults,
  searching,
  selectedMovie,
  onSelectMovie,
  topN,
  onTopNChange,
  onSubmit,
  onReset,
  loading,
}) {
  return (
    <div className="movie-search">
      <div className="field">
        <label htmlFor="movie-query">영화 제목 검색</label>
        <input
          id="movie-query"
          type="text"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="예: Toy Story, Star Wars..."
          autoComplete="off"
        />
      </div>

      {query.trim().length > 0 && (
        <div className="search-results">
          {searching && <p className="search-status">검색 중...</p>}
          {!searching && searchResults.length === 0 && (
            <p className="search-status">검색 결과가 없습니다.</p>
          )}
          {!searching &&
            searchResults.map((movie) => (
              <button
                type="button"
                key={movie.movie_id}
                className={`search-result-item ${
                  selectedMovie?.movie_id === movie.movie_id ? "active" : ""
                }`}
                onClick={() => onSelectMovie(movie)}
              >
                <span className="result-title">
                  {movie.clean_title}
                  {movie.release_year ? ` (${movie.release_year})` : ""}
                </span>
                <span className="result-meta">
                  ⭐ {movie.average_rating} · 평가 {movie.rating_count}건
                </span>
              </button>
            ))}
        </div>
      )}

      <div className="selected-movie">
        {selectedMovie ? (
          <div className="selected-movie-card">
            <p className="selected-label">선택한 영화</p>
            <p className="selected-title">
              {selectedMovie.clean_title}
              {selectedMovie.release_year ? ` (${selectedMovie.release_year})` : ""}
            </p>
            <p className="selected-genres">{selectedMovie.genres?.join(", ")}</p>
          </div>
        ) : (
          <p className="selected-empty">추천받고 싶은 영화를 검색하여 선택해 주세요.</p>
        )}
      </div>

      <div className="field">
        <label htmlFor="top-n">추천 개수</label>
        <select id="top-n" value={topN} onChange={(e) => onTopNChange(Number(e.target.value))}>
          {TOP_N_OPTIONS.map((n) => (
            <option key={n} value={n}>
              {n}개
            </option>
          ))}
        </select>
      </div>

      <div className="action-buttons">
        <button type="button" className="secondary-button" onClick={onReset}>
          초기화
        </button>
        <button
          type="button"
          className="primary-button"
          onClick={onSubmit}
          disabled={!selectedMovie || loading}
        >
          {loading ? "추천 분석 중..." : "추천받기"}
        </button>
      </div>
    </div>
  );
}
