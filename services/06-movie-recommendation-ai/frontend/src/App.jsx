import { useEffect, useState } from "react";
import "./App.css";

import MovieSearch from "./components/MovieSearch";
import RecommendationPanel from "./components/RecommendationPanel";
import { getModelInfo, recommendMovies, searchMovies } from "./services/api";

const SEARCH_DEBOUNCE_MS = 300;

function extractErrorMessage(error) {
  const response = error?.response;

  if (!response) {
    return "FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.";
  }

  if (response.status === 503) {
    return response.data?.detail ?? "추천 모델이 아직 준비되지 않았습니다.";
  }

  if (response.status === 404) {
    return response.data?.detail ?? "해당 영화를 찾을 수 없습니다.";
  }

  if (response.status === 422) {
    const details = response.data?.details;
    if (Array.isArray(details) && details.length > 0) {
      return details.map((item) => item.msg).join("\n");
    }
    return response.data?.error ?? "입력값을 확인해주세요.";
  }

  return response.data?.detail ?? "알 수 없는 오류가 발생했습니다.";
}

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);

  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);

  const [selectedMovie, setSelectedMovie] = useState(null);
  const [topN, setTopN] = useState(10);

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null));
  }, []);

  useEffect(() => {
    const keyword = query.trim();
    if (!keyword) {
      return;
    }

    const timer = setTimeout(() => {
      searchMovies(keyword, 20)
        .then(setSearchResults)
        .catch(() => setSearchResults([]))
        .finally(() => setSearching(false));
    }, SEARCH_DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [query]);

  function handleQueryChange(value) {
    setQuery(value);
    if (value.trim()) {
      setSearching(true);
    } else {
      setSearchResults([]);
      setSearching(false);
    }
  }

  function handleSelectMovie(movie) {
    setSelectedMovie(movie);
    setResult(null);
    setError("");
  }

  function handleReset() {
    setQuery("");
    setSearchResults([]);
    setSelectedMovie(null);
    setTopN(10);
    setResult(null);
    setError("");
  }

  async function handleSubmit() {
    if (!selectedMovie) return;

    setResult(null);
    setError("");
    setLoading(true);

    try {
      const data = await recommendMovies({ movie_id: selectedMovie.movie_id, top_n: topN });
      setResult(data);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  function handleReanalyze() {
    setResult(null);
    setError("");
  }

  return (
    <main className="page">
      <header className="page-header">
        <p className="badge">AI Service Blueprint</p>
        <h1>Movie Recommendation AI</h1>
        <p className="subtitle">좋아하는 영화를 선택하고 비슷한 영화를 추천받아 보세요.</p>
        {modelInfo?.model_loaded === false && (
          <p className="model-warning">
            아직 학습된 추천 모델이 없습니다. data 폴더에 MovieLens 100K 데이터를 추가한 뒤
            모델을 학습하면 추천 기능을 사용할 수 있습니다.
          </p>
        )}
      </header>

      <section className="dashboard">
        <div className="panel input-panel">
          <h2>영화 검색 및 선택</h2>
          <MovieSearch
            query={query}
            onQueryChange={handleQueryChange}
            searchResults={searchResults}
            searching={searching}
            selectedMovie={selectedMovie}
            onSelectMovie={handleSelectMovie}
            topN={topN}
            onTopNChange={setTopN}
            onSubmit={handleSubmit}
            onReset={handleReset}
            loading={loading}
          />
        </div>

        <div className="panel result-panel-wrapper">
          <h2>추천 결과</h2>
          <RecommendationPanel
            result={result}
            error={error}
            loading={loading}
            onReanalyze={handleReanalyze}
          />
        </div>
      </section>

      <footer className="page-footer">
        추천 결과는 MovieLens 100K 데이터 기반의 통계적 유사도이며, 실제 취향과 다를 수 있습니다.
      </footer>
    </main>
  );
}
