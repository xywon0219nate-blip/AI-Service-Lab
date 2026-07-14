import { useState } from "react";
import "./App.css";

// FastAPI 서버 주소
// 개발 중에는 vite.config.js의 proxy 설정을 통해
// http://127.0.0.1:8000/predict 로 전달됩니다.
const API_URL = "/predict";

// 입력폼에서 사용할 항목 목록
// key: 서버로 보낼 데이터의 이름
// label: 화면에 보여줄 이름
const FIELDS = [
  { key: "MedInc", label: "MedInc (평균 소득)" },
  { key: "HouseAge", label: "HouseAge (주택 연식)" },
  { key: "AveRooms", label: "AveRooms (평균 방 개수)" },
  { key: "AveBedrms", label: "AveBedrms (평균 침실 개수)" },
  { key: "Population", label: "Population (인구 수)" },
  { key: "AveOccup", label: "AveOccup (평균 거주 인원)" },
  { key: "Latitude", label: "Latitude (위도)" },
  { key: "Longitude", label: "Longitude (경도)" },
];

// 입력폼 초기값 (모두 빈 문자열로 시작)
const INITIAL_FORM = FIELDS.reduce((acc, field) => {
  acc[field.key] = "";
  return acc;
}, {});

export default function App() {
  // 입력폼 값들을 저장하는 state
  const [form, setForm] = useState(INITIAL_FORM);

  // 예측 결과를 저장하는 state
  const [result, setResult] = useState(null);

  // 에러 메시지를 저장하는 state
  const [error, setError] = useState("");

  // API 호출 중인지 여부를 저장하는 state
  const [loading, setLoading] = useState(false);

  // 입력값이 바뀔 때마다 실행되는 함수
  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  // "예측하기" 버튼을 눌렀을 때 실행되는 함수
  async function handleSubmit(e) {
    e.preventDefault();

    // 이전 결과와 에러 메시지 초기화
    setResult(null);
    setError("");
    setLoading(true);

    try {
      // 입력값(문자열)을 숫자로 변환해서 서버로 보낼 데이터 만들기
      const requestBody = {};
      FIELDS.forEach((field) => {
        requestBody[field.key] = Number(form[field.key]);
      });

      // FastAPI 서버로 예측 요청 보내기
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      // 서버가 에러 응답을 준 경우
      if (!response.ok) {
        throw new Error("서버 응답 오류");
      }

      const data = await response.json();
      setResult(data.predicted_price);
    } catch {
      // fetch 자체가 실패하거나(서버가 꺼져있는 경우) 응답이 실패한 경우
      setError("FastAPI 서버에 연결할 수 없습니다.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="dashboard">
      <section className="hero-card">
        {/* Header 영역 */}
        <p className="badge">AI Service Blueprint</p>
        <h1>House Price AI</h1>
        <p className="description">
          React + FastAPI 기반
          <br />
          부동산 가격 예측 서비스
        </p>

        {/* 입력폼 영역 */}
        <form className="predict-form" onSubmit={handleSubmit}>
          <div className="input-grid">
            {FIELDS.map((field) => (
              <label key={field.key} className="input-item">
                <span>{field.label}</span>
                <input
                  type="number"
                  step="any"
                  name={field.key}
                  value={form[field.key]}
                  onChange={handleChange}
                  required
                />
              </label>
            ))}
          </div>

          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? "예측 중..." : "예측하기"}
          </button>
        </form>

        {/* 에러 메시지 영역 */}
        {error && <p className="error-message">{error}</p>}

        {/* 예측 결과 카드 */}
        {result !== null && (
          <div className="result-card">
            <p className="result-label">예측 집값</p>
            <p className="result-value">{result}</p>
          </div>
        )}
      </section>
    </main>
  );
}