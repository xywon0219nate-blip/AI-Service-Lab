import { useState } from "react";
import axios from "axios";
import "./App.css";

// FastAPI 서버 주소
// ai-server(main.py)가 CORS를 모두 허용(allow_origins=["*"])하도록 설정되어 있어서
// 별도의 Vite 프록시 없이 아래 주소로 바로 요청을 보낼 수 있습니다.
const API_URL = "http://3.34.1.44:8000/predict";

// -----------------------------------------------------------
// 입력폼에서 사용할 항목 목록
// key: 서버(BikeFeatures)로 보낼 데이터의 이름과 동일해야 합니다.
// label: 화면에 보여줄 이름
// type: "number"(숫자 입력) 또는 "select"(선택형)
// step: number input의 증가 단위 (정수형은 1, 소수형은 0.1)
// -----------------------------------------------------------
const FIELDS = [
  { key: "hour", label: "시간 (Hour, 0~23)", type: "number", step: "1" },
  {
    key: "temperature",
    label: "기온 (Temperature, °C)",
    type: "number",
    step: "0.1",
  },
  { key: "humidity", label: "습도 (Humidity, %)", type: "number", step: "1" },
  {
    key: "wind_speed",
    label: "풍속 (Wind Speed, m/s)",
    type: "number",
    step: "0.1",
  },
  {
    key: "visibility",
    label: "가시거리 (Visibility, 10m)",
    type: "number",
    step: "1",
  },
  {
    key: "dew_point",
    label: "이슬점 (Dew Point, °C)",
    type: "number",
    step: "0.1",
  },
  {
    key: "solar_radiation",
    label: "일사량 (Solar Radiation, MJ/m2)",
    type: "number",
    step: "0.01",
  },
  {
    key: "rainfall",
    label: "강수량 (Rainfall, mm)",
    type: "number",
    step: "0.1",
  },
  {
    key: "snowfall",
    label: "적설량 (Snowfall, cm)",
    type: "number",
    step: "0.1",
  },
  {
    key: "season",
    label: "계절 (Season)",
    type: "select",
    options: ["Spring", "Summer", "Autumn", "Winter"],
  },
  {
    key: "holiday",
    label: "공휴일 여부 (Holiday)",
    type: "select",
    options: ["No Holiday", "Holiday"],
  },
  {
    key: "functioning_day",
    label: "운영일 여부 (Functioning Day)",
    type: "select",
    options: ["Yes", "No"],
  },
];

// 입력폼의 초기값 (선택형은 첫 번째 옵션, 숫자형은 빈 문자열)
const INITIAL_FORM = FIELDS.reduce((acc, field) => {
  acc[field.key] = field.type === "select" ? field.options[0] : "";
  return acc;
}, {});

export default function App() {
  // 입력폼 값들을 저장하는 state
  const [form, setForm] = useState(INITIAL_FORM);

  // 예측 결과를 저장하는 state ({ prediction, unit } 형태)
  const [result, setResult] = useState(null);

  // 에러 메시지를 저장하는 state
  const [error, setError] = useState("");

  // API 요청 중인지 여부를 저장하는 state
  const [loading, setLoading] = useState(false);

  // 입력값이 바뀔 때마다 실행되는 함수
  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  // "Predict" 버튼을 눌렀을 때 실행되는 함수
  async function handleSubmit(e) {
    e.preventDefault();

    // 이전 결과와 에러 메시지 초기화
    setResult(null);
    setError("");
    setLoading(true);

    try {
      // 입력값(문자열)을 FastAPI가 원하는 타입(숫자/문자)으로 변환
      const requestBody = {};
      FIELDS.forEach((field) => {
        const value = form[field.key];
        requestBody[field.key] =
          field.type === "number" ? Number(value) : value;
      });

      // axios로 FastAPI 서버에 POST 요청 보내기
      // axios는 fetch와 달리 응답 상태가 실패(4xx/5xx)면 자동으로 에러를 던져줍니다.
      const response = await axios.post(API_URL, requestBody);

      setResult(response.data);
    } catch {
      // 요청 자체가 실패하거나(서버가 꺼져있는 경우) 응답이 실패한 경우
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
        <h1>Seoul Bike Demand AI</h1>
        <p className="description">
          React + FastAPI 기반
          <br />
          서울시 공공자전거 수요 예측 서비스
        </p>

        {/* 입력폼 영역 */}
        <form className="predict-form" onSubmit={handleSubmit}>
          <div className="input-grid">
            {FIELDS.map((field) => (
              <label key={field.key} className="input-item">
                <span>{field.label}</span>
                {field.type === "select" ? (
                  <select
                    name={field.key}
                    value={form[field.key]}
                    onChange={handleChange}
                  >
                    {field.options.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step={field.step}
                    name={field.key}
                    value={form[field.key]}
                    onChange={handleChange}
                    required
                  />
                )}
              </label>
            ))}
          </div>

          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? "예측 중..." : "Predict"}
          </button>
        </form>

        {/* 에러 메시지 영역 */}
        {error && <p className="error-message">{error}</p>}

        {/* 예측 결과 카드 */}
        {result && (
          <div className="result-card">
            <p className="result-label">Prediction</p>
            <p className="result-value">{result.prediction} bikes</p>
          </div>
        )}
      </section>
    </main>
  );
}