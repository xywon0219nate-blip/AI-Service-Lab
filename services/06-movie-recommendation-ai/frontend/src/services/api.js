// services/api.js
// ---------------------------------------------------------------
// FastAPI 서버와 통신하는 유일한 창구.
// API 주소를 컴포넌트마다 하드코딩하지 않고 이 파일 하나만 거치도록 한다.
//
// - 기본값(baseURL="")은 상대 경로 요청을 의미하며, vite.config.js의 proxy 설정을 통해
//   개발 서버 / Docker Compose 환경에서 자동으로 backend(8000)로 전달된다.
// - AWS EC2 등 프록시가 없는 환경에서는 .env 파일에 VITE_API_BASE_URL을 지정해
//   FastAPI 주소를 직접 가리키도록 오버라이드할 수 있다.
//   예) VITE_API_BASE_URL=http://<EC2-IP>:8000
// ---------------------------------------------------------------

import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export async function getModelInfo() {
  const { data } = await api.get("/model-info");
  return data;
}

export async function searchMovies(search, limit = 20) {
  const { data } = await api.get("/movies", { params: { search, limit } });
  return data;
}

export async function recommendMovies(payload) {
  const { data } = await api.post("/recommend", payload);
  return data;
}

export default api;
