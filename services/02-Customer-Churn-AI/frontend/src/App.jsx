import { useState } from "react";
import "./App.css";

// FastAPI 서버 주소
// 개발 중에는 vite.config.js의 proxy 설정을 통해
// http://127.0.0.1:8000/predict 로 전달됩니다.
const API_URL = "/predict";

// -----------------------------------------------------------
// 화면에 실제로 보여줄 "핵심 입력값" 목록
// FastAPI /predict는 총 45개의 값을 요구하지만,
// 화면에는 예측에 큰 영향을 주는 핵심 항목만 보기 좋게 구성합니다.
// key: 서버로 보낼 데이터의 이름(FastAPI가 요구하는 컬럼명과 동일)
// label: 화면에 보여줄 이름
// type: "select"(선택형) 또는 "number"(숫자 입력)
// -----------------------------------------------------------
const FIELDS = [
	{ key: "Gender", label: "성별", type: "select", options: ["Female", "Male"] },
	{ key: "Age", label: "나이", type: "number" },
	{
		key: "Under 30",
		label: "30세 미만 여부",
		type: "select",
		options: ["No", "Yes"],
	},
	{
		key: "Senior Citizen",
		label: "고령자 여부",
		type: "select",
		options: ["No", "Yes"],
	},
	{
		key: "Married",
		label: "결혼 여부",
		type: "select",
		options: ["No", "Yes"],
	},
	{
		key: "Dependents",
		label: "부양가족 여부",
		type: "select",
		options: ["No", "Yes"],
	},
	{ key: "Tenure in Months", label: "이용 기간 (개월)", type: "number" },
	{
		key: "Phone Service",
		label: "전화 서비스 이용",
		type: "select",
		options: ["Yes", "No"],
	},
	{
		key: "Internet Service",
		label: "인터넷 서비스 이용",
		type: "select",
		options: ["Yes", "No"],
	},
	{
		key: "Contract",
		label: "계약 형태",
		type: "select",
		options: ["Month-to-Month", "One Year", "Two Year"],
	},
	{
		key: "Paperless Billing",
		label: "전자 청구서 이용",
		type: "select",
		options: ["Yes", "No"],
	},
	{
		key: "Payment Method",
		label: "결제 방법",
		type: "select",
		options: ["Bank Withdrawal", "Credit Card", "Mailed Check"],
	},
	{ key: "Monthly Charge", label: "월 요금 ($)", type: "number" },
	{ key: "Total Charges", label: "누적 요금 ($)", type: "number" },
	{ key: "Satisfaction Score", label: "만족도 점수 (1~5)", type: "number" },
];

// 화면에 보이지 않는 나머지 입력값들의 기본값
// FastAPI 모델은 학습 당시 사용한 45개 컬럼을 모두 필요로 하므로,
// 화면에서 다루지 않는 값들은 여기서 합리적인 기본값으로 채워 함께 전송합니다.
const HIDDEN_DEFAULTS = {
	"Number of Dependents": 0,
	Country: "United States",
	State: "California",
	City: "Los Angeles",
	"Zip Code": 90001,
	Latitude: 34.0522,
	Longitude: -118.2437,
	Population: 50000,
	Quarter: "Q3",
	"Referred a Friend": "No",
	"Number of Referrals": 0,
	Offer: "Offer E",
	"Avg Monthly Long Distance Charges": 0,
	"Multiple Lines": "No",
	"Internet Type": "DSL",
	"Avg Monthly GB Download": 10,
	"Online Security": "No",
	"Online Backup": "No",
	"Device Protection Plan": "No",
	"Premium Tech Support": "No",
	"Streaming TV": "No",
	"Streaming Movies": "No",
	"Streaming Music": "No",
	"Unlimited Data": "No",
	"Total Refunds": 0,
	"Total Extra Data Charges": 0,
	"Total Long Distance Charges": 0,
	"Customer Status": "Stayed",
	CLTV: 3000,
};

// 화면에 보이는 입력폼의 초기값 (선택형은 첫 번째 옵션, 숫자형은 빈 문자열)
const INITIAL_FORM = FIELDS.reduce((acc, field) => {
	acc[field.key] = field.type === "select" ? field.options[0] : "";
	return acc;
}, {});

export default function App() {
	// 입력폼 값들을 저장하는 state
	const [form, setForm] = useState(INITIAL_FORM);

	// 예측 결과를 저장하는 state ({ prediction, result } 형태)
	const [prediction, setPrediction] = useState(null);

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

	// "예측하기" 버튼을 눌렀을 때 실행되는 함수
	async function handleSubmit(e) {
		e.preventDefault();

		// 이전 결과와 에러 메시지 초기화
		setPrediction(null);
		setError("");
		setLoading(true);

		try {
			// 화면에 보이는 입력값을 FastAPI가 원하는 타입(숫자/문자)으로 변환
			const visibleValues = {};
			FIELDS.forEach((field) => {
				const value = form[field.key];
				visibleValues[field.key] =
					field.type === "number" ? Number(value) : value;
			});

			// 화면에 없는 나머지 값(기본값)과 합쳐서 전체 요청 데이터를 만듭니다.
			const requestBody = {
				...HIDDEN_DEFAULTS,
				...visibleValues,
				// 누적 요금(Total Charges)과 총 수익(Total Revenue)을 동일하게 맞춰줍니다.
				"Total Revenue": visibleValues["Total Charges"],
			};

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
			setPrediction(data);
		} catch {
			// fetch 자체가 실패하거나(서버가 꺼져있는 경우) 응답이 실패한 경우
			setError("FastAPI 서버에 연결할 수 없습니다.");
		} finally {
			setLoading(false);
		}
	}

	// prediction 값(0/1)에 따라 결과 카드에 다른 스타일 클래스를 적용
	const isChurn = prediction?.prediction === 1;

	return (
		<main className="dashboard">
			<section className="hero-card">
				{/* Header 영역 */}
				<p className="badge">AI Service Blueprint</p>
				<h1>Customer Churn AI</h1>
				<p className="description">
					React + FastAPI 기반
					<br />
					고객 이탈 예측 서비스
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
										step="any"
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
						{loading ? "예측 중..." : "예측하기"}
					</button>
				</form>

				{/* 에러 메시지 영역 */}
				{error && <p className="error-message">{error}</p>}

				{/* 예측 결과 카드 */}
				{prediction && (
					<div
						className={`result-card ${isChurn ? "result-churn" : "result-stay"}`}
					>
						<p className="result-label">예측 결과</p>
						<p className="result-value">{prediction.result}</p>
						<p className="result-sub">prediction: {prediction.prediction}</p>
					</div>
				)}
			</section>
		</main>
	);
}
