import { useEffect, useState } from "react";
import "./App.css";

import TransactionForm from "./components/TransactionForm";
import ResultPanel from "./components/ResultPanel";
import { getModelInfo, getSamples, predictTransaction } from "./services/api";

const INITIAL_FORM = {
	transaction_type: "PAYMENT",
	amount: "",
	sender_old_balance: "",
	receiver_old_balance: "",
	transaction_hour: "",
};

function extractErrorMessage(error) {
	const response = error?.response;

	if (!response) {
		return "FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.";
	}

	if (response.status === 503) {
		return response.data?.detail ?? "모델이 아직 준비되지 않았습니다.";
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
	const [form, setForm] = useState(INITIAL_FORM);
	const [sampleSource, setSampleSource] = useState(null);

	const [result, setResult] = useState(null);
	const [error, setError] = useState("");
	const [loading, setLoading] = useState(false);

	const [modelInfo, setModelInfo] = useState(null);
	const [samples, setSamples] = useState(null);

	useEffect(() => {
		getModelInfo()
			.then(setModelInfo)
			.catch(() => setModelInfo(null));

		getSamples()
			.then(setSamples)
			.catch(() => setSamples(null));
	}, []);

	function handleChange(e) {
		const { name, value } = e.target;
		setForm((prev) => ({ ...prev, [name]: value }));
		setSampleSource(null);
	}

	function handleReset() {
		setForm(INITIAL_FORM);
		setSampleSource(null);
		setResult(null);
		setError("");
	}

	function handleLoadSample(kind) {
		if (!samples || !samples[kind]) return;
		const sample = samples[kind];
		setForm({
			transaction_type: sample.transaction_type,
			amount: sample.amount,
			sender_old_balance: sample.sender_old_balance,
			receiver_old_balance: sample.receiver_old_balance,
			transaction_hour: sample.transaction_hour,
		});
		setSampleSource(sample.source);
		setResult(null);
		setError("");
	}

	async function handleSubmit(e) {
		e.preventDefault();
		setResult(null);
		setError("");
		setLoading(true);

		try {
			const payload = {
				transaction_type: form.transaction_type,
				amount: Number(form.amount),
				sender_old_balance: Number(form.sender_old_balance),
				receiver_old_balance: Number(form.receiver_old_balance),
				transaction_hour: Number(form.transaction_hour),
			};

			const data = await predictTransaction(payload);
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
				<h1>Card Fraud Detection AI</h1>
				<p className="subtitle">AI 기반 금융거래 이상 탐지 서비스</p>
				{modelInfo?.model_loaded === false && (
					<p className="model-warning">
						아직 학습된 모델이 없습니다. data 폴더에 PaySim 데이터를 추가한 뒤
						모델을 학습하면 분석 기능을 사용할 수 있습니다.
					</p>
				)}
			</header>

			<section className="dashboard">
				<div className="panel input-panel">
					<h2>거래 정보 입력</h2>
					<TransactionForm
						form={form}
						onChange={handleChange}
						onSubmit={handleSubmit}
						onReset={handleReset}
						onLoadSample={handleLoadSample}
						loading={loading}
						samplesAvailable={Boolean(samples)}
					/>
					{sampleSource && (
						<p className="sample-source-note">
							불러온 샘플 출처:{" "}
							{sampleSource === "real_data"
								? "실제 PaySim 데이터"
								: "데모(예시) 값"}
						</p>
					)}
				</div>

				<div className="panel result-panel-wrapper">
					<h2>분석 결과</h2>
					<ResultPanel
						result={result}
						error={error}
						loading={loading}
						onReanalyze={handleReanalyze}
					/>
				</div>
			</section>

			<footer className="page-footer">
				본 결과는 AI 모델 기반 참고 정보이며 실제 금융기관의 최종 거래 판단을
				대신하지 않습니다.
			</footer>
		</main>
	);
}
