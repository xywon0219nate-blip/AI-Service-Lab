// components/ResultPanel.jsx
// 분석 결과 영역 (초기 안내 / Loading / 오류 / 정상·의심 결과)

const RISK_LEVEL_STYLE = {
  Low: { color: "#16a34a", background: "#f0fdf4", border: "#bbf7d0" },
  Medium: { color: "#d97706", background: "#fffbeb", border: "#fde68a" },
  High: { color: "#dc2626", background: "#fef2f2", border: "#fecaca" },
};

export default function ResultPanel({ result, error, loading, onReanalyze }) {
  if (loading) {
    return (
      <div className="result-panel state-loading">
        <div className="spinner" />
        <p>거래 위험도를 분석하고 있습니다...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-panel state-error">
        <p className="state-title">분석에 실패했습니다</p>
        <p className="state-message">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-panel state-empty">
        <p className="state-title">거래 정보를 입력해주세요</p>
        <p className="state-message">
          왼쪽 폼에 거래 정보를 입력하고 &quot;위험도 분석하기&quot; 버튼을 누르면
          AI 모델이 이상거래 위험도를 분석합니다.
        </p>
        <p className="state-caption">
          정상/의심 샘플 버튼으로 예시 거래를 바로 불러와 볼 수도 있습니다.
        </p>
      </div>
    );
  }

  const isSuspicious = result.prediction === 1;
  const riskStyle = RISK_LEVEL_STYLE[result.risk_level] ?? RISK_LEVEL_STYLE.Low;
  const probabilityPercent = Math.round(result.fraud_probability * 1000) / 10;

  return (
    <div className={`result-panel state-result ${isSuspicious ? "suspicious" : "normal"}`}>
      <p className={`result-label ${isSuspicious ? "label-suspicious" : "label-normal"}`}>
        {isSuspicious ? "Suspicious" : "Normal"}
      </p>
      <p className="result-message">{result.message}</p>

      <div className="probability-block">
        <div className="probability-header">
          <span>Fraud Probability</span>
          <span>{probabilityPercent}%</span>
        </div>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{
              width: `${probabilityPercent}%`,
              background: riskStyle.color,
            }}
          />
        </div>
      </div>

      <div className="badge-row">
        <span
          className="risk-badge"
          style={{
            color: riskStyle.color,
            background: riskStyle.background,
            borderColor: riskStyle.border,
          }}
        >
          Risk Level: {result.risk_level}
        </span>
        <span className="threshold-badge">Threshold: {result.threshold}</span>
      </div>

      {result.risk_factors?.length > 0 && (
        <div className="risk-factors">
          <p className="risk-factors-title">주요 확인 요인 (참고용)</p>
          <ul>
            {result.risk_factors.map((factor) => (
              <li key={factor}>{factor}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="disclaimer">{result.disclaimer}</p>

      <button type="button" className="secondary-button reanalyze-button" onClick={onReanalyze}>
        다시 분석하기
      </button>
    </div>
  );
}
