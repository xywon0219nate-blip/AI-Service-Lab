// components/TransactionForm.jsx
// 거래 정보 입력 폼 (5개 입력값 + 샘플 불러오기 + 초기화 + 분석 버튼)

const TRANSACTION_TYPE_OPTIONS = [
  { value: "CASH_IN", label: "CASH_IN (입금)" },
  { value: "CASH_OUT", label: "CASH_OUT (현금 인출)" },
  { value: "DEBIT", label: "DEBIT (직불 거래)" },
  { value: "PAYMENT", label: "PAYMENT (결제)" },
  { value: "TRANSFER", label: "TRANSFER (계좌 이체)" },
];

export default function TransactionForm({
  form,
  onChange,
  onSubmit,
  onReset,
  onLoadSample,
  loading,
  samplesAvailable,
}) {
  return (
    <form className="transaction-form" onSubmit={onSubmit}>
      <div className="field">
        <label htmlFor="transaction_type">거래 유형</label>
        <select
          id="transaction_type"
          name="transaction_type"
          value={form.transaction_type}
          onChange={onChange}
        >
          {TRANSACTION_TYPE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label htmlFor="amount">거래 금액</label>
        <input
          id="amount"
          name="amount"
          type="number"
          min="0"
          step="0.01"
          value={form.amount}
          onChange={onChange}
          placeholder="예: 181000"
          required
        />
      </div>

      <div className="field">
        <label htmlFor="sender_old_balance">송금자 거래 전 잔액</label>
        <input
          id="sender_old_balance"
          name="sender_old_balance"
          type="number"
          min="0"
          step="0.01"
          value={form.sender_old_balance}
          onChange={onChange}
          placeholder="예: 181000"
          required
        />
      </div>

      <div className="field">
        <label htmlFor="receiver_old_balance">수취인 거래 전 잔액</label>
        <input
          id="receiver_old_balance"
          name="receiver_old_balance"
          type="number"
          min="0"
          step="0.01"
          value={form.receiver_old_balance}
          onChange={onChange}
          placeholder="예: 0"
          required
        />
      </div>

      <div className="field">
        <label htmlFor="transaction_hour">거래 발생 시간 (0~23시)</label>
        <input
          id="transaction_hour"
          name="transaction_hour"
          type="number"
          min="0"
          max="23"
          step="1"
          value={form.transaction_hour}
          onChange={onChange}
          placeholder="예: 2"
          required
        />
      </div>

      <div className="sample-buttons">
        <button
          type="button"
          className="secondary-button"
          onClick={() => onLoadSample("normal")}
          disabled={!samplesAvailable}
        >
          정상 샘플
        </button>
        <button
          type="button"
          className="secondary-button"
          onClick={() => onLoadSample("suspicious")}
          disabled={!samplesAvailable}
        >
          의심 샘플
        </button>
        <button type="button" className="secondary-button" onClick={onReset}>
          초기화
        </button>
      </div>

      <button type="submit" className="primary-button" disabled={loading}>
        {loading ? "분석 중..." : "위험도 분석하기"}
      </button>
    </form>
  );
}
