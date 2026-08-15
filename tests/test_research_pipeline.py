import pandas as pd

from trading_engine.pipeline.research_pipeline import run_research_pipeline


def test_signal_callback_only_receives_current_prefix():
    index = pd.date_range("2026-01-01", periods=4, freq="min")
    data = pd.DataFrame({"close": [100, 101, 102, 103]}, index=index)
    seen_lengths = []

    def signal_fn(prefix, i):
        seen_lengths.append(len(prefix))
        return {"pnl": 1.0} if i == 2 else None

    def plan_fn(signal):
        return signal

    def execute_fn(plan, full_data, i):
        return plan

    result = run_research_pipeline(data, signal_fn, plan_fn, execute_fn)

    assert seen_lengths == [1, 2, 3, 4]
    assert len(result.trades) == 1
    assert result.equity.iloc[-1] == 1.0
