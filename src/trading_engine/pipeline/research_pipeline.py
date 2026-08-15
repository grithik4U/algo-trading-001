"""Chronological orchestration contract for the trading research stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class PipelineResult:
    signals: list[Any]
    plans: list[Any]
    trades: pd.DataFrame
    equity: pd.Series


def run_research_pipeline(
    data: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, int], Any | None],
    plan_fn: Callable[[Any], Any | None],
    execute_fn: Callable[[Any, pd.DataFrame, int], Any | None],
) -> PipelineResult:
    """Run signal, risk and execution callbacks in strict timestamp order.

    signal_fn receives the complete prefix ending at the current bar. It is
    therefore responsible for using only information available by that bar.
    Execution remains a separate callback so fills, slippage and fees are
    explicit and testable.
    """
    if data.empty:
        return PipelineResult([], [], pd.DataFrame(), pd.Series(dtype=float))
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be monotonically increasing")

    signals: list[Any] = []
    plans: list[Any] = []
    trades: list[Any] = []
    equity_values: list[float] = []
    equity = 0.0

    for i in range(len(data)):
        prefix = data.iloc[: i + 1]
        signal = signal_fn(prefix, i)
        if signal is None:
            equity_values.append(equity)
            continue
        signals.append(signal)

        plan = plan_fn(signal)
        if plan is None:
            equity_values.append(equity)
            continue
        plans.append(plan)

        result = execute_fn(plan, data, i)
        if result is not None:
            trades.append(result)
            pnl = float(result.get("pnl", 0.0)) if isinstance(result, dict) else float(getattr(result, "pnl", 0.0))
            equity += pnl
        equity_values.append(equity)

    return PipelineResult(
        signals=signals,
        plans=plans,
        trades=pd.DataFrame(trades),
        equity=pd.Series(equity_values, index=data.index, name="equity"),
    )
