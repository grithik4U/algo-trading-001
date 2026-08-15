"""Performance analytics for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class PerformanceReport:
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    expectancy: float
    avg_r: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    avg_mae: float | None
    avg_mfe: float | None


def performance_report(trades: pd.DataFrame, equity: pd.Series | None = None, periods_per_year: int = 252) -> PerformanceReport:
    """Calculate trade-level and equity-level performance statistics."""
    if "pnl" not in trades.columns:
        raise ValueError("trades must contain pnl")
    n = len(trades)
    if n == 0:
        return PerformanceReport(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None)

    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = gp / gl if gl > 0 else math.inf
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0

    max_dd = max_dd_pct = sharpe = sortino = 0.0
    if equity is not None and len(equity):
        eq = pd.to_numeric(equity, errors="coerce").dropna()
        peak = eq.cummax()
        dd = eq - peak
        max_dd = float(-dd.min())
        max_dd_pct = float((-dd / peak.replace(0, pd.NA)).max())
        returns = eq.pct_change().dropna()
        if len(returns) > 1 and returns.std(ddof=1) > 0:
            sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year))
        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std(ddof=1) > 0:
            sortino = float(returns.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))

    avg_r = float(pd.to_numeric(trades["r_multiple"], errors="coerce").mean()) if "r_multiple" in trades else 0.0
    mae = float(pd.to_numeric(trades["mae"], errors="coerce").mean()) if "mae" in trades else None
    mfe = float(pd.to_numeric(trades["mfe"], errors="coerce").mean()) if "mfe" in trades else None

    return PerformanceReport(
        trades=n, wins=len(wins), losses=len(losses), win_rate=len(wins) / n,
        net_pnl=float(pnl.sum()), gross_profit=gp, gross_loss=gl,
        profit_factor=pf, expectancy=float(pnl.mean()), avg_r=avg_r,
        max_drawdown=max_dd, max_drawdown_pct=max_dd_pct,
        sharpe=sharpe, sortino=sortino, avg_win=avg_win, avg_loss=avg_loss,
        payoff_ratio=avg_win / avg_loss if avg_loss else math.inf,
        avg_mae=mae, avg_mfe=mfe,
    )
