"""Chronological event-driven backtest engine with explicit fill assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class BacktestTrade:
    entry_timestamp: datetime
    exit_timestamp: datetime
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    exit_reason: str


@dataclass(frozen=True)
class BacktestResult:
    starting_equity: float
    ending_equity: float
    net_pnl: float
    return_fraction: float
    max_drawdown: float
    trades: tuple[BacktestTrade, ...]


def run_backtest(
    bars: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, int], object | None],
    starting_equity: float,
    fee_per_unit: float = 0.0,
    slippage_per_unit: float = 0.0,
) -> BacktestResult:
    """Replay bars in order and simulate one position at a time.

    ``signal_fn`` receives only bars through index ``i``. It may return an
    object with direction, entry, stop, target and quantity attributes. Orders
    are assumed to fill on the next bar's open, with slippage applied against
    the trader. Stops/targets are evaluated using the same bar's OHLC; if both
    are touched, the conservative stop-first assumption is used.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if starting_equity <= 0 or fee_per_unit < 0 or slippage_per_unit < 0:
        raise ValueError("invalid backtest parameters")

    data = bars.sort_index()
    equity = float(starting_equity)
    peak = equity
    max_drawdown = 0.0
    trades: list[BacktestTrade] = []
    position = None

    for i in range(len(data) - 1):
        if position is None:
            signal = signal_fn(data.iloc[: i + 1], i)
            if signal is None:
                continue
            direction = getattr(signal, "direction")
            quantity = float(getattr(signal, "quantity"))
            stop = float(getattr(signal, "stop"))
            target = float(getattr(signal, "target"))
            if direction not in {"long", "short"} or quantity <= 0:
                continue
            next_bar = data.iloc[i + 1]
            raw_entry = float(next_bar["open"])
            entry = raw_entry + slippage_per_unit if direction == "long" else raw_entry - slippage_per_unit
            position = {
                "entry_timestamp": data.index[i + 1],
                "entry": entry,
                "direction": direction,
                "quantity": quantity,
                "stop": stop,
                "target": target,
                "entry_slippage": slippage_per_unit * quantity,
            }
            continue

        bar = data.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        direction = position["direction"]
        exit_price = None
        reason = None
        if direction == "long":
            stop_hit = low <= position["stop"]
            target_hit = high >= position["target"]
            if stop_hit:
                exit_price, reason = position["stop"], "stop"
            elif target_hit:
                exit_price, reason = position["target"], "target"
        else:
            stop_hit = high >= position["stop"]
            target_hit = low <= position["target"]
            if stop_hit:
                exit_price, reason = position["stop"], "stop"
            elif target_hit:
                exit_price, reason = position["target"], "target"

        if exit_price is None:
            continue

        exit_price = exit_price - slippage_per_unit if direction == "long" else exit_price + slippage_per_unit
        qty = position["quantity"]
        gross = (exit_price - position["entry"]) * qty if direction == "long" else (position["entry"] - exit_price) * qty
        fees = fee_per_unit * qty * 2
        slippage = position["entry_slippage"] + slippage_per_unit * qty
        net = gross - fees
        equity += net
        trade = BacktestTrade(
            position["entry_timestamp"], data.index[i], direction,
            position["entry"], exit_price, qty, gross, fees, slippage, net, reason,
        )
        trades.append(trade)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
        position = None

    if position is not None:
        final_ts = data.index[-1]
        final_raw = float(data.iloc[-1]["close"])
        final_price = final_raw - slippage_per_unit if position["direction"] == "long" else final_raw + slippage_per_unit
        qty = position["quantity"]
        gross = (final_price - position["entry"]) * qty if position["direction"] == "long" else (position["entry"] - final_price) * qty
        fees = fee_per_unit * qty * 2
        net = gross - fees
        equity += net
        trades.append(BacktestTrade(position["entry_timestamp"], final_ts, position["direction"], position["entry"], final_price, qty, gross, fees, position["entry_slippage"] + slippage_per_unit * qty, net, "end_of_data"))
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)

    return BacktestResult(starting_equity, equity, equity - starting_equity, (equity / starting_equity) - 1, max_drawdown, tuple(trades))
