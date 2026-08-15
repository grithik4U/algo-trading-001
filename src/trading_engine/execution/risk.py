"""Execution and risk sizing primitives for qualified setups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradePlan:
    direction: str
    entry: float
    stop: float
    target: float
    risk_per_unit: float
    reward_per_unit: float
    rr: float
    quantity: float
    risk_amount: float
    valid: bool
    reason: str


def build_trade_plan(
    *,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    account_equity: float,
    risk_fraction: float,
    min_rr: float = 2.0,
    tick_size: float = 0.0,
    point_value: float = 1.0,
) -> TradePlan:
    """Create a fixed-risk plan; no order is submitted by this function."""
    if direction not in {"long", "short"}:
        raise ValueError("direction must be long or short")
    if account_equity <= 0 or not 0 < risk_fraction <= 1:
        raise ValueError("invalid account risk parameters")
    if point_value <= 0 or min_rr <= 0:
        raise ValueError("point_value and min_rr must be > 0")
    if tick_size < 0:
        raise ValueError("tick_size must be >= 0")

    risk_per_unit = (entry - stop) if direction == "long" else (stop - entry)
    reward_per_unit = (target - entry) if direction == "long" else (entry - target)
    if risk_per_unit <= 0:
        return TradePlan(direction, entry, stop, target, risk_per_unit, reward_per_unit, 0.0, 0.0, 0.0, False, "invalid_stop")
    if reward_per_unit <= 0:
        return TradePlan(direction, entry, stop, target, risk_per_unit, reward_per_unit, 0.0, 0.0, 0.0, False, "invalid_target")

    rr = reward_per_unit / risk_per_unit
    if rr < min_rr:
        return TradePlan(direction, entry, stop, target, risk_per_unit, reward_per_unit, rr, 0.0, 0.0, False, "rr_below_minimum")

    risk_amount = account_equity * risk_fraction
    quantity = risk_amount / (risk_per_unit * point_value)
    if tick_size:
        quantity = quantity  # contract-size rounding belongs to instrument adapter

    return TradePlan(direction, entry, stop, target, risk_per_unit, reward_per_unit, rr, quantity, risk_amount, True, "qualified")
