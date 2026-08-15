"""Trade-level volume profile calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfile:
    volume_at_price: pd.Series
    buy_volume_at_price: pd.Series
    sell_volume_at_price: pd.Series
    delta_at_price: pd.Series
    poc: float
    vah: float
    val: float
    hvns: tuple[float, ...]
    lvns: tuple[float, ...]


def build_trade_volume_profile(trades: pd.DataFrame, *, tick_size: float, value_area_pct: float = 0.70, hvn_quantile: float = 0.80, lvn_quantile: float = 0.20) -> VolumeProfile:
    """Build a volume-at-price profile from aggregate trades."""
    if trades.empty:
        empty = pd.Series(dtype=float, name="volume")
        return VolumeProfile(empty, empty, empty, empty, np.nan, np.nan, np.nan, (), ())
    if tick_size <= 0 or not 0 < value_area_pct <= 1:
        raise ValueError("tick_size must be > 0 and value_area_pct must be in (0, 1]")
    required = {"price", "quantity", "buyer_maker"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"missing trade columns: {sorted(missing)}")

    frame = trades[["price", "quantity", "buyer_maker"]].copy()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame = frame.dropna(subset=["price", "quantity"])
    frame = frame[frame["quantity"] > 0]
    if frame.empty:
        empty = pd.Series(dtype=float, name="volume")
        return VolumeProfile(empty, empty, empty, empty, np.nan, np.nan, np.nan, (), ())

    frame["price_level"] = (np.round(frame["price"] / tick_size) * tick_size).round(12)
    frame["buy_volume"] = np.where(~frame["buyer_maker"].astype(bool), frame["quantity"], 0.0)
    frame["sell_volume"] = np.where(frame["buyer_maker"].astype(bool), frame["quantity"], 0.0)
    grouped = frame.groupby("price_level", sort=True)
    buy = grouped["buy_volume"].sum()
    sell = grouped["sell_volume"].sum()
    total = buy + sell
    delta = buy - sell

    poc = float(total.idxmax())
    target = float(total.sum() * value_area_pct)
    ordered = total.sort_values(ascending=False)
    included = ordered.cumsum() <= target
    if not included.any():
        included.iloc[0] = True
    value_prices = ordered.index[included]
    vah = float(max(value_prices))
    val = float(min(value_prices))

    hvns = tuple(float(x) for x in total[total >= total.quantile(hvn_quantile)].index)
    lvns = tuple(float(x) for x in total[total <= total.quantile(lvn_quantile)].index)
    total.name = "volume"
    buy.name = "buy_volume"
    sell.name = "sell_volume"
    delta.name = "delta"
    return VolumeProfile(total, buy, sell, delta, poc, vah, val, hvns, lvns)
