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


def _empty_profile() -> VolumeProfile:
    empty = pd.Series(dtype=float, name="volume")
    return VolumeProfile(empty, empty, empty, empty, np.nan, np.nan, np.nan, (), ())


def _value_area(total: pd.Series, value_area_pct: float) -> tuple[float, float]:
    levels = total.sort_index()
    positions = {price: i for i, price in enumerate(levels.index)}
    poc_price = float(total.idxmax())
    left = right = positions[poc_price]
    included = float(levels.iloc[left])
    target = float(levels.sum() * value_area_pct)
    while included < target and (left > 0 or right < len(levels) - 1):
        left_candidate = float(levels.iloc[left - 1]) if left > 0 else -np.inf
        right_candidate = float(levels.iloc[right + 1]) if right < len(levels) - 1 else -np.inf
        if right_candidate >= left_candidate:
            right += 1
            included += float(levels.iloc[right])
        else:
            left -= 1
            included += float(levels.iloc[left])
    return float(levels.index[left]), float(levels.index[right])


def _structural_nodes(total: pd.Series, *, smoothing_bins: int, prominence: float, min_separation_bins: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(total) < 3:
        return (), ()
    profile = total.sort_index()
    smooth = profile.rolling(smoothing_bins, center=True, min_periods=1).mean() if smoothing_bins > 1 else profile
    values = smooth.to_numpy(dtype=float)
    radius = max(1, smoothing_bins // 2)
    candidates_h: list[tuple[int, float]] = []
    candidates_l: list[tuple[int, float]] = []
    for i in range(1, len(values) - 1):
        lo, hi = max(0, i - radius), min(len(values), i + radius + 1)
        neighborhood = np.concatenate((values[lo:i], values[i + 1:hi]))
        if neighborhood.size == 0:
            continue
        baseline = float(np.mean(neighborhood))
        if baseline <= 0:
            continue
        rel = (values[i] - baseline) / baseline
        if values[i] >= values[i - 1] and values[i] >= values[i + 1] and rel >= prominence:
            candidates_h.append((i, rel))
        elif values[i] <= values[i - 1] and values[i] <= values[i + 1] and -rel >= prominence:
            candidates_l.append((i, -rel))

    def select(candidates: list[tuple[int, float]]) -> tuple[float, ...]:
        selected: list[int] = []
        for idx, _strength in sorted(candidates, key=lambda x: x[1], reverse=True):
            if all(abs(idx - other) >= min_separation_bins for other in selected):
                selected.append(idx)
        return tuple(float(profile.index[i]) for i in sorted(selected))

    return select(candidates_h), select(candidates_l)


def build_trade_volume_profile(
    trades: pd.DataFrame,
    *,
    tick_size: float,
    value_area_pct: float = 0.70,
    profile_bin_ticks: int = 1,
    node_smoothing_bins: int = 5,
    node_prominence: float = 0.20,
    node_min_separation_bins: int = 3,
) -> VolumeProfile:
    """Build a trade-level profile with configurable resolution and structural nodes."""
    if trades.empty:
        return _empty_profile()
    if tick_size <= 0 or not 0 < value_area_pct <= 1:
        raise ValueError("tick_size must be > 0 and value_area_pct must be in (0, 1]")
    if profile_bin_ticks < 1 or node_smoothing_bins < 1 or node_prominence < 0 or node_min_separation_bins < 1:
        raise ValueError("profile/node parameters are invalid")
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
        return _empty_profile()

    ticks = np.rint(frame["price"].to_numpy(dtype=float) / tick_size).astype(np.int64)
    frame["price_level"] = (ticks // profile_bin_ticks) * profile_bin_ticks
    frame["buy_volume"] = np.where(~frame["buyer_maker"].astype(bool), frame["quantity"], 0.0)
    frame["sell_volume"] = np.where(frame["buyer_maker"].astype(bool), frame["quantity"], 0.0)
    grouped = frame.groupby("price_level", sort=True)
    buy_ticks = grouped["buy_volume"].sum()
    sell_ticks = grouped["sell_volume"].sum()
    total_ticks = buy_ticks + sell_ticks
    delta_ticks = buy_ticks - sell_ticks
    scale = tick_size * profile_bin_ticks
    index = pd.Index(total_ticks.index.to_numpy(dtype=np.int64) * scale, dtype=float)
    buy = pd.Series(buy_ticks.to_numpy(dtype=float), index=index, name="buy_volume")
    sell = pd.Series(sell_ticks.to_numpy(dtype=float), index=index, name="sell_volume")
    total = pd.Series(total_ticks.to_numpy(dtype=float), index=index, name="volume")
    delta = pd.Series(delta_ticks.to_numpy(dtype=float), index=index, name="delta")
    poc = float(total.idxmax())
    val, vah = _value_area(total, value_area_pct)
    hvns, lvns = _structural_nodes(total, smoothing_bins=node_smoothing_bins, prominence=node_prominence, min_separation_bins=node_min_separation_bins)
    return VolumeProfile(total, buy, sell, delta, poc, vah, val, hvns, lvns)
