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
    """Expand contiguously from POC until the requested volume is included."""
    levels = total.sort_index()
    positions = {price: i for i, price in enumerate(levels.index)}
    poc_price = float(total.idxmax())
    left = right = positions[poc_price]
    included_volume = float(levels.iloc[left])
    target = float(levels.sum() * value_area_pct)

    while included_volume < target and (left > 0 or right < len(levels) - 1):
        left_candidate = float(levels.iloc[left - 1]) if left > 0 else -np.inf
        right_candidate = float(levels.iloc[right + 1]) if right < len(levels) - 1 else -np.inf
        if right_candidate >= left_candidate:
            right += 1
            included_volume += float(levels.iloc[right])
        else:
            left -= 1
            included_volume += float(levels.iloc[left])

    return float(levels.index[left]), float(levels.index[right])


def _structural_nodes(
    total: pd.Series,
    *,
    smoothing_ticks: int,
    prominence: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Find local volume peaks/troughs on a smoothed price profile.

    ``prominence`` is measured relative to the mean volume of the immediate
    neighborhood. This prevents a nearly flat profile from producing dozens
    of meaningless HVN/LVN labels.
    """
    if len(total) < 3:
        return (), ()

    profile = total.sort_index()
    if smoothing_ticks > 1:
        smooth = profile.rolling(smoothing_ticks, center=True, min_periods=1).mean()
    else:
        smooth = profile

    values = smooth.to_numpy(dtype=float)
    hvns: list[float] = []
    lvns: list[float] = []
    radius = max(1, smoothing_ticks // 2)

    for i in range(1, len(values) - 1):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        neighborhood = np.concatenate((values[lo:i], values[i + 1:hi]))
        if neighborhood.size == 0:
            continue
        baseline = float(np.mean(neighborhood))
        if baseline <= 0:
            continue
        relative = (values[i] - baseline) / baseline
        if values[i] >= values[i - 1] and values[i] >= values[i + 1] and relative >= prominence:
            hvns.append(float(profile.index[i]))
        elif values[i] <= values[i - 1] and values[i] <= values[i + 1] and -relative >= prominence:
            lvns.append(float(profile.index[i]))

    return tuple(hvns), tuple(lvns)


def build_trade_volume_profile(
    trades: pd.DataFrame,
    *,
    tick_size: float,
    value_area_pct: float = 0.70,
    profile_bin_ticks: int = 1,
    node_smoothing_ticks: int = 5,
    node_prominence: float = 0.20,
) -> VolumeProfile:
    """Build a trade-level volume profile.

    Prices are represented internally as integer tick indices to avoid binary
    floating-point artifacts. Value area is contiguous around POC. HVN/LVN
    detection uses local structural peaks/troughs on a smoothed profile.
    """
    if trades.empty:
        return _empty_profile()
    if tick_size <= 0:
        raise ValueError("tick_size must be > 0")
    if not 0 < value_area_pct <= 1:
        raise ValueError("value_area_pct must be in (0, 1]")
    if profile_bin_ticks < 1 or node_smoothing_ticks < 1 or node_prominence < 0:
        raise ValueError("profile_bin_ticks and node_smoothing_ticks must be >= 1 and node_prominence >= 0")

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

    tick_index = np.rint(frame["price"].to_numpy(dtype=float) / tick_size).astype(np.int64)
    frame["price_level"] = (tick_index // profile_bin_ticks) * profile_bin_ticks
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
    hvns, lvns = _structural_nodes(
        total,
        smoothing_ticks=node_smoothing_ticks,
        prominence=node_prominence,
    )

    return VolumeProfile(total, buy, sell, delta, poc, vah, val, hvns, lvns)
