"""Trade-level volume profile calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeNode:
    low: float
    high: float
    center: float
    volume: float
    relative_volume: float
    delta: float
    prominence: float
    node_type: str


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
    hvn_nodes: tuple[VolumeNode, ...] = ()
    lvn_nodes: tuple[VolumeNode, ...] = ()


def _empty_profile() -> VolumeProfile:
    empty = pd.Series(dtype=float, name="volume")
    return VolumeProfile(empty, empty, empty, empty, np.nan, np.nan, np.nan, (), (), (), ())


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


def _structural_nodes(total: pd.Series, delta: pd.Series, *, bin_size: float, smoothing_bins: int, prominence: float, min_separation_bins: int, min_relative_volume: float) -> tuple[tuple[VolumeNode, ...], tuple[VolumeNode, ...]]:
    if len(total) < 3:
        return (), ()
    profile = total.sort_index()
    smooth = profile.rolling(smoothing_bins, center=True, min_periods=1).mean() if smoothing_bins > 1 else profile
    values = smooth.to_numpy(dtype=float)
    raw = profile.to_numpy(dtype=float)
    delta_values = delta.reindex(profile.index).fillna(0.0).to_numpy(dtype=float)
    radius = max(1, smoothing_bins // 2)
    mean_volume = float(profile.mean())
    candidates: dict[str, list[tuple[int, float]]] = {"HVN": [], "LVN": []}
    for i in range(1, len(values) - 1):
        lo, hi = max(0, i - radius), min(len(values), i + radius + 1)
        neighborhood = np.concatenate((values[lo:i], values[i + 1:hi]))
        if neighborhood.size == 0:
            continue
        baseline = float(np.mean(neighborhood))
        if baseline <= 0:
            continue
        rel = (values[i] - baseline) / baseline
        relative_volume = raw[i] / mean_volume if mean_volume else 0.0
        if values[i] >= values[i - 1] and values[i] >= values[i + 1] and rel >= prominence and relative_volume >= min_relative_volume:
            candidates["HVN"].append((i, rel))
        elif values[i] <= values[i - 1] and values[i] <= values[i + 1] and -rel >= prominence:
            candidates["LVN"].append((i, -rel))

    def select(kind: str) -> tuple[VolumeNode, ...]:
        selected: list[tuple[int, float]] = []
        for idx, strength in sorted(candidates[kind], key=lambda x: x[1], reverse=True):
            if all(abs(idx - other[0]) >= min_separation_bins for other in selected):
                selected.append((idx, strength))
        nodes: list[VolumeNode] = []
        for idx, strength in sorted(selected):
            center = float(profile.index[idx])
            half_width = bin_size / 2.0
            nodes.append(VolumeNode(low=center - half_width, high=center + half_width, center=center, volume=float(raw[idx]), relative_volume=float(raw[idx] / mean_volume) if mean_volume else 0.0, delta=float(delta_values[idx]), prominence=float(strength), node_type=kind))
        return tuple(nodes)
    return select("HVN"), select("LVN")


def build_trade_volume_profile(trades: pd.DataFrame, *, tick_size: float, value_area_pct: float = 0.70, profile_bin_ticks: int = 10, node_smoothing_bins: int = 3, node_prominence: float = 0.25, node_min_separation_bins: int = 3, node_min_relative_volume: float = 1.0) -> VolumeProfile:
    """Build a trade-level profile with resolution-aware structural nodes."""
    if trades.empty:
        return _empty_profile()
    if tick_size <= 0 or not 0 < value_area_pct <= 1:
        raise ValueError("tick_size must be > 0 and value_area_pct must be in (0, 1]")
    if profile_bin_ticks < 1 or node_smoothing_bins < 1 or node_prominence < 0 or node_min_separation_bins < 1 or node_min_relative_volume < 0:
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
    profile_ticks = ticks // profile_bin_ticks
    frame["profile_bin"] = profile_ticks
    frame["buy_volume"] = np.where(~frame["buyer_maker"].astype(bool), frame["quantity"], 0.0)
    frame["sell_volume"] = np.where(frame["buyer_maker"].astype(bool), frame["quantity"], 0.0)
    grouped = frame.groupby("profile_bin", sort=True)
    buy_bins = grouped["buy_volume"].sum()
    sell_bins = grouped["sell_volume"].sum()
    total_bins = buy_bins + sell_bins
    delta_bins = buy_bins - sell_bins

    scale = tick_size * profile_bin_ticks
    index = pd.Index(total_bins.index.to_numpy(dtype=np.int64) * scale, dtype=float)
    buy = pd.Series(buy_bins.to_numpy(dtype=float), index=index, name="buy_volume")
    sell = pd.Series(sell_bins.to_numpy(dtype=float), index=index, name="sell_volume")
    total = pd.Series(total_bins.to_numpy(dtype=float), index=index, name="volume")
    delta = pd.Series(delta_bins.to_numpy(dtype=float), index=index, name="delta")
    poc = float(total.idxmax())
    val, vah = _value_area(total, value_area_pct)
    hvn_nodes, lvn_nodes = _structural_nodes(total, delta, bin_size=scale, smoothing_bins=node_smoothing_bins, prominence=node_prominence, min_separation_bins=node_min_separation_bins, min_relative_volume=node_min_relative_volume)
    return VolumeProfile(total, buy, sell, delta, poc, vah, val, tuple(n.center for n in hvn_nodes), tuple(n.center for n in lvn_nodes), hvn_nodes, lvn_nodes)


def profile_resolution_stability(profiles: dict[int, VolumeProfile], *, overlap_tolerance_bins: int = 2) -> pd.DataFrame:
    """Summarize node persistence across profile resolutions."""
    rows: list[dict[str, float | int | str]] = []
    for resolution, profile in sorted(profiles.items()):
        for node in profile.hvn_nodes + profile.lvn_nodes:
            rows.append({"profile_bin_ticks": resolution, "node_type": node.node_type, "center": node.center, "low": node.low, "high": node.high, "relative_volume": node.relative_volume, "prominence": node.prominence})
    return pd.DataFrame(rows)
