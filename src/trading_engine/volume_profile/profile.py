"""Volume-at-price profile calculations.

The profile consumes OHLCV today, but keeps the output contract compatible
with later replacement by native trade/tick volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeProfile:
    prices: np.ndarray
    volumes: np.ndarray
    poc: float
    vah: float
    val: float
    hvn: tuple[float, ...]
    lvn: tuple[float, ...]


def _allocate_volume(row: pd.Series, edges: np.ndarray, method: str) -> np.ndarray:
    n_bins = len(edges) - 1
    allocation = np.zeros(n_bins, dtype=float)
    low = float(row["low"])
    high = float(row["high"])
    volume = float(row["volume"])
    if high < low:
        raise ValueError("high cannot be below low")
    if volume < 0:
        raise ValueError("volume cannot be negative")
    if method == "typical_price":
        price = (high + low + float(row["close"])) / 3.0
        idx = min(max(np.searchsorted(edges, price, side="right") - 1, 0), n_bins - 1)
        allocation[idx] = volume
        return allocation
    if method != "uniform_range":
        raise ValueError("method must be 'uniform_range' or 'typical_price'")
    if high == low:
        idx = min(max(np.searchsorted(edges, high, side="right") - 1, 0), n_bins - 1)
        allocation[idx] = volume
        return allocation
    overlap = np.maximum(0.0, np.minimum(edges[1:], high) - np.maximum(edges[:-1], low))
    total = overlap.sum()
    if total > 0:
        allocation = volume * overlap / total
    return allocation


def build_volume_profile(
    df: pd.DataFrame,
    bins: int = 100,
    value_area: float = 0.70,
    allocation: str = "uniform_range",
    node_window: int = 2,
) -> VolumeProfile:
    """Build POC, VAH, VAL, HVNs and LVNs from the supplied volume data."""
    required = {"high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or bins < 2 or not 0 < value_area <= 1 or node_window < 1:
        raise ValueError("invalid volume-profile parameters")

    low = float(df["low"].min())
    high = float(df["high"].max())
    if high == low:
        high = low + max(abs(low) * 1e-9, 1e-9)
    edges = np.linspace(low, high, bins + 1)
    volumes = np.zeros(bins, dtype=float)
    for _, row in df.iterrows():
        volumes += _allocate_volume(row, edges, allocation)

    prices = (edges[:-1] + edges[1:]) / 2.0
    poc_idx = int(np.argmax(volumes))
    target = volumes.sum() * value_area
    included = {poc_idx}
    total = volumes[poc_idx]
    left, right = poc_idx - 1, poc_idx + 1
    while total < target and (left >= 0 or right < bins):
        lv = volumes[left] if left >= 0 else -1.0
        rv = volumes[right] if right < bins else -1.0
        if rv >= lv:
            included.add(right); total += volumes[right]; right += 1
        else:
            included.add(left); total += volumes[left]; left -= 1

    hvn: list[float] = []
    lvn: list[float] = []
    for i in range(node_window, bins - node_window):
        local = volumes[i - node_window:i + node_window + 1]
        if volumes[i] == local.max() and volumes[i] > local.min():
            hvn.append(float(prices[i]))
        if volumes[i] == local.min() and volumes[i] < local.max():
            lvn.append(float(prices[i]))

    return VolumeProfile(
        prices=prices,
        volumes=volumes,
        poc=float(prices[poc_idx]),
        vah=float(prices[max(included)]),
        val=float(prices[min(included)]),
        hvn=tuple(hvn),
        lvn=tuple(lvn),
    )
