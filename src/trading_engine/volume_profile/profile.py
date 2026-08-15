"""Volume-at-price profile calculations.

The initial implementation is OHLCV-based. Because candle data does not reveal
where within a bar each unit of volume traded, volume is distributed across
price bins using a configurable allocation method. Tick/trade data can later
replace this input without changing the profile output contract.
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


def _allocate_volume(
    row: pd.Series,
    edges: np.ndarray,
    method: str,
) -> np.ndarray:
    """Allocate one candle's volume to price bins."""
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
        idx = np.searchsorted(edges, price, side="right") - 1
        idx = min(max(idx, 0), n_bins - 1)
        allocation[idx] = volume
        return allocation

    if method != "uniform_range":
        raise ValueError("method must be 'uniform_range' or 'typical_price'")

    if high == low:
        idx = np.searchsorted(edges, high, side="right") - 1
        idx = min(max(idx, 0), n_bins - 1)
        allocation[idx] = volume
        return allocation

    overlap = np.maximum(
        0.0,
        np.minimum(edges[1:], high) - np.maximum(edges[:-1], low),
    )
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
    """Build an OHLCV volume profile with POC, value area and nodes.

    Value-area bins are expanded from the POC by selecting the adjacent side
    with greater volume until ``value_area`` of total volume is included.
    HVN/LVN are local volume maxima/minima relative to ``node_window``.
    """
    required = {"high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("df cannot be empty")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if not 0 < value_area <= 1:
        raise ValueError("value_area must be in (0, 1]")
    if node_window < 1:
        raise ValueError("node_window must be >= 1")

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
    poc = float(prices[poc_idx])

    target = volumes.sum() * value_area
    included = {poc_idx}
    total = volumes[poc_idx]
    left = poc_idx - 1
    right = poc_idx + 1
    while total < target and (left >= 0 or right < bins):
        left_volume = volumes[left] if left >= 0 else -1.0
        right_volume = volumes[right] if right < bins else -1.0
        if right_volume >= left_volume:
            included.add(right)
            total += volumes[right]
            right += 1
        else:
            included.add(left)
            total += volumes[left]
            left -= 1

    val = float(prices[min(included)])
    vah = float(prices[max(included)])

    hvn: list[float] = []
    lvn: list[float] = []
    for i in range(node_window, bins - node_window):
        local = volumes[i - node_window : i + node_window + 1]
        center = volumes[i]
        if center == local.max() and center > local.min():
            hvn.append(float(prices[i]))
        if center == local.min() and center < local.max():
            lvn.append(float(prices[i]))

    return VolumeProfile(
        prices=prices,
        volumes=volumes,
        poc=poc,
        vah=vah,
        val=val,
        hvn=tuple(hvn),
        lvn=tuple(lvn),
    )
