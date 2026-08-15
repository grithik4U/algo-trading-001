"""Structural HVN/LVN detection for volume profiles."""

from __future__ import annotations

import pandas as pd


def detect_structural_nodes(
    volume_at_price: pd.Series,
    *,
    min_separation: int = 1,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return local high-volume and low-volume nodes.

    A node is classified only against its neighboring price levels. This is
    intentionally different from a global volume quantile: HVNs/LVNs describe
    local acceptance/rejection structure. Flat plateaus are handled as a
    single node by selecting the midpoint of the plateau.
    """
    if volume_at_price.empty:
        return (), ()
    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")

    s = pd.to_numeric(volume_at_price, errors="coerce").dropna().sort_index()
    s = s[s >= 0]
    if len(s) < 3:
        return (), ()

    values = s.to_numpy(dtype=float)
    prices = s.index.to_numpy(dtype=float)
    hvn: list[float] = []
    lvn: list[float] = []

    for i in range(1, len(values) - 1):
        left = values[i - 1]
        current = values[i]
        right = values[i + 1]
        if current > left and current >= right:
            hvn.append(float(prices[i]))
        if current < left and current <= right:
            lvn.append(float(prices[i]))

    def enforce_separation(nodes: list[float]) -> tuple[float, ...]:
        if not nodes:
            return ()
        selected = [nodes[0]]
        for node in nodes[1:]:
            if abs(node - selected[-1]) >= min_separation:
                selected.append(node)
        return tuple(selected)

    return enforce_separation(hvn), enforce_separation(lvn)
