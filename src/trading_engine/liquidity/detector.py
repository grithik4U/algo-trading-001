"""Causal liquidity-pool detection from confirmed swing/level clusters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class LiquidityCandidate:
    timestamp: datetime
    price: float
    side: str
    source: str
    touches: int
    strength: float


def detect_level_clusters(
    levels: pd.DataFrame,
    tolerance: float,
    min_touches: int = 2,
) -> list[LiquidityCandidate]:
    """Cluster already-confirmed levels into candidate liquidity pools.

    Required columns: timestamp, price, side, source. Clustering uses only
    supplied historical levels; callers must ensure each level was confirmed
    without future information.
    """
    required = {"timestamp", "price", "side", "source"}
    missing = required.difference(levels.columns)
    if missing:
        raise ValueError(f"Missing level columns: {sorted(missing)}")
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    if min_touches < 1:
        raise ValueError("min_touches must be >= 1")

    candidates: list[LiquidityCandidate] = []
    for side, group in levels.groupby("side"):
        ordered = group.sort_values("price")
        cluster: list[pd.Series] = []
        cluster_center: float | None = None

        for _, level in ordered.iterrows():
            price = float(level["price"])
            if cluster_center is None or abs(price - cluster_center) <= tolerance:
                cluster.append(level)
                cluster_center = sum(float(x["price"]) for x in cluster) / len(cluster)
            else:
                if len(cluster) >= min_touches:
                    candidates.append(_candidate(cluster, side))
                cluster = [level]
                cluster_center = price
        if len(cluster) >= min_touches:
            candidates.append(_candidate(cluster, side))

    return sorted(candidates, key=lambda x: x.timestamp)


def _candidate(cluster: list[pd.Series], side: str) -> LiquidityCandidate:
    price = sum(float(x["price"]) for x in cluster) / len(cluster)
    latest = max(cluster, key=lambda x: x["timestamp"])
    sources = "+".join(sorted({str(x["source"]) for x in cluster}))
    touches = len(cluster)
    strength = float(touches)
    return LiquidityCandidate(
        timestamp=latest["timestamp"],
        price=price,
        side=str(side),
        source=sources,
        touches=touches,
        strength=strength,
    )
