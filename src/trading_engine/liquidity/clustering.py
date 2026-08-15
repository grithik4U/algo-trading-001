"""Cluster nearby structural levels into candidate liquidity pools.

The clustering logic is intentionally price-based and causal: a pool can only
contain levels whose structural timestamps are already known.  It does not
claim that every cluster represents actual resting stop orders; it creates a
measurable proxy that can be tested against subsequent price behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LiquidityPool:
    """A cluster of nearby structural levels."""

    pool_id: int
    side: str
    price: float
    members: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    strength: float


def cluster_levels(
    levels: pd.DataFrame,
    tolerance: float = 0.0005,
    min_members: int = 2,
) -> pd.DataFrame:
    """Cluster BSL/SSL levels that are within ``tolerance`` of each other.

    ``tolerance`` is expressed as a relative price distance. For example,
    0.0005 means 5 basis points. Levels are processed in timestamp order and
    each new level is assigned to the nearest existing cluster of the same
    side when it falls within tolerance.

    Required columns: timestamp, price, side.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if min_members < 1:
        raise ValueError("min_members must be >= 1")

    required = {"timestamp", "price", "side"}
    missing = required.difference(levels.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if levels.empty:
        return pd.DataFrame(
            columns=[
                "pool_id", "side", "price", "members",
                "first_timestamp", "last_timestamp", "strength",
            ]
        )

    ordered = levels.sort_values("timestamp")
    clusters: list[dict] = []

    for _, row in ordered.iterrows():
        price = float(row["price"])
        side = str(row["side"])
        candidates = [
            c for c in clusters
            if c["side"] == side
            and abs(price - c["price"]) / max(abs(c["price"]), 1e-12) <= tolerance
        ]

        if candidates:
            cluster = min(candidates, key=lambda c: abs(price - c["price"]))
            old_members = cluster["members"]
            cluster["price"] = (
                cluster["price"] * old_members + price
            ) / (old_members + 1)
            cluster["members"] += 1
            cluster["last_timestamp"] = row["timestamp"]
            cluster["strength"] = float(cluster["members"])
        else:
            clusters.append(
                {
                    "side": side,
                    "price": price,
                    "members": 1,
                    "first_timestamp": row["timestamp"],
                    "last_timestamp": row["timestamp"],
                    "strength": 1.0,
                }
            )

    result = [
        {
            "pool_id": i,
            **cluster,
        }
        for i, cluster in enumerate(clusters)
        if cluster["members"] >= min_members
    ]

    return pd.DataFrame(
        result,
        columns=[
            "pool_id", "side", "price", "members",
            "first_timestamp", "last_timestamp", "strength",
        ],
    )
