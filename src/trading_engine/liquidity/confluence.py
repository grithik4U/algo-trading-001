"""Confluence zones built from nearby liquidity levels."""

from __future__ import annotations

import pandas as pd


def build_confluence_zones(
    registry: pd.DataFrame,
    tolerance: float = 0.0005,
    min_sources: int = 2,
) -> pd.DataFrame:
    """Group nearby levels into price zones and score independent confluence.

    A zone is a research proxy for an area where multiple independently
    derived structural references overlap. It is not assumed to represent
    actual resting orders.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if min_sources < 1:
        raise ValueError("min_sources must be >= 1")

    required = {"timestamp", "price", "side", "source"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if registry.empty:
        return pd.DataFrame(
            columns=["zone_id", "side", "price", "levels", "sources", "score"]
        )

    clusters: list[dict] = []
    for _, row in registry.sort_values("timestamp").iterrows():
        price = float(row["price"])
        side = str(row["side"])
        candidates = [
            z for z in clusters
            if z["side"] == side
            and abs(price - z["price"]) / max(abs(z["price"]), 1e-12) <= tolerance
        ]
        if candidates:
            zone = min(candidates, key=lambda z: abs(price - z["price"]))
            n = zone["levels"]
            zone["price"] = (zone["price"] * n + price) / (n + 1)
            zone["levels"] += 1
            zone["sources"].add(str(row["source"]))
        else:
            clusters.append(
                {
                    "side": side,
                    "price": price,
                    "levels": 1,
                    "sources": {str(row["source"])},
                }
            )

    result = []
    zone_id = 0
    for zone in clusters:
        source_count = len(zone["sources"])
        if source_count < min_sources:
            continue
        result.append(
            {
                "zone_id": zone_id,
                "side": zone["side"],
                "price": zone["price"],
                "levels": zone["levels"],
                "sources": tuple(sorted(zone["sources"])),
                "score": float(source_count + 0.25 * (zone["levels"] - source_count)),
            }
        )
        zone_id += 1

    return pd.DataFrame(
        result,
        columns=["zone_id", "side", "price", "levels", "sources", "score"],
    )
