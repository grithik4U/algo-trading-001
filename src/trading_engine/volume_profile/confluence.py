"""Join liquidity zones with volume-profile reference prices."""

from __future__ import annotations

import pandas as pd


def score_profile_confluence(
    zones: pd.DataFrame,
    profile_prices: dict[str, float],
    tolerance: float = 0.0005,
) -> pd.DataFrame:
    """Add proximity features for POC/VAH/VAL and profile nodes.

    ``profile_prices`` may contain POC, VAH, VAL and selected HVN/LVN levels.
    A reference is considered proximate when its relative distance from the
    zone price is within ``tolerance``.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    required = {"zone_id", "price"}
    missing = required.difference(zones.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = zones.copy()
    for name, reference in profile_prices.items():
        ref = float(reference)
        distance = (result["price"] - ref).abs() / result["price"].abs().clip(lower=1e-12)
        result[f"near_{name}"] = distance <= tolerance
        result[f"distance_{name}"] = distance

    proximity_cols = [c for c in result.columns if c.startswith("near_")]
    result["profile_confluence_count"] = (
        result[proximity_cols].sum(axis=1) if proximity_cols else 0
    )
    return result
