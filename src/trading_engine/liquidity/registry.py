"""Unified registry for structural and session liquidity levels."""

from __future__ import annotations

import pandas as pd


def build_liquidity_registry(
    structural_levels: pd.DataFrame,
    session_levels: pd.DataFrame,
) -> pd.DataFrame:
    """Combine structural and session levels into one normalized registry.

    The registry is a feature store, not a claim that each row represents a
    visible resting order. ``source`` identifies where the level originated.
    """
    columns = [
        "timestamp", "price", "side", "timeframe", "source",
        "liquidity_class", "structural_significance", "session",
    ]

    frames: list[pd.DataFrame] = []

    if not structural_levels.empty:
        s = structural_levels.copy()
        s["source"] = "structure"
        if "liquidity_class" not in s:
            s["liquidity_class"] = "internal"
        if "structural_significance" not in s:
            s["structural_significance"] = 0.0
        s["session"] = pd.NA
        frames.append(s)

    if not session_levels.empty:
        s = session_levels.copy()
        s["source"] = "session"
        s["liquidity_class"] = "external"
        s["structural_significance"] = 1.0
        if "session" not in s:
            s["session"] = pd.NA
        frames.append(s)

    if not frames:
        return pd.DataFrame(columns=columns)

    result = pd.concat(frames, ignore_index=True, sort=False)
    result["side"] = result["side"].replace(
        {"high": "BSL", "low": "SSL"}
    )
    return result[columns].sort_values("timestamp", ignore_index=True)
