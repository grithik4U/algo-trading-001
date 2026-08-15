"""Causal FVG detection and linkage to displacement/MSS events."""

from __future__ import annotations

import pandas as pd


def detect_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish/bearish three-candle gaps using closed bars only."""
    required = {"high", "low"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    rows: list[dict] = []
    bars = df.reset_index()
    timestamp_col = df.index.name or "index"
    bars = bars.rename(columns={timestamp_col: "timestamp"})

    for i in range(2, len(bars)):
        first = bars.iloc[i - 2]
        middle = bars.iloc[i - 1]
        current = bars.iloc[i]
        if float(current["low"]) > float(first["high"]):
            rows.append({
                "created_timestamp": current["timestamp"], "direction": "bullish",
                "lower_price": float(first["high"]), "upper_price": float(current["low"]),
                "gap_size": float(current["low"]) - float(first["high"]),
                "middle_displacement": bool(middle.get("is_displacement", False)),
                "middle_index": i - 1,
            })
        elif float(current["high"]) < float(first["low"]):
            rows.append({
                "created_timestamp": current["timestamp"], "direction": "bearish",
                "lower_price": float(current["high"]), "upper_price": float(first["low"]),
                "gap_size": float(first["low"]) - float(current["high"]),
                "middle_displacement": bool(middle.get("is_displacement", False)),
                "middle_index": i - 1,
            })
    return pd.DataFrame(rows, columns=[
        "created_timestamp", "direction", "lower_price", "upper_price",
        "gap_size", "middle_displacement", "middle_index",
    ])


def link_fvg_to_event(
    fvg: pd.Series,
    event_timestamp,
    event_direction: str,
    max_bars_after_event: int = 3,
    df: pd.DataFrame | None = None,
) -> bool:
    """Require directional alignment and a causal FVG within the event window."""
    if event_direction not in {"bullish", "bearish"}:
        raise ValueError("event_direction must be bullish or bearish")
    if max_bars_after_event < 0:
        raise ValueError("max_bars_after_event must be >= 0")
    if str(fvg["direction"]) != event_direction or fvg["created_timestamp"] < event_timestamp:
        return False
    if df is None:
        return True
    try:
        event_idx = df.index.get_loc(event_timestamp)
        fvg_idx = df.index.get_loc(fvg["created_timestamp"])
    except KeyError:
        return False
    return 0 <= fvg_idx - event_idx <= max_bars_after_event
