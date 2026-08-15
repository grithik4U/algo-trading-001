"""Causal three-candle fair-value-gap / imbalance detection."""

from __future__ import annotations

import pandas as pd


def detect_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish/bearish three-candle gaps using closed bars only.

    Bullish FVG: current low > high two bars earlier.
    Bearish FVG: current high < low two bars earlier.
    The middle candle is the displacement candidate when that feature exists.
    """
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
            rows.append(
                {
                    "created_timestamp": current["timestamp"],
                    "direction": "bullish",
                    "lower_price": float(first["high"]),
                    "upper_price": float(current["low"]),
                    "gap_size": float(current["low"]) - float(first["high"]),
                    "middle_displacement": bool(
                        middle.get("is_displacement", False)
                    ),
                }
            )
        elif float(current["high"]) < float(first["low"]):
            rows.append(
                {
                    "created_timestamp": current["timestamp"],
                    "direction": "bearish",
                    "lower_price": float(current["high"]),
                    "upper_price": float(first["low"]),
                    "gap_size": float(first["low"]) - float(current["high"]),
                    "middle_displacement": bool(
                        middle.get("is_displacement", False)
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "created_timestamp", "direction", "lower_price", "upper_price",
            "gap_size", "middle_displacement",
        ],
    )
