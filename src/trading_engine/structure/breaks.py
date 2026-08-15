"""Causal market-structure break detection."""

from __future__ import annotations

import pandas as pd


def detect_structure_breaks(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    require_displacement: bool = False,
) -> pd.DataFrame:
    """Detect closes through previously confirmed swing levels.

    Required OHLC columns: high, low, close.
    Required swing columns: timestamp, price, side. ``side`` may be high/low,
    BSL/SSL. A break is reported on the first closed bar that accepts beyond
    the confirmed swing level. If ``is_displacement`` exists, it can optionally
    be required on the break bar.
    """
    required_df = {"high", "low", "close"}
    missing_df = required_df.difference(df.columns)
    if missing_df:
        raise ValueError(f"Missing OHLC columns: {sorted(missing_df)}")
    required_swings = {"timestamp", "price", "side"}
    missing_swings = required_swings.difference(swings.columns)
    if missing_swings:
        raise ValueError(f"Missing swing columns: {sorted(missing_swings)}")

    bars = df.reset_index()
    timestamp_col = df.index.name or "index"
    bars = bars.rename(columns={timestamp_col: "timestamp"})
    events: list[dict] = []

    for _, swing in swings.iterrows():
        level = float(swing["price"])
        side = str(swing["side"])
        if side in {"high", "BSL"}:
            direction = "bullish"
            accepted = bars["close"] > level
        elif side in {"low", "SSL"}:
            direction = "bearish"
            accepted = bars["close"] < level
        else:
            raise ValueError(f"Unsupported swing side: {side}")

        candidates = bars.loc[bars["timestamp"] > swing["timestamp"]]
        if require_displacement and "is_displacement" in candidates:
            candidates = candidates.loc[candidates["is_displacement"]]

        for idx, bar in candidates.iterrows():
            if bool(accepted.loc[idx]):
                events.append(
                    {
                        "swing_timestamp": swing["timestamp"],
                        "break_timestamp": bar["timestamp"],
                        "level_price": level,
                        "direction": direction,
                        "break_type": "bullish_structure_break"
                        if direction == "bullish"
                        else "bearish_structure_break",
                    }
                )
                break

    return pd.DataFrame(
        events,
        columns=[
            "swing_timestamp", "break_timestamp", "level_price",
            "direction", "break_type",
        ],
    )
