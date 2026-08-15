"""Causal lifecycle tracking for fair-value gaps."""

from __future__ import annotations

import pandas as pd


def track_fvg_lifecycle(df: pd.DataFrame, fvgs: pd.DataFrame) -> pd.DataFrame:
    """Track each FVG from creation through first mitigation or invalidation.

    Bullish FVGs are progressively filled when price trades into the gap and
    are invalidated when price closes below the lower boundary. Bearish gaps
    mirror this logic. Classification is based only on bars after creation.
    """
    required_df = {"high", "low", "close"}
    missing_df = required_df.difference(df.columns)
    if missing_df:
        raise ValueError(f"Missing OHLC columns: {sorted(missing_df)}")
    required_fvg = {
        "created_timestamp", "direction", "lower_price", "upper_price"
    }
    missing_fvg = required_fvg.difference(fvgs.columns)
    if missing_fvg:
        raise ValueError(f"Missing FVG columns: {sorted(missing_fvg)}")

    bars = df.reset_index()
    timestamp_col = df.index.name or "index"
    bars = bars.rename(columns={timestamp_col: "timestamp"})
    rows: list[dict] = []

    for fvg_id, fvg in fvgs.reset_index(drop=True).iterrows():
        created = fvg["created_timestamp"]
        lower = float(fvg["lower_price"])
        upper = float(fvg["upper_price"])
        direction = str(fvg["direction"])
        future = bars.loc[bars["timestamp"] > created]

        state = "open"
        mitigation_timestamp = None
        invalidation_timestamp = None
        fill_fraction = 0.0

        for _, bar in future.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            width = max(upper - lower, 1e-12)

            if direction == "bullish":
                if low < upper:
                    fill_fraction = max(fill_fraction, min(1.0, (upper - max(low, lower)) / width))
                    if mitigation_timestamp is None:
                        mitigation_timestamp = bar["timestamp"]
                if close < lower:
                    state = "invalidated"
                    invalidation_timestamp = bar["timestamp"]
                    break
            elif direction == "bearish":
                if high > lower:
                    fill_fraction = max(fill_fraction, min(1.0, (min(high, upper) - lower) / width))
                    if mitigation_timestamp is None:
                        mitigation_timestamp = bar["timestamp"]
                if close > upper:
                    state = "invalidated"
                    invalidation_timestamp = bar["timestamp"]
                    break
            else:
                raise ValueError(f"Unsupported FVG direction: {direction}")

            if fill_fraction >= 1.0:
                state = "mitigated"
                break

        rows.append(
            {
                "fvg_id": fvg_id,
                "created_timestamp": created,
                "direction": direction,
                "lower_price": lower,
                "upper_price": upper,
                "state": state,
                "fill_fraction": fill_fraction,
                "mitigation_timestamp": mitigation_timestamp,
                "invalidation_timestamp": invalidation_timestamp,
            }
        )

    return pd.DataFrame(rows)
