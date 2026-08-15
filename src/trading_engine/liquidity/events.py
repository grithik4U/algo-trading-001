"""Causal liquidity-pool event detection.

A pool is considered swept when price trades through it and later closes back
through the level. If price remains beyond the level instead, the event is
classified as acceptance/breakout. The detector emits events only after the
classification bar has closed, avoiding look-ahead in downstream research.
"""

from __future__ import annotations

import pandas as pd


def detect_pool_events(
    df: pd.DataFrame,
    pools: pd.DataFrame,
    max_bars: int = 20,
) -> pd.DataFrame:
    """Detect sweeps and breakouts of candidate liquidity pools.

    Required OHLC columns: ``high``, ``low``, ``close``.
    Required pool columns: ``pool_id``, ``side``, ``price``.

    For BSL, a breach occurs when ``high > price``. A later close below the
    level is a sweep; a close above it is acceptance. SSL is mirrored.
    ``event_timestamp`` is the classification timestamp, not the original
    pool timestamp.
    """
    if max_bars < 1:
        raise ValueError("max_bars must be >= 1")

    required_df = {"high", "low", "close"}
    missing_df = required_df.difference(df.columns)
    if missing_df:
        raise ValueError(f"Missing OHLC columns: {sorted(missing_df)}")

    required_pool = {"pool_id", "side", "price"}
    missing_pool = required_pool.difference(pools.columns)
    if missing_pool:
        raise ValueError(f"Missing pool columns: {sorted(missing_pool)}")

    events: list[dict] = []
    bars = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})

    for _, pool in pools.iterrows():
        level = float(pool["price"])
        side = str(pool["side"])
        pool_id = pool["pool_id"]

        for i, bar in bars.iterrows():
            breached = (
                float(bar["high"]) > level if side == "BSL"
                else float(bar["low"]) < level
            )
            if not breached:
                continue

            end = min(i + max_bars + 1, len(bars))
            future = bars.iloc[i:end]
            classification = None
            event_row = None

            for _, candidate in future.iterrows():
                close = float(candidate["close"])
                if side == "BSL":
                    if close < level:
                        classification = "sweep"
                        event_row = candidate
                        break
                    if close > level:
                        classification = "breakout"
                        event_row = candidate
                        break
                else:
                    if close > level:
                        classification = "sweep"
                        event_row = candidate
                        break
                    if close < level:
                        classification = "breakout"
                        event_row = candidate
                        break

            if classification is None:
                continue

            excursion = (
                float(bar["high"]) - level if side == "BSL"
                else level - float(bar["low"])
            )
            events.append(
                {
                    "pool_id": pool_id,
                    "side": side,
                    "level_price": level,
                    "breach_timestamp": bar["timestamp"],
                    "event_timestamp": event_row["timestamp"],
                    "event_type": classification,
                    "excursion": excursion,
                    "bars_to_event": int(
                        event_row.name - bar.name
                    ),
                }
            )
            break

    return pd.DataFrame(
        events,
        columns=[
            "pool_id", "side", "level_price", "breach_timestamp",
            "event_timestamp", "event_type", "excursion", "bars_to_event",
        ],
    )
