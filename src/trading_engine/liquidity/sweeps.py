"""Causal liquidity-sweep detection."""

from __future__ import annotations

import pandas as pd


def detect_sweeps(
    df: pd.DataFrame,
    levels: pd.DataFrame,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Detect breaches followed by same-bar reclaim of known liquidity levels.

    A level becomes eligible only at ``confirmed_at``. This prevents a
    backtest from trading against structural information that was not known
    at the time.

    BSL sweep: high trades above the level and close returns at/below it.
    SSL sweep: low trades below the level and close returns at/above it.
    """
    required = {"high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    required_levels = {"timestamp", "price", "side", "confirmed_at"}
    missing_levels = required_levels.difference(levels.columns)
    if missing_levels:
        raise ValueError(f"Missing level columns: {sorted(missing_levels)}")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    records: list[dict] = []
    for _, level in levels.iterrows():
        confirmed_at = level["confirmed_at"]
        if pd.isna(confirmed_at):
            continue
        eligible = df.loc[df.index >= confirmed_at]
        if eligible.empty:
            continue

        price = float(level["price"])
        side = level["side"]
        for timestamp, bar in eligible.iterrows():
            if side == "BSL":
                breached = float(bar["high"]) > price + tolerance
                reclaimed = float(bar["close"]) <= price + tolerance
                depth = float(bar["high"]) - price
            elif side == "SSL":
                breached = float(bar["low"]) < price - tolerance
                reclaimed = float(bar["close"]) >= price - tolerance
                depth = price - float(bar["low"])
            else:
                raise ValueError(f"Unsupported side: {side}")

            if breached and reclaimed:
                records.append(
                    {
                        "timestamp": timestamp,
                        "level_timestamp": level["timestamp"],
                        "price": price,
                        "side": side,
                        "scope": level.get("scope"),
                        "strength": level.get("strength"),
                        "sweep_depth": depth,
                        "reclaimed": True,
                    }
                )
                break

    return pd.DataFrame(
        records,
        columns=[
            "timestamp",
            "level_timestamp",
            "price",
            "side",
            "scope",
            "strength",
            "sweep_depth",
            "reclaimed",
        ],
    )
