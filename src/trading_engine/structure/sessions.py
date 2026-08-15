"""Session-aware structural levels.

Session boundaries are explicit configuration. Timestamps are expected to be
UTC-aware; the session timezone controls which local trading date a bar belongs
to. The engine records completed session highs/lows so downstream strategies
can use prior-session levels without look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class SessionDefinition:
    name: str
    start: str  # HH:MM local time
    end: str  # HH:MM local time
    timezone: str = "UTC"


def _local_minutes(index: pd.DatetimeIndex, timezone: str) -> pd.Series:
    if index.tz is None:
        raise ValueError("Datetime index must be timezone-aware")
    local = index.tz_convert(ZoneInfo(timezone))
    return pd.Series(local.hour * 60 + local.minute, index=index)


def session_extremes(
    df: pd.DataFrame,
    sessions: tuple[SessionDefinition, ...],
) -> pd.DataFrame:
    """Return completed-session highs/lows as structural liquidity levels.

    Required columns: high, low. The dataframe index must be timezone-aware.
    Overnight sessions (end earlier than start) are supported.
    """
    required = {"high", "low"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if df.index.tz is None:
        raise ValueError("Datetime index must be timezone-aware")

    rows: list[dict] = []
    for definition in sessions:
        start_h, start_m = map(int, definition.start.split(":"))
        end_h, end_m = map(int, definition.end.split(":"))
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        minutes = _local_minutes(df.index, definition.timezone)
        dates = pd.Series(
            df.index.tz_convert(ZoneInfo(definition.timezone)).date,
            index=df.index,
        )

        if end > start:
            mask = (minutes >= start) & (minutes < end)
            session_dates = dates
        elif end < start:
            mask = (minutes >= start) | (minutes < end)
            # Bars after midnight belong to the session that started the prior date.
            session_dates = dates.where(minutes >= start, dates - pd.Timedelta(days=1))
        else:
            raise ValueError(f"Session {definition.name} cannot have equal start/end")

        grouped = df.loc[mask].groupby(session_dates.loc[mask])
        for session_date, group in grouped:
            if group.empty:
                continue
            session_high = float(group["high"].max())
            session_low = float(group["low"].min())
            last_ts = group.index[-1]
            rows.extend(
                [
                    {
                        "timestamp": last_ts,
                        "session_date": session_date,
                        "session": definition.name,
                        "price": session_high,
                        "side": "BSL",
                        "timeframe": "session",
                    },
                    {
                        "timestamp": last_ts,
                        "session_date": session_date,
                        "session": definition.name,
                        "price": session_low,
                        "side": "SSL",
                        "timeframe": "session",
                    },
                ]
            )

    return pd.DataFrame(
        rows,
        columns=[
            "timestamp", "session_date", "session", "price", "side", "timeframe"
        ],
    ).sort_values("timestamp", ignore_index=True)
