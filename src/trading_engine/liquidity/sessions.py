"""Session high/low liquidity levels.

Session definitions are explicit and timezone-aware. No market-specific
session assumptions are hard-coded into the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd


@dataclass(frozen=True)
class SessionDefinition:
    name: str
    start: time
    end: time


class SessionLevelEngine:
    """Build completed-session high/low levels from OHLC data.

    The input index must be a timezone-aware ``DatetimeIndex``. Session
    boundaries are interpreted in the supplied timezone. Overnight sessions
    are supported when ``end < start``.
    """

    def __init__(self, timezone: str, sessions: list[SessionDefinition]) -> None:
        if not sessions:
            raise ValueError("At least one session definition is required")
        self.timezone = timezone
        self.sessions = sessions

    def levels(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"high", "low"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")
        if df.index.tz is None:
            raise ValueError("DataFrame index must be timezone-aware")

        local = df.tz_convert(self.timezone)
        records: list[dict] = []

        for definition in self.sessions:
            session_key = self._session_key(local.index, definition)
            work = local.copy()
            work["_session_key"] = session_key
            for key, group in work.groupby("_session_key", sort=True):
                if key is None or group.empty:
                    continue
                records.extend(
                    [
                        {
                            "session": definition.name,
                            "session_date": key,
                            "side": "BSL",
                            "price": float(group["high"].max()),
                            "start": group.index.min(),
                            "end": group.index.max(),
                        },
                        {
                            "session": definition.name,
                            "session_date": key,
                            "side": "SSL",
                            "price": float(group["low"].min()),
                            "start": group.index.min(),
                            "end": group.index.max(),
                        },
                    ]
                )

        return pd.DataFrame(
            records,
            columns=["session", "session_date", "side", "price", "start", "end"],
        )

    @staticmethod
    def _session_key(index: pd.DatetimeIndex, definition: SessionDefinition) -> pd.Series:
        dates = index.date
        clock = index.time
        overnight = definition.end < definition.start
        values: list[object] = []
        for date, current in zip(dates, clock):
            if overnight:
                in_session = current >= definition.start or current < definition.end
                session_date = date if current >= definition.start else date - pd.Timedelta(days=1)
            else:
                in_session = definition.start <= current < definition.end
                session_date = date
            values.append(session_date if in_session else None)
        return pd.Series(values, index=index, dtype="object")
