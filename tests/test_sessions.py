import pandas as pd

from trading_engine.structure.sessions import SessionDefinition, session_extremes


def test_completed_session_produces_high_and_low_levels():
    index = pd.date_range(
        "2026-01-02 08:00", periods=4, freq="h", tz="UTC"
    )
    df = pd.DataFrame(
        {"high": [100, 103, 101, 102], "low": [98, 99, 97, 100]},
        index=index,
    )

    sessions = (SessionDefinition("London", "08:00", "11:00", "UTC"),)
    levels = session_extremes(df, sessions)

    assert set(levels["side"]) == {"BSL", "SSL"}
    assert set(levels["price"]) == {103.0, 97.0}


def test_overnight_session_assigns_after_midnight_to_prior_session_date():
    index = pd.date_range(
        "2026-01-02 22:00", periods=5, freq="h", tz="UTC"
    )
    df = pd.DataFrame(
        {"high": [100, 102, 101, 104, 103], "low": [98, 99, 97, 100, 101]},
        index=index,
    )

    sessions = (SessionDefinition("Asia", "22:00", "02:00", "UTC"),)
    levels = session_extremes(df, sessions)

    assert len(levels) == 2
    assert levels.iloc[0]["session_date"].isoformat() == "2026-01-02"
    assert set(levels["price"]) == {102.0, 97.0}
