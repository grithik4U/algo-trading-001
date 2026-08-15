"""Walk-forward validation utilities with chronological train/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import pandas as pd


T = TypeVar("T")


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def expanding_windows(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[WalkForwardWindow]:
    """Create expanding in-sample/out-of-sample windows without shuffling."""
    if n_samples <= 0 or train_size <= 0 or test_size <= 0:
        raise ValueError("sample and window sizes must be > 0")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be > 0")

    windows: list[WalkForwardWindow] = []
    test_start = train_size
    while test_start + test_size <= n_samples:
        windows.append(WalkForwardWindow(0, test_start, test_start, test_start + test_size))
        test_start += step
    return windows


def run_walk_forward(
    data: pd.DataFrame,
    windows: list[WalkForwardWindow],
    train_fn: Callable[[pd.DataFrame], T],
    test_fn: Callable[[pd.DataFrame, T], T],
) -> list[T]:
    """Fit only on each training window and evaluate only on its later test window."""
    results: list[T] = []
    for window in windows:
        train = data.iloc[window.train_start:window.train_end]
        test = data.iloc[window.test_start:window.test_end]
        model = train_fn(train)
        results.append(test_fn(test, model))
    return results
