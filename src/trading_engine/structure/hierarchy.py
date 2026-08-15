"""Multi-timeframe structural hierarchy and liquidity classification."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StructureRule:
    """Configuration for a structural timeframe."""

    timeframe: str
    significance: float


DEFAULT_RULES = (
    StructureRule("1W", 1.00),
    StructureRule("1D", 0.90),
    StructureRule("4H", 0.80),
    StructureRule("1H", 0.65),
    StructureRule("15min", 0.45),
    StructureRule("5min", 0.30),
    StructureRule("1min", 0.15),
)


def classify_structural_levels(
    levels: pd.DataFrame,
    rules: tuple[StructureRule, ...] = DEFAULT_RULES,
    external_threshold: float = 0.65,
) -> pd.DataFrame:
    """Assign a structural significance score and internal/external label.

    Required columns: ``timestamp``, ``price``, ``side``, ``timeframe``.
    The default mapping is a research starting point, not a claim that these
    exact thresholds are universally optimal.
    """
    if not 0 <= external_threshold <= 1:
        raise ValueError("external_threshold must be between 0 and 1")

    required = {"timestamp", "price", "side", "timeframe"}
    missing = required.difference(levels.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    weights = {rule.timeframe: rule.significance for rule in rules}
    result = levels.copy()
    result["structural_significance"] = result["timeframe"].map(weights).fillna(0.0)
    result["liquidity_class"] = result["structural_significance"].map(
        lambda value: "external" if value >= external_threshold else "internal"
    )
    result["liquidity_side"] = result["side"].map(
        {"high": "BSL", "low": "SSL", "BSL": "BSL", "SSL": "SSL"}
    )
    return result
