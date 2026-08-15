"""Evidence-based setup confluence without arbitrary additive scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SetupEvidence:
    timestamp: datetime
    direction: str
    liquidity_sweep: bool
    reclaimed: bool
    displacement: bool
    mss: bool
    linked_fvg: bool
    orderflow_confirmed: bool
    profile_location: str
    profile_confluence: bool


@dataclass(frozen=True)
class SetupDecision:
    timestamp: datetime
    direction: str
    status: str
    reasons: tuple[str, ...]


def evaluate_setup(evidence: SetupEvidence) -> SetupDecision:
    """Apply structural prerequisites and evidence gates.

    This deliberately avoids a fixed point score. A setup is actionable only
    when the causal chain is intact; profile and order-flow features provide
    confirmation/context rather than compensating for missing structure.
    """
    reasons: list[str] = []

    if not evidence.liquidity_sweep:
        return SetupDecision(evidence.timestamp, evidence.direction, "rejected", ("no_liquidity_sweep",))
    if not evidence.reclaimed:
        return SetupDecision(evidence.timestamp, evidence.direction, "rejected", ("no_reclaim",))
    if not evidence.displacement:
        return SetupDecision(evidence.timestamp, evidence.direction, "rejected", ("no_displacement",))
    if not evidence.mss:
        return SetupDecision(evidence.timestamp, evidence.direction, "rejected", ("no_mss",))
    if not evidence.linked_fvg:
        return SetupDecision(evidence.timestamp, evidence.direction, "rejected", ("no_linked_fvg",))

    if evidence.orderflow_confirmed:
        reasons.append("orderflow_confirmed")
    if evidence.profile_confluence:
        reasons.append(f"profile_{evidence.profile_location}")

    if evidence.orderflow_confirmed or evidence.profile_confluence:
        return SetupDecision(evidence.timestamp, evidence.direction, "qualified", tuple(reasons))
    return SetupDecision(evidence.timestamp, evidence.direction, "watch", ("structure_complete_but_confirmation_missing",))
