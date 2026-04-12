from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    from .simulator import HiddenWorld
except ImportError:
    from simulator import HiddenWorld


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def _clamp_strict_score(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _f1_score(predicted: set[str], truth: set[str]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class TerminalMetrics:
    prevented_loss_ratio: float
    suspect_f1: float
    intervention_precision: float
    disposition_correct: float
    benign_harm_ratio: float
    cost_ratio: float
    false_suspect_ratio: float
    blocked_fraud_value_realized: float
    blocked_fraud_value_pending: float
    escaped_fraud_value_realized: float
    fraud_value_still_at_risk: float
    episode_score: float

    def as_metadata(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def compute_terminal_metrics(world: HiddenWorld) -> TerminalMetrics:
    blocked_fraud_value_realized = sum(
        payout.amount
        for payout in world.payouts.values()
        if payout.fraudulent and payout.status == "blocked_realized"
    )
    blocked_fraud_value_pending = sum(
        payout.amount
        for payout in world.payouts.values()
        if payout.fraudulent and payout.status == "blocked_pending"
    )
    escaped_fraud_value_realized = sum(
        payout.amount
        for payout in world.payouts.values()
        if payout.fraudulent and payout.status == "escaped"
    )
    total_fraud_value = sum(
        payout.amount for payout in world.payouts.values() if payout.fraudulent
    )
    fraud_value_still_at_risk = max(
        total_fraud_value
        - blocked_fraud_value_realized
        - blocked_fraud_value_pending
        - escaped_fraud_value_realized,
        0.0,
    )

    prevented_loss_ratio = (
        (blocked_fraud_value_realized + blocked_fraud_value_pending) / total_fraud_value
        if total_fraud_value > 0
        else 0.0
    )
    suspect_f1 = _f1_score(set(world.submitted_suspect_ids), set(world.ring_member_ids))

    total_interventions = len(world.interventions)
    correctly_targeted_interventions = sum(
        1 for intervention in world.interventions if intervention.correct
    )
    if world.case_truth == "benign" and total_interventions == 0:
        intervention_precision = 1.0
    else:
        intervention_precision = (
            correctly_targeted_interventions / max(total_interventions, 1)
        )

    decision = world.submitted_decision
    disposition_correct = float(
        (decision == "escalate" and world.case_truth == "fraud")
        or (decision == "clear" and world.case_truth == "benign")
    )

    benign_harm_ratio = _clamp(
        world.benign_harm_value_realized / max(world.benign_harm_normalizer, 1.0)
    )
    cost_ratio = _clamp(world.investigation_cost_used / max(world.cost_budget, 1))
    false_suspect_ratio = (
        _clamp(len(world.submitted_suspect_ids) / 4.0)
        if world.case_truth == "benign"
        else 0.0
    )

    if world.case_truth == "fraud":
        episode_score = _clamp_strict_score(
            0.55 * prevented_loss_ratio
            + 0.20 * suspect_f1
            + 0.10 * intervention_precision
            + 0.05 * disposition_correct
            + 0.10 * (1.0 - cost_ratio)
            - 0.20 * benign_harm_ratio
        )
    else:
        episode_score = _clamp_strict_score(
            0.55 * disposition_correct
            + 0.20 * (1.0 - benign_harm_ratio)
            + 0.15 * (1.0 - false_suspect_ratio)
            + 0.10 * (1.0 - cost_ratio)
        )

    return TerminalMetrics(
        prevented_loss_ratio=prevented_loss_ratio,
        suspect_f1=suspect_f1,
        intervention_precision=intervention_precision,
        disposition_correct=disposition_correct,
        benign_harm_ratio=benign_harm_ratio,
        cost_ratio=cost_ratio,
        false_suspect_ratio=false_suspect_ratio,
        blocked_fraud_value_realized=blocked_fraud_value_realized,
        blocked_fraud_value_pending=blocked_fraud_value_pending,
        escaped_fraud_value_realized=escaped_fraud_value_realized,
        fraud_value_still_at_risk=fraud_value_still_at_risk,
        episode_score=episode_score,
    )
