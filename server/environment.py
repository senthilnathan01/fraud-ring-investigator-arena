from __future__ import annotations

from typing import Any

try:
    from .fraud_ring_investigator_arena_environment import (
        DEFAULT_TASK_ID,
        FraudRingInvestigatorArenaEnvironment,
    )
except ImportError:
    from server.fraud_ring_investigator_arena_environment import (
        DEFAULT_TASK_ID,
        FraudRingInvestigatorArenaEnvironment,
    )


TASKS: dict[str, dict[str, Any]] = {
    "easy": {
        "id": "easy",
        "track_id": "easy_single_ring_v1",
        "name": "Easy Single Ring V1",
        "difficulty": "easy",
        "description": (
            "Compact single-ring cases with clearer payout pressure and fewer benign confounders."
        ),
        "max_steps": 8,
        "reward_range": [0.0, 1.0],
        "baseline_score": 0.6235,
        "prompt": (
            "Investigate a compact alert-driven case with a clearer single-ring fraud "
            "or benign lookalike pattern, then decide whether to clear or escalate "
            "before the first payout wave settles."
        ),
        "grader": {
            "type": "llm",
            "prompt_template": (
                "Score the fraud investigation from 0.0 to 1.0 for this easy task. "
                "Reward prevented fraud loss, correct suspect identification, and the "
                "correct final clear-or-escalate disposition. Penalize benign harm "
                "and unnecessary investigation cost."
            ),
        },
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
    "medium": {
        "id": "medium",
        "track_id": "medium_confounded_ring_v1",
        "name": "Medium Confounded Ring V1",
        "difficulty": "medium",
        "description": (
            "Noisier local cases with additional benign confounders and one or two payout waves."
        ),
        "max_steps": 10,
        "reward_range": [0.0, 1.0],
        "baseline_score": 0.5995,
        "prompt": (
            "Investigate a noisier local case with additional benign confounders and "
            "one or two payout waves, using sequential tool calls and interventions "
            "to decide whether to clear or escalate."
        ),
        "grader": {
            "type": "llm",
            "prompt_template": (
                "Score the fraud investigation from 0.0 to 1.0 for this medium task. "
                "Reward prevented fraud loss, correct suspect identification, and the "
                "correct final clear-or-escalate disposition. Penalize benign harm, "
                "reckless intervention, and unnecessary investigation cost."
            ),
        },
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
    "hard": {
        "id": "hard",
        "track_id": "hard_reserve_ring_v1",
        "name": "Hard Reserve Ring V1",
        "difficulty": "hard",
        "description": (
            "Harder cases with deeper hidden structure, more confounders, and possible reserve-route behavior."
        ),
        "max_steps": 12,
        "reward_range": [0.0, 1.0],
        "baseline_score": 0.4573,
        "prompt": (
            "Investigate a harder case with deeper hidden structure, more confounders, "
            "and possible reserve-route behavior that punishes premature intervention, "
            "then submit a final clear or escalate decision."
        ),
        "grader": {
            "type": "llm",
            "prompt_template": (
                "Score the fraud investigation from 0.0 to 1.0 for this hard task. "
                "Reward prevented fraud loss, correct suspect identification, and the "
                "correct final clear-or-escalate disposition. Penalize benign harm, "
                "premature intervention, reserve-route misses, and unnecessary "
                "investigation cost."
            ),
        },
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
}

TRACK_TO_TASK_ID = {task["track_id"]: task_id for task_id, task in TASKS.items()}


def export_task_manifest() -> list[dict[str, Any]]:
    return [dict(task) for task in TASKS.values()]


__all__ = [
    "DEFAULT_TASK_ID",
    "FraudRingInvestigatorArenaEnvironment",
    "TASKS",
    "TRACK_TO_TASK_ID",
    "export_task_manifest",
]
