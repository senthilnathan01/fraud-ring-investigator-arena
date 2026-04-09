from __future__ import annotations

from typing import Any

try:
    from .fraud_ring_investigator_arena_environment import (
        DEFAULT_TASK_ID,
        FraudRingInvestigatorArenaEnvironment,
        grade_easy_single_ring_v1,
        grade_hard_reserve_ring_v1,
        grade_medium_confounded_ring_v1,
    )
except ImportError:
    from server.fraud_ring_investigator_arena_environment import (
        DEFAULT_TASK_ID,
        FraudRingInvestigatorArenaEnvironment,
        grade_easy_single_ring_v1,
        grade_hard_reserve_ring_v1,
        grade_medium_confounded_ring_v1,
    )


TASKS: dict[str, dict[str, Any]] = {
    "task1": {
        "id": "task1",
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
        "grader": "server.environment:grade_easy",
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
    "task2": {
        "id": "task2",
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
        "grader": "server.environment:grade_medium",
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
    "task3": {
        "id": "task3",
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
        "grader": "server.environment:grade_hard",
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
}

TRACK_TO_TASK_ID = {task["track_id"]: task_id for task_id, task in TASKS.items()}


def grade_easy(*args: Any, **kwargs: Any) -> float:
    return grade_easy_single_ring_v1(*args, **kwargs)


def grade_medium(*args: Any, **kwargs: Any) -> float:
    return grade_medium_confounded_ring_v1(*args, **kwargs)


def grade_hard(*args: Any, **kwargs: Any) -> float:
    return grade_hard_reserve_ring_v1(*args, **kwargs)


GRADERS = {
    "task1": grade_easy,
    "task2": grade_medium,
    "task3": grade_hard,
}


def export_task_manifest() -> list[dict[str, Any]]:
    return [dict(task) for task in TASKS.values()]


__all__ = [
    "DEFAULT_TASK_ID",
    "FraudRingInvestigatorArenaEnvironment",
    "GRADERS",
    "TASKS",
    "TRACK_TO_TASK_ID",
    "export_task_manifest",
    "grade_easy",
    "grade_medium",
    "grade_hard",
]
