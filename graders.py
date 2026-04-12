from __future__ import annotations

from typing import Any


def _normalize_reward(reward: float | None) -> float:
    if reward is None:
        return 0.0
    return min(max(float(reward), 0.0), 1.0)


def _extract_task_id(state: Any) -> str | None:
    if isinstance(state, dict):
        task_id = state.get("task_id")
    else:
        task_id = getattr(state, "task_id", None)
    return str(task_id) if task_id is not None else None


def _grade_for_task(expected_task_id: str, state: Any, reward: float | None) -> float:
    actual_task_id = _extract_task_id(state)
    if actual_task_id != expected_task_id:
        return 0.0
    return _normalize_reward(reward)


def grade_easy(state: Any, reward: float | None) -> float:
    return _grade_for_task("easy", state, reward)


def grade_medium(state: Any, reward: float | None) -> float:
    return _grade_for_task("medium", state, reward)


def grade_hard(state: Any, reward: float | None) -> float:
    return _grade_for_task("hard", state, reward)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


TASK_GRADER_PAIRS = [
    ("easy", grade_easy),
    ("medium", grade_medium),
    ("hard", grade_hard),
]


__all__ = [
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]
