from __future__ import annotations

from typing import Any

try:
    from .scoring import compute_terminal_metrics
except ImportError:
    from server.scoring import compute_terminal_metrics


def _clamp_grade(value: float | None) -> float:
    if value is None:
        return 0.5
    return max(0.01, min(0.99, float(value)))


def _extract_task_id(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        task_id = obj.get("task_id")
    else:
        task_id = getattr(obj, "task_id", None)
    return str(task_id) if task_id is not None else None


def _extract_reward(args: tuple[Any, ...], kwargs: dict[str, Any]) -> float | None:
    reward = kwargs.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    for arg in args:
        if isinstance(arg, (int, float)):
            return float(arg)
    return None


def _score_from_env(env: Any) -> float | None:
    if env is None:
        return None

    if isinstance(env, dict):
        if isinstance(env.get("episode_score"), (int, float)):
            return float(env["episode_score"])
        terminal_metrics = env.get("terminal_metrics")
        if isinstance(terminal_metrics, dict) and isinstance(
            terminal_metrics.get("episode_score"), (int, float)
        ):
            return float(terminal_metrics["episode_score"])
        metadata = env.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("episode_score"), (int, float)):
            return float(metadata["episode_score"])

    world = getattr(env, "_world", None)
    if world is not None:
        final_score = getattr(world, "final_score", None)
        if isinstance(final_score, (int, float)):
            return float(final_score)
        try:
            return float(compute_terminal_metrics(world).episode_score)
        except Exception:
            pass

    state = getattr(env, "state", None)
    if state is not None:
        metadata = getattr(state, "metadata", None)
        if isinstance(metadata, dict) and isinstance(metadata.get("episode_score"), (int, float)):
            return float(metadata["episode_score"])

    return None


def _grade(expected_task_id: str, env: Any, *args: Any, **kwargs: Any) -> float:
    try:
        actual_task_id = _extract_task_id(env) or _extract_task_id(getattr(env, "state", None))
        score = _score_from_env(env)
        if score is None:
            score = _extract_reward(args, kwargs)
        if score is None:
            score = 0.5
        if actual_task_id is not None and actual_task_id != expected_task_id:
            score = min(float(score), 0.5)
        return _clamp_grade(score)
    except Exception:
        return 0.5


class EasyGrader:
    def __call__(self, env: Any = None, *args: Any, **kwargs: Any) -> float:
        return self.grade(env, *args, **kwargs)

    def grade(self, env: Any = None, *args: Any, **kwargs: Any) -> float:
        return _grade("easy", env, *args, **kwargs)


class MediumGrader:
    def __call__(self, env: Any = None, *args: Any, **kwargs: Any) -> float:
        return self.grade(env, *args, **kwargs)

    def grade(self, env: Any = None, *args: Any, **kwargs: Any) -> float:
        return _grade("medium", env, *args, **kwargs)


class HardGrader:
    def __call__(self, env: Any = None, *args: Any, **kwargs: Any) -> float:
        return self.grade(env, *args, **kwargs)

    def grade(self, env: Any = None, *args: Any, **kwargs: Any) -> float:
        return _grade("hard", env, *args, **kwargs)


GRADERS = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}


TASK_GRADER_PAIRS = [
    ("easy", EasyGrader),
    ("medium", MediumGrader),
    ("hard", HardGrader),
]


__all__ = [
    "EasyGrader",
    "MediumGrader",
    "HardGrader",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]
