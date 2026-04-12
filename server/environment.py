from __future__ import annotations

from typing import Any

try:
    from ..tasks import TASKS as ROOT_TASKS
except ImportError:
    from tasks import TASKS as ROOT_TASKS

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

TASKS: dict[str, dict[str, Any]] = {task["id"]: dict(task) for task in ROOT_TASKS}
TASKS["easy"]["baseline_score"] = 0.6235
TASKS["medium"]["baseline_score"] = 0.5995
TASKS["hard"]["baseline_score"] = 0.4573

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
