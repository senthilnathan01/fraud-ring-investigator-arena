from __future__ import annotations

TASKS = [
    {
        "id": "easy",
        "track_id": "easy_single_ring_v1",
        "name": "Easy Single Ring V1",
        "difficulty": "easy",
        "description": "Compact single-ring cases with clearer payout pressure and fewer benign confounders.",
        "max_steps": 8,
        "weight": 0.3,
        "reward_range": [0.0, 1.0],
        "grader": "server.graders:EasyGrader",
        "prompt": "Investigate a compact alert-driven case with a clearer single-ring fraud or benign lookalike pattern, then decide whether to clear or escalate before the first payout wave settles.",
        "reward_definition": "Step penalties for investigation actions plus terminal score driven by prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition correctness, and cost_ratio.",
    },
    {
        "id": "medium",
        "track_id": "medium_confounded_ring_v1",
        "name": "Medium Confounded Ring V1",
        "difficulty": "medium",
        "description": "Noisier local cases with additional benign confounders and one or two payout waves.",
        "max_steps": 10,
        "weight": 0.5,
        "reward_range": [0.0, 1.0],
        "grader": "server.graders:MediumGrader",
        "prompt": "Investigate a noisier local case with additional benign confounders and one or two payout waves, using sequential tool calls and interventions to decide whether to clear or escalate.",
        "reward_definition": "Step penalties for investigation actions plus terminal score driven by prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition correctness, and cost_ratio.",
    },
    {
        "id": "hard",
        "track_id": "hard_reserve_ring_v1",
        "name": "Hard Reserve Ring V1",
        "difficulty": "hard",
        "description": "Harder cases with deeper hidden structure, more confounders, and possible reserve-route behavior.",
        "max_steps": 12,
        "weight": 1.0,
        "reward_range": [0.0, 1.0],
        "grader": "server.graders:HardGrader",
        "prompt": "Investigate a harder case with deeper hidden structure, more confounders, and possible reserve-route behavior that punishes premature intervention, then submit a final clear or escalate decision.",
        "reward_definition": "Step penalties for investigation actions plus terminal score driven by prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition correctness, and cost_ratio.",
    },
]


TASKS_BY_ID = {task["id"]: dict(task) for task in TASKS}
TRACK_TO_TASK_ID = {task["track_id"]: task["id"] for task in TASKS}


__all__ = [
    "TASKS",
    "TASKS_BY_ID",
    "TRACK_TO_TASK_ID",
]
