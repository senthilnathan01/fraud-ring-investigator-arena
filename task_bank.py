from __future__ import annotations


TASK_MANIFEST: list[dict[str, str]] = [
    {
        "id": "easy_single_ring_v1",
        "name": "Easy Single Ring V1",
        "prompt": (
            "Investigate a compact alert-driven case with a clearer single-ring fraud "
            "or benign lookalike pattern, then decide whether to clear or escalate "
            "before the first payout wave settles."
        ),
        "grader": (
            "Deterministic episode scoring over prevented fraud loss, benign harm, "
            "suspect set quality, final disposition correctness, and investigation cost."
        ),
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
    {
        "id": "medium_confounded_ring_v1",
        "name": "Medium Confounded Ring V1",
        "prompt": (
            "Investigate a noisier local case with additional benign confounders and "
            "one or two payout waves, using sequential tool calls and interventions "
            "to decide whether to clear or escalate."
        ),
        "grader": (
            "Deterministic episode scoring over prevented fraud loss, benign harm, "
            "suspect set quality, final disposition correctness, and investigation cost."
        ),
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
    {
        "id": "hard_reserve_ring_v1",
        "name": "Hard Reserve Ring V1",
        "prompt": (
            "Investigate a harder case with deeper hidden structure, more confounders, "
            "and possible reserve-route behavior that punishes premature intervention, "
            "then submit a final clear or escalate decision."
        ),
        "grader": (
            "Deterministic episode scoring over prevented fraud loss, benign harm, "
            "suspect set quality, final disposition correctness, and investigation cost."
        ),
        "reward_definition": (
            "Step penalties for investigation actions plus terminal score driven by "
            "prevented_loss_ratio, benign_harm_ratio, suspect_f1, disposition "
            "correctness, and cost_ratio."
        ),
    },
]


def list_task_ids() -> list[str]:
    return [task["id"] for task in TASK_MANIFEST]


def export_manifest_tasks() -> list[dict[str, str]]:
    return [task.copy() for task in TASK_MANIFEST]


def get_task_prompt(task_id: str) -> str:
    for task in TASK_MANIFEST:
        if task["id"] == task_id:
            return task["prompt"]
    raise KeyError(f"Unknown task_id: {task_id}")
