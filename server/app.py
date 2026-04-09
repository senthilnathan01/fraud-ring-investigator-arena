from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to run the Fraud Ring Investigator Arena server. Install project dependencies first."
    ) from exc

try:
    from ..models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation
    from .fraud_ring_investigator_arena_environment import (
        FraudRingInvestigatorArenaEnvironment,
    )
except ImportError:
    from models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation
    from server.fraud_ring_investigator_arena_environment import (
        FraudRingInvestigatorArenaEnvironment,
    )

TASK_MANIFEST = [
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

app = create_app(
    FraudRingInvestigatorArenaEnvironment,
    FraudRingInvestigatorArenaAction,
    FraudRingInvestigatorArenaObservation,
    env_name="fraud_ring_investigator_arena",
    max_concurrent_envs=32,
)


@app.get("/tasks")
def list_tasks() -> dict[str, object]:
    return {
        "environment": "fraud_ring_investigator_arena",
        "tasks": TASK_MANIFEST,
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
