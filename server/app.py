from __future__ import annotations

from copy import deepcopy

from fastapi.routing import APIRoute

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
    {
        "id": "medium_confounded_ring_v1",
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
    {
        "id": "hard_reserve_ring_v1",
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
]

app = create_app(
    FraudRingInvestigatorArenaEnvironment,
    FraudRingInvestigatorArenaAction,
    FraudRingInvestigatorArenaObservation,
    env_name="fraud_ring_investigator_arena",
    max_concurrent_envs=32,
)

_generated_openapi = app.openapi


def custom_openapi() -> dict[str, object]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = deepcopy(_generated_openapi())
    reset_request = openapi_schema.get("components", {}).get("schemas", {}).get("ResetRequest")
    if isinstance(reset_request, dict):
        properties = reset_request.setdefault("properties", {})
        task_ids = [task["id"] for task in TASK_MANIFEST]
        properties["task_id"] = {
            "type": "string",
            "title": "Task Id",
            "description": "Optional task identifier selecting one of the declared benchmark tracks.",
            "enum": task_ids,
        }
        properties["task_name"] = {
            "type": "string",
            "title": "Task Name",
            "description": "Alias for task_id.",
            "enum": task_ids,
        }
        reset_request["examples"] = [
            {"task_id": task_id, "seed": 42}
            for task_id in task_ids
        ] + [
            {"episode_id": "episode-001", "seed": 42},
            {},
        ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def _remove_get_route(path: str) -> None:
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            isinstance(route, APIRoute)
            and route.path == path
            and route.methods is not None
            and "GET" in route.methods
        )
    ]


_remove_get_route("/health")
_remove_get_route("/metadata")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "healthy", "task_count": len(TASK_MANIFEST)}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "name": "fraud_ring_investigator_arena",
        "description": (
            "A sequential fraud investigation environment with partial observability, "
            "costly interventions, delayed payout consequences, and deterministic scoring."
        ),
        "version": "0.1.0",
        "task_count": len(TASK_MANIFEST),
        "tasks": TASK_MANIFEST,
    }


@app.get("/tasks")
def list_tasks() -> dict[str, object]:
    return {
        "environment": "fraud_ring_investigator_arena",
        "task_count": len(TASK_MANIFEST),
        "tasks": TASK_MANIFEST,
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
