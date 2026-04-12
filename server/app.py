from __future__ import annotations

from copy import deepcopy

from fastapi import Body
from fastapi.routing import APIRoute

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to run the Fraud Ring Investigator Arena server. Install project dependencies first."
    ) from exc

try:
    from ..models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation
    from .environment import FraudRingInvestigatorArenaEnvironment, export_task_manifest
    from ..graders import GRADERS
except ImportError:
    from models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation
    from server.environment import FraudRingInvestigatorArenaEnvironment, export_task_manifest
    from graders import GRADERS

TASK_MANIFEST = export_task_manifest()

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
            "costly interventions, delayed payout consequences, and normalized episode scoring."
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


@app.get("/grader")
def list_graders() -> dict[str, object]:
    return {
        "environment": "fraud_ring_investigator_arena",
        "grader_count": len(GRADERS),
        "graders": sorted(GRADERS.keys()),
    }


@app.post("/grader")
def grade_episode(payload: dict[str, object] = Body(default_factory=dict)) -> dict[str, object]:
    task_id = str(payload.get("task_id") or "")
    state = payload.get("state") or {}
    reward = payload.get("reward")

    grader = GRADERS.get(task_id)
    if grader is None:
        return {"task_id": task_id, "score": 0.0, "error": "unknown_task"}

    score = float(grader(state, reward if isinstance(reward, (int, float)) else 0.0))
    return {"task_id": task_id, "score": score}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
