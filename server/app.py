from __future__ import annotations

from fastapi.routing import APIRoute

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to run the Fraud Ring Investigator Arena server. Install project dependencies first."
    ) from exc

try:
    from ..models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation
    from ..task_bank import export_manifest_tasks, list_task_ids
    from .fraud_ring_investigator_arena_environment import (
        FraudRingInvestigatorArenaEnvironment,
    )
except ImportError:
    from models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation
    from task_bank import export_manifest_tasks, list_task_ids
    from server.fraud_ring_investigator_arena_environment import (
        FraudRingInvestigatorArenaEnvironment,
    )

app = create_app(
    FraudRingInvestigatorArenaEnvironment,
    FraudRingInvestigatorArenaAction,
    FraudRingInvestigatorArenaObservation,
    env_name="fraud_ring_investigator_arena",
    max_concurrent_envs=32,
)


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
    return {"status": "healthy", "task_count": len(list_task_ids())}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "name": "fraud_ring_investigator_arena",
        "description": (
            "A sequential fraud investigation environment with partial observability, "
            "costly interventions, delayed payout consequences, and deterministic scoring."
        ),
        "version": "0.1.0",
        "task_count": len(list_task_ids()),
        "tasks": export_manifest_tasks(),
    }


@app.get("/tasks")
def list_tasks() -> dict[str, object]:
    return {
        "environment": "fraud_ring_investigator_arena",
        "task_count": len(list_task_ids()),
        "tasks": export_manifest_tasks(),
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
