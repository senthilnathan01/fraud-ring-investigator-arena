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

app = create_app(
    FraudRingInvestigatorArenaEnvironment,
    FraudRingInvestigatorArenaAction,
    FraudRingInvestigatorArenaObservation,
    env_name="fraud_ring_investigator_arena",
    max_concurrent_envs=32,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
