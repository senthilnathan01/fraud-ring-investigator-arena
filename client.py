from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    FraudRingInvestigatorArenaAction,
    FraudRingInvestigatorArenaObservation,
    FraudRingInvestigatorArenaState,
)


class FraudRingInvestigatorArenaEnv(
    EnvClient[
        FraudRingInvestigatorArenaAction,
        FraudRingInvestigatorArenaObservation,
        FraudRingInvestigatorArenaState,
    ]
):
    def _step_payload(self, action: FraudRingInvestigatorArenaAction) -> dict[str, Any]:
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[FraudRingInvestigatorArenaObservation]:
        obs_data = payload.get("observation", {})
        observation = FraudRingInvestigatorArenaObservation(
            **obs_data,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> FraudRingInvestigatorArenaState:
        return FraudRingInvestigatorArenaState(**payload)
