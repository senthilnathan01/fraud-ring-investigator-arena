from __future__ import annotations

from dataclasses import dataclass, field

try:
    from ..models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation
except ImportError:
    from models import FraudRingInvestigatorArenaAction, FraudRingInvestigatorArenaObservation


def _visible_actionable_ids(
    observation: FraudRingInvestigatorArenaObservation,
) -> list[str]:
    return [
        entity.entity_id
        for entity in observation.visible_entities
        if entity.entity_type in {"account", "merchant"}
    ]


def _seed_entity_id(observation: FraudRingInvestigatorArenaObservation) -> str | None:
    return (
        observation.seed_alert.seed_entity_ids[0]
        if observation.seed_alert.seed_entity_ids
        else None
    )


class BasePolicy:
    def act(
        self, observation: FraudRingInvestigatorArenaObservation
    ) -> FraudRingInvestigatorArenaAction:
        raise NotImplementedError


@dataclass
class SeedOnlyTriagePolicy(BasePolicy):
    inspected: bool = False

    def act(
        self, observation: FraudRingInvestigatorArenaObservation
    ) -> FraudRingInvestigatorArenaAction:
        seed_entity_id = _seed_entity_id(observation)
        if not self.inspected and seed_entity_id is not None:
            self.inspected = True
            return FraudRingInvestigatorArenaAction(
                action_type="inspect_entity",
                entity_id=seed_entity_id,
            )
        if (
            observation.seed_alert.severity_band == "high"
            and observation.visible_payouts
            and _visible_actionable_ids(observation)
        ):
            suspects = _visible_actionable_ids(observation)[:2]
            return FraudRingInvestigatorArenaAction(
                action_type="submit_case",
                decision="escalate",
                suspect_entity_ids=suspects,
            )
        return FraudRingInvestigatorArenaAction(
            action_type="submit_case",
            decision="clear",
            suspect_entity_ids=[],
        )


@dataclass
class FixedSequenceInvestigatorPolicy(BasePolicy):
    stage: int = 0
    chosen_relation: str | None = None
    traced: bool = False

    def act(
        self, observation: FraudRingInvestigatorArenaObservation
    ) -> FraudRingInvestigatorArenaAction:
        seed_entity_id = _seed_entity_id(observation)
        if seed_entity_id is None:
            visible = _visible_actionable_ids(observation)
            seed_entity_id = visible[0] if visible else None
        if seed_entity_id is None:
            return FraudRingInvestigatorArenaAction(
                action_type="submit_case",
                decision="clear",
                suspect_entity_ids=[],
            )
        if self.stage == 0:
            self.stage += 1
            return FraudRingInvestigatorArenaAction(
                action_type="inspect_entity",
                entity_id=seed_entity_id,
            )
        if self.stage == 1:
            self.stage += 1
            return FraudRingInvestigatorArenaAction(
                action_type="expand_links",
                entity_id=seed_entity_id,
                relation_type="transfers",
            )
        if self.stage == 2:
            self.stage += 1
            relation = (
                "shared_device"
                if "device" in observation.seed_alert.alert_type
                else "shared_identity"
                if "identity" in observation.seed_alert.alert_type
                else "merchant_interaction"
            )
            self.chosen_relation = relation
            return FraudRingInvestigatorArenaAction(
                action_type="expand_links",
                entity_id=seed_entity_id,
                relation_type=relation,
            )
        if self.stage == 3:
            self.stage += 1
            self.traced = True
            return FraudRingInvestigatorArenaAction(
                action_type="trace_funds",
                entity_id=seed_entity_id,
                direction="outgoing",
                depth=2,
            )
        for payout in observation.visible_payouts:
            if payout.amount is None:
                return FraudRingInvestigatorArenaAction(
                    action_type="inspect_payout",
                    payout_id=payout.payout_id,
                )
            if payout.status in {None, "pending"}:
                return FraudRingInvestigatorArenaAction(
                    action_type="hold_payout",
                    payout_id=payout.payout_id,
                )
        suspects = _visible_actionable_ids(observation)[:4]
        if suspects:
            return FraudRingInvestigatorArenaAction(
                action_type="submit_case",
                decision="escalate",
                suspect_entity_ids=suspects,
            )
        return FraudRingInvestigatorArenaAction(
            action_type="submit_case",
            decision="clear",
            suspect_entity_ids=[],
        )


@dataclass
class AggressiveFreezeFirstPolicy(BasePolicy):
    inspected: bool = False
    acted: bool = False

    def act(
        self, observation: FraudRingInvestigatorArenaObservation
    ) -> FraudRingInvestigatorArenaAction:
        seed_entity_id = _seed_entity_id(observation)
        if seed_entity_id is None:
            visible = _visible_actionable_ids(observation)
            seed_entity_id = visible[0] if visible else None
        if not self.inspected and seed_entity_id is not None:
            self.inspected = True
            return FraudRingInvestigatorArenaAction(
                action_type="inspect_entity",
                entity_id=seed_entity_id,
            )
        if not self.acted:
            self.acted = True
            if observation.visible_payouts:
                return FraudRingInvestigatorArenaAction(
                    action_type="hold_payout",
                    payout_id=observation.visible_payouts[0].payout_id,
                )
            if seed_entity_id is not None:
                return FraudRingInvestigatorArenaAction(
                    action_type="freeze_entity",
                    entity_id=seed_entity_id,
                )
        suspects = _visible_actionable_ids(observation)[:4]
        if suspects:
            return FraudRingInvestigatorArenaAction(
                action_type="submit_case",
                decision="escalate",
                suspect_entity_ids=suspects,
            )
        return FraudRingInvestigatorArenaAction(
            action_type="submit_case",
            decision="clear",
            suspect_entity_ids=[],
        )
