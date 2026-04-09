from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


EntityType = Literal["account", "merchant", "device", "identity"]
RelationType = Literal[
    "transfers",
    "shared_device",
    "shared_identity",
    "merchant_interaction",
]
DirectionType = Literal["incoming", "outgoing", "both"]
DecisionType = Literal["clear", "escalate"]
InterventionType = Literal["hold_payout", "freeze_entity"]


class SeedAlert(BaseModel):
    alert_id: str
    alert_type: str
    severity_band: str
    narrative: str
    seed_entity_ids: list[str] = Field(default_factory=list)
    seed_payout_id: str | None = None


class VisibleEntity(BaseModel):
    entity_id: str
    entity_type: EntityType
    display_name: str
    risk_flags: list[str] = Field(default_factory=list)
    activity_summary: str
    known_status: str = "visible"
    detail_summary: str | None = None


class VisibleLink(BaseModel):
    link_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    summary: str


class VisiblePayout(BaseModel):
    payout_id: str
    amount_band: str
    settlement_tick_band: str
    source_entity_id: str
    destination_type: str
    amount: float | None = None
    settlement_tick: int | None = None
    destination_entity_id: str | None = None
    holdable: bool | None = None
    status: str | None = None
    detail_summary: str | None = None


class ActiveIntervention(BaseModel):
    intervention_type: InterventionType
    target_id: str
    status: str
    created_at_step: int
    created_at_time_tick: int


class ActionResult(BaseModel):
    status: str
    error: str | None = None
    new_entities: list[str] = Field(default_factory=list)
    new_links: list[str] = Field(default_factory=list)
    new_payouts: list[str] = Field(default_factory=list)
    state_changes: dict[str, Any] = Field(default_factory=dict)


class FraudRingInvestigatorArenaAction(Action):
    action_type: Literal[
        "inspect_entity",
        "expand_links",
        "trace_funds",
        "inspect_payout",
        "hold_payout",
        "freeze_entity",
        "advance_time",
        "submit_case",
    ] = Field(..., description="Typed environment action")
    entity_id: str | None = Field(default=None, description="Visible entity ID")
    payout_id: str | None = Field(default=None, description="Visible payout ID")
    relation_type: RelationType | None = Field(default=None)
    direction: DirectionType | None = Field(default=None)
    depth: int | None = Field(default=None, description="Fund-trace depth, 1 or 2")
    decision: DecisionType | None = Field(default=None)
    suspect_entity_ids: list[str] = Field(default_factory=list)


class FraudRingInvestigatorArenaObservation(Observation):
    task_id: str
    task_name: str
    case_id: str
    step_count: int
    steps_remaining: int
    time_tick: int
    time_ticks_remaining: int
    investigation_cost_used: int
    seed_alert: SeedAlert
    visible_entities: list[VisibleEntity] = Field(default_factory=list)
    visible_links: list[VisibleLink] = Field(default_factory=list)
    visible_payouts: list[VisiblePayout] = Field(default_factory=list)
    active_interventions: list[ActiveIntervention] = Field(default_factory=list)
    blocked_value_realized: float = 0.0
    escaped_value_realized: float = 0.0
    last_action_result: ActionResult | None = None


class FraudRingInvestigatorArenaState(State):
    task_id: str = ""
    task_name: str = ""
    case_id: str = ""
    time_tick: int = 0
    max_steps: int = 0
    max_time_ticks: int = 0
    cost_budget: int = 0
    investigation_cost_used: int = 0
    done: bool = False
