from __future__ import annotations

import os
import random
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        ActionResult,
        FraudRingInvestigatorArenaAction,
        FraudRingInvestigatorArenaObservation,
        FraudRingInvestigatorArenaState,
    )
    from .scoring import compute_terminal_metrics
    from .simulator import (
        ACTION_COSTS,
        InterventionRecord,
        TRACK_CONFIGS,
        HiddenWorld,
        build_observation,
        build_world,
        reveal_entity_slice,
        reveal_payouts_for_entities,
    )
except ImportError:
    from models import (
        ActionResult,
        FraudRingInvestigatorArenaAction,
        FraudRingInvestigatorArenaObservation,
        FraudRingInvestigatorArenaState,
    )
    from server.scoring import compute_terminal_metrics
    from server.simulator import (
        ACTION_COSTS,
        InterventionRecord,
        TRACK_CONFIGS,
        HiddenWorld,
        build_observation,
        build_world,
        reveal_entity_slice,
        reveal_payouts_for_entities,
    )


BASE_ACTION_PENALTIES: dict[str, float] = {
    "inspect_entity": -0.01,
    "expand_links": -0.01,
    "inspect_payout": -0.01,
    "trace_funds": -0.02,
    "hold_payout": -0.03,
    "freeze_entity": -0.05,
    "advance_time": 0.0,
    "submit_case": 0.0,
}


class FraudRingInvestigatorArenaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._world: HiddenWorld | None = None
        self._state = FraudRingInvestigatorArenaState()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> FraudRingInvestigatorArenaObservation:
        del episode_id
        task_name = (
            kwargs.get("task_id")
            or kwargs.get("task_name")
            or os.getenv("FRAUD_RING_ARENA_TASK", "medium_confounded_ring_v1")
        )
        if task_name not in TRACK_CONFIGS:
            raise ValueError(f"Unknown task_id/task_name: {task_name}")
        if seed is None:
            seed = random.randint(0, 89_999)
        self._world = build_world(task_name, int(seed))
        self._sync_state()
        return build_observation(
            self._world,
            last_action_result=None,
            reward=0.0,
            done=False,
            metadata={"seed": float(seed)},
        )

    def step(
        self,
        action: FraudRingInvestigatorArenaAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> FraudRingInvestigatorArenaObservation:
        del timeout_s, kwargs
        if self._world is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        world = self._world
        if world.done:
            return build_observation(
                world,
                last_action_result=ActionResult(
                    status="invalid",
                    error="Episode already completed.",
                ),
                reward=0.0,
                done=True,
                metadata={"episode_score": world.final_score or 0.0},
            )

        world.step_count += 1
        world.investigation_cost_used += ACTION_COSTS[action.action_type]

        error = self._validate_action(world, action)
        if error is not None:
            reward, done = self._handle_invalid_action(world, action, error)
            self._sync_state()
            return build_observation(
                world,
                last_action_result=ActionResult(status="invalid", error=error),
                reward=reward,
                done=done,
                metadata=self._metadata(world),
            )

        if action.action_type == "submit_case":
            reward = self._handle_submit(world, action)
            self._sync_state()
            return build_observation(
                world,
                last_action_result=ActionResult(
                    status="submitted",
                    state_changes={
                        "decision": world.submitted_decision,
                        "suspect_entity_ids": world.submitted_suspect_ids,
                    },
                ),
                reward=reward,
                done=True,
                metadata=self._metadata(world),
            )

        last_action_result, shaping_reward = self._apply_non_terminal_action(world, action)
        done = False
        reward = shaping_reward

        if world.step_count >= world.max_steps:
            reward = self._finalize_timeout(world)
            done = True
            last_action_result.state_changes["timeout"] = True

        self._sync_state()
        return build_observation(
            world,
            last_action_result=last_action_result,
            reward=reward,
            done=done,
            metadata=self._metadata(world),
        )

    @property
    def state(self) -> FraudRingInvestigatorArenaState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="fraud_ring_investigator_arena",
            description=(
                "A sequential fraud investigation environment with partial observability, "
                "costly interventions, delayed payout consequences, and deterministic scoring."
            ),
            version="0.1.0",
        )

    def _sync_state(self) -> None:
        world = self._world
        if world is None:
            self._state = FraudRingInvestigatorArenaState()
            return
        self._state = FraudRingInvestigatorArenaState(
            episode_id=world.case_id,
            task_id=world.task_name,
            step_count=world.step_count,
            task_name=world.task_name,
            case_id=world.case_id,
            time_tick=world.time_tick,
            max_steps=world.max_steps,
            max_time_ticks=world.max_time_ticks,
            cost_budget=world.cost_budget,
            investigation_cost_used=world.investigation_cost_used,
            done=world.done,
        )

    def _metadata(self, world: HiddenWorld) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "seed": float(world.seed),
            "task_id": world.task_name,
            "task_name": world.task_name,
            "task_count": float(len(TRACK_CONFIGS)),
        }
        if world.done and world.terminal_metrics is not None:
            metadata["episode_score"] = float(world.final_score or 0.0)
            metadata["terminal_metrics"] = world.terminal_metrics
        return metadata

    def _validate_action(
        self, world: HiddenWorld, action: FraudRingInvestigatorArenaAction
    ) -> str | None:
        if len(set(action.suspect_entity_ids)) != len(action.suspect_entity_ids):
            return "Duplicate suspect IDs are not allowed."

        if action.action_type == "inspect_entity":
            if not action.entity_id:
                return "inspect_entity requires entity_id."
            if action.entity_id not in world.visible_entity_ids:
                return "entity_id must already be visible."
            return None

        if action.action_type == "expand_links":
            if not action.entity_id or not action.relation_type:
                return "expand_links requires entity_id and relation_type."
            if action.entity_id not in world.visible_entity_ids:
                return "entity_id must already be visible."
            return None

        if action.action_type == "trace_funds":
            if not action.entity_id or action.direction is None or action.depth is None:
                return "trace_funds requires entity_id, direction, and depth."
            if action.entity_id not in world.visible_entity_ids:
                return "entity_id must already be visible."
            if action.depth not in {1, 2}:
                return "trace_funds depth must be 1 or 2."
            return None

        if action.action_type == "inspect_payout":
            if not action.payout_id:
                return "inspect_payout requires payout_id."
            payout = world.payouts.get(action.payout_id)
            if payout is None or action.payout_id not in world.visible_payout_ids:
                return "payout_id must already be visible."
            if payout.status in {"blocked_realized", "escaped", "settled_benign"}:
                return "payout is already settled."
            return None

        if action.action_type == "hold_payout":
            if not action.payout_id:
                return "hold_payout requires payout_id."
            payout = world.payouts.get(action.payout_id)
            if payout is None or action.payout_id not in world.visible_payout_ids:
                return "payout_id must already be visible."
            if payout.status not in {"pending"}:
                return "payout must be pending and not already held."
            if not payout.holdable:
                return "payout is not holdable."
            return None

        if action.action_type == "freeze_entity":
            if not action.entity_id:
                return "freeze_entity requires entity_id."
            if action.entity_id not in world.visible_entity_ids:
                return "entity_id must already be visible."
            entity = world.entities[action.entity_id]
            if entity.entity_type not in {"account", "merchant"}:
                return "Only account or merchant entities can be frozen."
            if action.entity_id in world.frozen_entity_ids:
                return "entity is already frozen."
            return None

        if action.action_type == "advance_time":
            if world.time_tick >= world.max_time_ticks:
                return "No time ticks remain."
            return None

        if action.action_type == "submit_case":
            if action.decision not in {"clear", "escalate"}:
                return "submit_case requires decision clear or escalate."
            if action.decision == "clear" and action.suspect_entity_ids:
                return "clear requires an empty suspect list."
            if action.decision == "escalate" and not action.suspect_entity_ids:
                return "escalate requires at least one suspect."
            if len(action.suspect_entity_ids) > 6:
                return "At most 6 suspects may be submitted."
            visible_actionable = set(world.visible_actionable_entity_ids())
            for suspect_id in action.suspect_entity_ids:
                if suspect_id not in visible_actionable:
                    return "Suspects must be visible actionable entities."
            return None

        return "Unknown action type."

    def _handle_invalid_action(
        self,
        world: HiddenWorld,
        action: FraudRingInvestigatorArenaAction,
        error: str,
    ) -> tuple[float, bool]:
        if world.step_count >= world.max_steps:
            return self._finalize_timeout(world), True
        penalty = BASE_ACTION_PENALTIES.get(action.action_type, 0.0) - 0.05
        return penalty, False

    def _apply_non_terminal_action(
        self,
        world: HiddenWorld,
        action: FraudRingInvestigatorArenaAction,
    ) -> tuple[ActionResult, float]:
        reward = BASE_ACTION_PENALTIES[action.action_type]
        new_entities: list[str] = []
        new_links: list[str] = []
        new_payouts: list[str] = []
        state_changes: dict[str, Any] = {}

        if action.action_type == "inspect_entity":
            world.inspected_entities.add(action.entity_id or "")
            revealed_entities, revealed_links = reveal_entity_slice(
                world, action.entity_id or "", relation_type=None, limit=1
            )
            new_entities.extend(revealed_entities)
            new_links.extend(revealed_links)
            if action.entity_id:
                state_changes["entity_profiled"] = action.entity_id

        elif action.action_type == "expand_links":
            relation_key = (action.entity_id or "", action.relation_type or "")
            already_expanded = relation_key in world.expanded_relations
            world.expanded_relations.add(relation_key)
            revealed_entities, revealed_links = reveal_entity_slice(
                world,
                action.entity_id or "",
                relation_type=action.relation_type,
                limit=3,
            )
            new_entities.extend(revealed_entities)
            new_links.extend(revealed_links)
            state_changes["expanded_relation"] = action.relation_type
            if already_expanded and not revealed_entities and not revealed_links:
                reward -= 0.02

        elif action.action_type == "trace_funds":
            trace_key = (action.entity_id or "", action.direction or "", int(action.depth or 0))
            already_traced = trace_key in world.traced_specs
            world.traced_specs.add(trace_key)
            revealed_entities, revealed_links = reveal_entity_slice(
                world,
                action.entity_id or "",
                relation_type="transfers",
                limit=2 if action.depth == 1 else 4,
            )
            new_entities.extend(revealed_entities)
            new_links.extend(revealed_links)
            new_payouts.extend(
                reveal_payouts_for_entities(
                    world,
                    entity_ids=[action.entity_id or "", *revealed_entities],
                    limit=2,
                )
            )
            state_changes["path_summary"] = self._build_path_summary(
                world, action.entity_id or "", action.direction or "both", int(action.depth or 1)
            )
            if already_traced and not new_entities and not new_links and not new_payouts:
                reward -= 0.02

        elif action.action_type == "inspect_payout":
            if action.payout_id:
                world.inspected_payouts.add(action.payout_id)
                payout = world.payouts[action.payout_id]
                state_changes["payout_inspected"] = action.payout_id
                if payout.destination_entity_id not in world.visible_entity_ids:
                    world.visible_entity_ids.add(payout.destination_entity_id)
                    new_entities.append(payout.destination_entity_id)

        elif action.action_type == "hold_payout":
            payout = world.payouts[action.payout_id or ""]
            if payout.fraudulent:
                payout.status = "blocked_pending"
                correct = True
            else:
                payout.status = "held_benign"
                world.benign_harm_value_realized += payout.amount
                reward -= 0.05
                correct = False
            world.interventions.append(
                InterventionRecord("hold_payout", payout.payout_id, correct)
            )
            state_changes["payout_status"] = payout.status
            if payout.destination_entity_id not in world.visible_entity_ids:
                world.visible_entity_ids.add(payout.destination_entity_id)
                new_entities.append(payout.destination_entity_id)

        elif action.action_type == "freeze_entity":
            entity_id = action.entity_id or ""
            world.frozen_entity_ids.add(entity_id)
            correct = entity_id in world.ring_member_ids
            if not correct:
                world.benign_harm_value_realized += world.entities[entity_id].freeze_harm_value
                reward -= 0.05
            world.interventions.append(
                InterventionRecord("freeze_entity", entity_id, correct)
            )
            affected_payouts: list[str] = []
            for payout in world.payouts.values():
                if (
                    payout.fraudulent
                    and payout.status == "pending"
                    and entity_id in payout.route_entity_ids
                ):
                    if (
                        world.reserve_route_enabled
                        and not world.reserve_route_triggered
                        and world.reserve_route_armed_payout_id is None
                        and payout.reserve_source_id is not None
                    ):
                        world.reserve_route_armed_payout_id = payout.payout_id
                        affected_payouts.append(f"{payout.payout_id}:reserve_route_armed")
                    else:
                        payout.status = "blocked_pending"
                        affected_payouts.append(f"{payout.payout_id}:blocked_pending")
            state_changes["frozen_entity"] = entity_id
            if affected_payouts:
                state_changes["affected_payouts"] = affected_payouts

        elif action.action_type == "advance_time":
            reward += self._advance_time(world, new_entities, new_links, new_payouts, state_changes)

        evidence_found = bool(new_entities or new_links or new_payouts or state_changes)
        if evidence_found and action.action_type not in {"hold_payout", "freeze_entity", "advance_time"}:
            reward += 0.01
        elif not evidence_found and action.action_type in {
            "inspect_entity",
            "expand_links",
            "trace_funds",
            "inspect_payout",
        }:
            reward -= 0.02

        return (
            ActionResult(
                status="ok",
                new_entities=new_entities,
                new_links=new_links,
                new_payouts=new_payouts,
                state_changes=state_changes,
            ),
            reward,
        )

    def _advance_time(
        self,
        world: HiddenWorld,
        new_entities: list[str],
        new_links: list[str],
        new_payouts: list[str],
        state_changes: dict[str, Any],
    ) -> float:
        reward_delta = 0.0
        world.time_tick += 1
        state_changes["time_tick"] = world.time_tick

        if (
            world.reserve_route_armed_payout_id is not None
            and not world.reserve_route_triggered
            and world.reserve_route_armed_payout_id in world.payouts
        ):
            payout = world.payouts[world.reserve_route_armed_payout_id]
            if payout.status == "pending" and payout.reserve_source_id is not None:
                original_source = payout.source_entity_id
                payout.source_entity_id = payout.reserve_source_id
                payout.route_entity_ids = [payout.reserve_source_id, *payout.route_entity_ids]
                payout.settlement_tick = min(world.max_time_ticks, payout.settlement_tick + 1)
                world.reserve_route_triggered = True
                state_changes["reserve_reroute"] = payout.payout_id
                if payout.reserve_source_id not in world.visible_entity_ids:
                    world.visible_entity_ids.add(payout.reserve_source_id)
                    new_entities.append(payout.reserve_source_id)
                reward_delta += 0.0
                for link_id, link in world.links.items():
                    if {
                        link.source_entity_id,
                        link.target_entity_id,
                    } == {original_source, payout.reserve_source_id}:
                        if link_id not in world.visible_link_ids:
                            world.visible_link_ids.add(link_id)
                            new_links.append(link_id)
                        break

        matured: list[str] = []
        for payout in sorted(world.payouts.values(), key=lambda payout: payout.payout_id):
            if payout.settlement_tick != world.time_tick:
                continue
            matured.append(payout.payout_id)
            if payout.fraudulent:
                if payout.status == "blocked_pending":
                    payout.status = "blocked_realized"
                    reward_delta += 0.05
                elif payout.status == "pending":
                    payout.status = "escaped"
                    reward_delta -= 0.05
            else:
                if payout.status == "pending":
                    payout.status = "settled_benign"
            _ = reveal_payouts_for_entities(
                world, [payout.source_entity_id, payout.destination_entity_id], limit=2
            )
            if payout.destination_entity_id not in world.visible_entity_ids:
                world.visible_entity_ids.add(payout.destination_entity_id)
                new_entities.append(payout.destination_entity_id)
            if payout.payout_id not in world.visible_payout_ids:
                world.visible_payout_ids.add(payout.payout_id)
                new_payouts.append(payout.payout_id)
        if matured:
            state_changes["matured_payouts"] = matured
        return reward_delta

    def _build_path_summary(
        self,
        world: HiddenWorld,
        entity_id: str,
        direction: str,
        depth: int,
    ) -> str:
        adjacent_transfers = [
            link
            for link in world.links.values()
            if link.relation_type == "transfers" and entity_id in {link.source_entity_id, link.target_entity_id}
        ]
        if direction == "incoming":
            adjacent_transfers = [
                link for link in adjacent_transfers if link.target_entity_id == entity_id
            ]
        elif direction == "outgoing":
            adjacent_transfers = [
                link for link in adjacent_transfers if link.source_entity_id == entity_id
            ]
        counterparties = {
            link.target_entity_id if link.source_entity_id == entity_id else link.source_entity_id
            for link in adjacent_transfers
        }
        return (
            f"Trace depth {depth} from {entity_id} exposes {len(counterparties)} transfer counterparties "
            f"under {direction} flow."
        )

    def _handle_submit(
        self,
        world: HiddenWorld,
        action: FraudRingInvestigatorArenaAction,
    ) -> float:
        world.submitted_decision = action.decision
        world.submitted_suspect_ids = list(action.suspect_entity_ids)
        world.done = True
        metrics = compute_terminal_metrics(world)
        world.final_score = metrics.episode_score
        world.terminal_metrics = metrics.as_metadata()
        return metrics.episode_score

    def _finalize_timeout(self, world: HiddenWorld) -> float:
        world.done = True
        world.submitted_decision = None
        world.submitted_suspect_ids = []
        metrics = compute_terminal_metrics(world)
        world.final_score = metrics.episode_score
        world.terminal_metrics = metrics.as_metadata()
        return metrics.episode_score
