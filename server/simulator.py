from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal
from uuid import uuid4

try:
    from ..models import (
        ActionResult,
        ActiveIntervention,
        FraudRingInvestigatorArenaObservation,
        SeedAlert,
        VisibleEntity,
        VisibleLink,
        VisiblePayout,
    )
except ImportError:
    from models import (
        ActionResult,
        ActiveIntervention,
        FraudRingInvestigatorArenaObservation,
        SeedAlert,
        VisibleEntity,
        VisibleLink,
        VisiblePayout,
    )


TrackName = Literal[
    "easy_single_ring_v1",
    "medium_confounded_ring_v1",
    "hard_reserve_ring_v1",
]

TRACK_ID_TO_TASK_ALIAS: dict[TrackName, str] = {
    "easy_single_ring_v1": "task1",
    "medium_confounded_ring_v1": "task2",
    "hard_reserve_ring_v1": "task3",
}

ACTION_COSTS: dict[str, int] = {
    "inspect_entity": 1,
    "expand_links": 1,
    "trace_funds": 2,
    "inspect_payout": 1,
    "hold_payout": 3,
    "freeze_entity": 5,
    "advance_time": 0,
    "submit_case": 0,
}

FRAUD_MOTIFS = [
    "fan_in_cashout",
    "scatter_gather",
    "gather_scatter",
    "merchant_loop",
]

BENIGN_FAMILIES = [
    "seasonal_merchant_spike",
    "shared_device_household_cluster",
    "payroll_or_marketplace_aggregator",
    "refund_or_reimbursement_wave",
]

ALERT_FAMILIES = [
    "payout_anomaly",
    "shared_device_anomaly",
    "shared_identity_anomaly",
    "merchant_velocity_anomaly",
    "transfer_pattern_anomaly",
]


@dataclass(frozen=True)
class TrackConfig:
    task_name: TrackName
    max_steps: int
    max_time_ticks: int
    cost_budget: int
    fraud_probability: float
    ring_size_range: tuple[int, int]
    confounder_range: tuple[int, int]
    payout_waves_range: tuple[int, int]
    visible_seed_range: tuple[int, int]
    reserve_route_probability: float


TRACK_CONFIGS: dict[TrackName, TrackConfig] = {
    "easy_single_ring_v1": TrackConfig(
        task_name="easy_single_ring_v1",
        max_steps=8,
        max_time_ticks=2,
        cost_budget=10,
        fraud_probability=0.75,
        ring_size_range=(3, 5),
        confounder_range=(0, 1),
        payout_waves_range=(1, 1),
        visible_seed_range=(4, 6),
        reserve_route_probability=0.0,
    ),
    "medium_confounded_ring_v1": TrackConfig(
        task_name="medium_confounded_ring_v1",
        max_steps=10,
        max_time_ticks=3,
        cost_budget=14,
        fraud_probability=0.70,
        ring_size_range=(4, 7),
        confounder_range=(1, 3),
        payout_waves_range=(1, 2),
        visible_seed_range=(3, 5),
        reserve_route_probability=0.0,
    ),
    "hard_reserve_ring_v1": TrackConfig(
        task_name="hard_reserve_ring_v1",
        max_steps=12,
        max_time_ticks=4,
        cost_budget=18,
        fraud_probability=0.60,
        ring_size_range=(5, 9),
        confounder_range=(2, 5),
        payout_waves_range=(2, 3),
        visible_seed_range=(2, 4),
        reserve_route_probability=0.5,
    ),
}


@dataclass
class HiddenEntity:
    entity_id: str
    entity_type: str
    display_name: str
    risk_flags: list[str]
    activity_summary: str
    detail_summary: str
    hidden_role: str | None = None
    known_status: str = "visible"
    freeze_harm_value: float = 0.0


@dataclass
class HiddenLink:
    link_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    summary: str
    suspicious: bool = False


@dataclass
class HiddenPayout:
    payout_id: str
    source_entity_id: str
    destination_entity_id: str
    amount: float
    settlement_tick: int
    destination_type: str
    fraudulent: bool
    holdable: bool = True
    status: str = "pending"
    route_entity_ids: list[str] = field(default_factory=list)
    risk_clue: str = ""
    reserve_source_id: str | None = None


@dataclass
class InterventionRecord:
    intervention_type: str
    target_id: str
    correct: bool


@dataclass
class HiddenWorld:
    task_name: TrackName
    seed: int
    case_id: str
    max_steps: int
    max_time_ticks: int
    cost_budget: int
    case_truth: str
    motif_family: str
    alert_family: str
    seed_alert: SeedAlert
    entities: dict[str, HiddenEntity]
    links: dict[str, HiddenLink]
    payouts: dict[str, HiddenPayout]
    ring_member_ids: set[str]
    benign_confounder_ids: set[str]
    benign_harm_normalizer: float
    visible_entity_ids: set[str] = field(default_factory=set)
    visible_link_ids: set[str] = field(default_factory=set)
    visible_payout_ids: set[str] = field(default_factory=set)
    inspected_entities: set[str] = field(default_factory=set)
    inspected_payouts: set[str] = field(default_factory=set)
    expanded_relations: set[tuple[str, str]] = field(default_factory=set)
    traced_specs: set[tuple[str, str, int]] = field(default_factory=set)
    frozen_entity_ids: set[str] = field(default_factory=set)
    interventions: list[InterventionRecord] = field(default_factory=list)
    reserve_route_enabled: bool = False
    reserve_route_armed_payout_id: str | None = None
    reserve_route_triggered: bool = False
    benign_harm_value_realized: float = 0.0
    submitted_decision: str | None = None
    submitted_suspect_ids: list[str] = field(default_factory=list)
    step_count: int = 0
    time_tick: int = 0
    investigation_cost_used: int = 0
    done: bool = False
    final_score: float | None = None
    terminal_metrics: dict[str, float] | None = None

    def visible_actionable_entity_ids(self) -> list[str]:
        return sorted(
            entity_id
            for entity_id in self.visible_entity_ids
            if self.entities[entity_id].entity_type in {"account", "merchant"}
        )


def _choice_without_replacement(
    rng: random.Random, pool: list[str], count: int
) -> list[str]:
    count = max(0, min(count, len(pool)))
    return rng.sample(pool, count)


def _amount_band(amount: float) -> str:
    if amount < 250:
        return "low"
    if amount < 700:
        return "medium"
    return "high"


def _settlement_tick_band(current_tick: int, settlement_tick: int) -> str:
    delta = settlement_tick - current_tick
    if delta <= 0:
        return "now"
    if delta == 1:
        return "soon"
    return "later"


def _make_entity_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:03d}"


def _make_entity(
    entity_id: str,
    entity_type: str,
    display_name: str,
    activity_summary: str,
    detail_summary: str,
    risk_flags: Iterable[str] = (),
    hidden_role: str | None = None,
    freeze_harm_value: float = 0.0,
) -> HiddenEntity:
    return HiddenEntity(
        entity_id=entity_id,
        entity_type=entity_type,
        display_name=display_name,
        risk_flags=list(dict.fromkeys(risk_flags)),
        activity_summary=activity_summary,
        detail_summary=detail_summary,
        hidden_role=hidden_role,
        freeze_harm_value=freeze_harm_value,
    )


def _add_link(
    links: dict[str, HiddenLink],
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    summary: str,
    suspicious: bool = False,
) -> str:
    link_id = f"{relation_type}_{len(links) + 1:03d}"
    links[link_id] = HiddenLink(
        link_id=link_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        summary=summary,
        suspicious=suspicious,
    )
    return link_id


def _account_activity(rng: random.Random) -> tuple[str, str, list[str]]:
    options = [
        (
            "steady consumer spending with weekly transfers",
            "Recent counterparties are concentrated around one settlement corridor.",
            ["velocity_spike"],
        ),
        (
            "marketplace cash-in and payout activity",
            "Recent outflows are more fragmented than the account's baseline pattern.",
            ["fragmented_flow"],
        ),
        (
            "small-business settlement behavior with end-of-day batching",
            "Settlement timing overlaps with a higher-risk linked merchant cluster.",
            ["settlement_overlap"],
        ),
        (
            "peer-to-peer transfers with occasional merchant spend",
            "A shared artifact link connects this account to multiple case entities.",
            ["shared_artifact"],
        ),
    ]
    return rng.choice(options)


def _merchant_activity(rng: random.Random) -> tuple[str, str, list[str]]:
    options = [
        (
            "card-present merchant traffic with moderate payout batching",
            "Merchant settlement cadence is unusually synchronized with the alerted cluster.",
            ["merchant_velocity"],
        ),
        (
            "digital marketplace settlement and refund handling",
            "Refund-like movement is clustered around a narrow group of linked accounts.",
            ["refund_spike"],
        ),
        (
            "aggregated platform payouts across many counterparties",
            "A subset of settlements routes through the same linked artifact chain.",
            ["linked_settlement"],
        ),
    ]
    return rng.choice(options)


def _create_background_entities(
    rng: random.Random,
) -> tuple[dict[str, HiddenEntity], list[str], list[str], list[str], list[str]]:
    account_count = rng.randint(12, 24)
    merchant_count = rng.randint(3, 7)
    device_count = rng.randint(5, 10)
    identity_count = rng.randint(5, 10)

    entities: dict[str, HiddenEntity] = {}
    account_ids: list[str] = []
    merchant_ids: list[str] = []
    device_ids: list[str] = []
    identity_ids: list[str] = []

    for index in range(1, account_count + 1):
        entity_id = _make_entity_id("acct", index)
        activity_summary, detail_summary, risk_flags = _account_activity(rng)
        entities[entity_id] = _make_entity(
            entity_id,
            "account",
            f"Account {index}",
            activity_summary,
            detail_summary,
            risk_flags,
            freeze_harm_value=float(rng.randint(120, 280)),
        )
        account_ids.append(entity_id)

    for index in range(1, merchant_count + 1):
        entity_id = _make_entity_id("mcht", index)
        activity_summary, detail_summary, risk_flags = _merchant_activity(rng)
        entities[entity_id] = _make_entity(
            entity_id,
            "merchant",
            f"Merchant {index}",
            activity_summary,
            detail_summary,
            risk_flags,
            freeze_harm_value=float(rng.randint(180, 420)),
        )
        merchant_ids.append(entity_id)

    for index in range(1, device_count + 1):
        entity_id = _make_entity_id("dev", index)
        entities[entity_id] = _make_entity(
            entity_id,
            "device",
            f"Device {index}",
            "linked login or instrument artifact",
            "This device is shared across multiple visible and hidden entities.",
            ["artifact_link"],
        )
        device_ids.append(entity_id)

    for index in range(1, identity_count + 1):
        entity_id = _make_entity_id("idn", index)
        entities[entity_id] = _make_entity(
            entity_id,
            "identity",
            f"Identity {index}",
            "KYC identity with linked account ownership",
            "This identity ties together multiple accounts or merchants in the local world.",
            ["identity_overlap"],
        )
        identity_ids.append(entity_id)

    return entities, account_ids, merchant_ids, device_ids, identity_ids


def _create_background_links(
    rng: random.Random,
    entities: dict[str, HiddenEntity],
    account_ids: list[str],
    merchant_ids: list[str],
    device_ids: list[str],
    identity_ids: list[str],
) -> dict[str, HiddenLink]:
    links: dict[str, HiddenLink] = {}

    for account_id in account_ids:
        device_id = rng.choice(device_ids)
        identity_id = rng.choice(identity_ids)
        merchant_id = rng.choice(merchant_ids)
        _add_link(
            links,
            account_id,
            device_id,
            "shared_device",
            f"{account_id} shares a device artifact with {device_id}",
        )
        _add_link(
            links,
            account_id,
            identity_id,
            "shared_identity",
            f"{account_id} is associated with identity {identity_id}",
        )
        _add_link(
            links,
            account_id,
            merchant_id,
            "merchant_interaction",
            f"{account_id} has recent merchant interaction with {merchant_id}",
        )

    transfer_edges = max(8, len(account_ids) // 2)
    for _ in range(transfer_edges):
        src, dst = rng.sample(account_ids, 2)
        _add_link(
            links,
            src,
            dst,
            "transfers",
            f"{src} transferred funds to {dst}",
        )

    return links


def _mark_entity_suspicious(
    entity: HiddenEntity, detail_summary: str, extra_flags: Iterable[str], hidden_role: str
) -> None:
    entity.hidden_role = hidden_role
    entity.detail_summary = detail_summary
    entity.risk_flags = list(dict.fromkeys([*entity.risk_flags, *extra_flags]))


def _build_fraud_case(
    rng: random.Random,
    config: TrackConfig,
    entities: dict[str, HiddenEntity],
    account_ids: list[str],
    merchant_ids: list[str],
    links: dict[str, HiddenLink],
) -> tuple[str, set[str], dict[str, HiddenPayout], list[str], str, bool, str | None]:
    motif_family = rng.choice(FRAUD_MOTIFS)
    payouts: dict[str, HiddenPayout] = {}
    payout_focus_ids: list[str] = []
    ring_member_ids: set[str] = set()
    reserve_source_id: str | None = None
    reserve_route_enabled = rng.random() < config.reserve_route_probability

    ring_size = rng.randint(*config.ring_size_range)
    selected_accounts = _choice_without_replacement(rng, account_ids, max(ring_size, 4))

    if motif_family == "fan_in_cashout":
        hub_id = selected_accounts[0]
        source_ids = selected_accounts[1:-1]
        cashout_id = selected_accounts[-1]
        ring_member_ids.update([hub_id, *source_ids, cashout_id])
        _mark_entity_suspicious(
            entities[hub_id],
            "Incoming flows converge from multiple recently linked accounts before a near-term payout.",
            ["fan_in"],
            "hub_account",
        )
        _mark_entity_suspicious(
            entities[cashout_id],
            "Cash-out routing is closely coupled to the convergence node's outgoing activity.",
            ["cashout_link"],
            "mule_out",
        )
        for source_id in source_ids:
            _mark_entity_suspicious(
                entities[source_id],
                "This account contributes short-horizon inflows into the same convergence path.",
                ["linked_inflow"],
                "mule_in",
            )
            _add_link(
                links,
                source_id,
                hub_id,
                "transfers",
                f"{source_id} transfers into shared hub {hub_id}",
                suspicious=True,
            )
        payout_count = rng.randint(*config.payout_waves_range)
        for index in range(1, payout_count + 1):
            payout_id = f"payout_{index:03d}"
            amount = float(rng.randint(350, 950))
            settlement_tick = min(config.max_time_ticks, index)
            payouts[payout_id] = HiddenPayout(
                payout_id=payout_id,
                source_entity_id=hub_id,
                destination_entity_id=cashout_id,
                amount=amount,
                settlement_tick=settlement_tick,
                destination_type=entities[cashout_id].entity_type,
                fraudulent=True,
                status="pending",
                route_entity_ids=[hub_id, cashout_id],
                risk_clue="Payout destination sits on a high-velocity cash-out corridor.",
            )
            payout_focus_ids.append(payout_id)
        reserve_source_id = rng.choice(account_ids)

    elif motif_family == "scatter_gather":
        orchestrator_id = selected_accounts[0]
        beneficiary_id = selected_accounts[-1]
        intermediary_ids = selected_accounts[1:-1]
        ring_member_ids.update([orchestrator_id, beneficiary_id, *intermediary_ids])
        _mark_entity_suspicious(
            entities[orchestrator_id],
            "Outgoing transfers branch rapidly into multiple intermediaries on a short schedule.",
            ["scatter_origin"],
            "orchestrator",
        )
        _mark_entity_suspicious(
            entities[beneficiary_id],
            "Reconverged funds leave through a concentrated downstream payout path.",
            ["gather_sink"],
            "cashout_beneficiary",
        )
        for intermediary_id in intermediary_ids:
            _mark_entity_suspicious(
                entities[intermediary_id],
                "This intermediary sits between a branching origin and a shared beneficiary.",
                ["bridge_flow"],
                "intermediate_mule",
            )
            _add_link(
                links,
                orchestrator_id,
                intermediary_id,
                "transfers",
                f"{orchestrator_id} distributes funds to intermediary {intermediary_id}",
                suspicious=True,
            )
            _add_link(
                links,
                intermediary_id,
                beneficiary_id,
                "transfers",
                f"{intermediary_id} reconverges value into {beneficiary_id}",
                suspicious=True,
            )
        payout_count = rng.randint(*config.payout_waves_range)
        for index in range(1, payout_count + 1):
            payout_id = f"payout_{index:03d}"
            amount = float(rng.randint(420, 900))
            settlement_tick = min(config.max_time_ticks, index + 1)
            payouts[payout_id] = HiddenPayout(
                payout_id=payout_id,
                source_entity_id=beneficiary_id,
                destination_entity_id=orchestrator_id,
                amount=amount,
                settlement_tick=settlement_tick,
                destination_type=entities[orchestrator_id].entity_type,
                fraudulent=True,
                status="pending",
                route_entity_ids=[orchestrator_id, *intermediary_ids, beneficiary_id],
                risk_clue="The payout path reconverges value that was first scattered across intermediaries.",
            )
            payout_focus_ids.append(payout_id)
        reserve_source_id = rng.choice(account_ids)

    elif motif_family == "gather_scatter":
        inbound_ids = selected_accounts[:-2]
        hub_id = selected_accounts[-2]
        outbound_id = selected_accounts[-1]
        ring_member_ids.update([hub_id, outbound_id, *inbound_ids])
        _mark_entity_suspicious(
            entities[hub_id],
            "Several incoming sources gather here before value redistributes outward.",
            ["gather_hub"],
            "hub_account",
        )
        _mark_entity_suspicious(
            entities[outbound_id],
            "This downstream node participates in short-horizon redistribution after a gather step.",
            ["scatter_exit"],
            "mule_out",
        )
        for inbound_id in inbound_ids:
            _mark_entity_suspicious(
                entities[inbound_id],
                "Recent activity shows repeated inbound contribution into the same hub account.",
                ["linked_gather"],
                "mule_in",
            )
            _add_link(
                links,
                inbound_id,
                hub_id,
                "transfers",
                f"{inbound_id} contributes funds into hub {hub_id}",
                suspicious=True,
            )
        _add_link(
            links,
            hub_id,
            outbound_id,
            "transfers",
            f"{hub_id} redistributes funds to outbound account {outbound_id}",
            suspicious=True,
        )
        payout_count = rng.randint(*config.payout_waves_range)
        for index in range(1, payout_count + 1):
            payout_id = f"payout_{index:03d}"
            amount = float(rng.randint(280, 760))
            settlement_tick = min(config.max_time_ticks, index)
            payouts[payout_id] = HiddenPayout(
                payout_id=payout_id,
                source_entity_id=outbound_id,
                destination_entity_id=hub_id,
                amount=amount,
                settlement_tick=settlement_tick,
                destination_type=entities[hub_id].entity_type,
                fraudulent=True,
                status="pending",
                route_entity_ids=[*inbound_ids, hub_id, outbound_id],
                risk_clue="The payout follows a gather-then-scatter pattern through the same hub account.",
            )
            payout_focus_ids.append(payout_id)
        reserve_source_id = rng.choice(account_ids)

    else:
        orchestrator_id = selected_accounts[0]
        settlement_id = selected_accounts[1]
        cashout_id = selected_accounts[2]
        shell_merchant_id = rng.choice(merchant_ids)
        ring_member_ids.update([orchestrator_id, settlement_id, cashout_id, shell_merchant_id])
        _mark_entity_suspicious(
            entities[orchestrator_id],
            "This account repeatedly routes spend through one merchant before settlement.",
            ["merchant_loop"],
            "orchestrator",
        )
        _mark_entity_suspicious(
            entities[settlement_id],
            "Settlement activity from a linked merchant cluster feeds into near-term payout traffic.",
            ["settlement_bridge"],
            "settlement_account",
        )
        _mark_entity_suspicious(
            entities[cashout_id],
            "Downstream cash-out behavior closely follows merchant settlement timing.",
            ["cashout_link"],
            "cashout_account",
        )
        _mark_entity_suspicious(
            entities[shell_merchant_id],
            "Merchant interaction density is unusually concentrated around the alerted accounts.",
            ["merchant_velocity", "loop_overlap"],
            "shell_merchant",
        )
        _add_link(
            links,
            orchestrator_id,
            shell_merchant_id,
            "merchant_interaction",
            f"{orchestrator_id} routes transactions through merchant {shell_merchant_id}",
            suspicious=True,
        )
        _add_link(
            links,
            shell_merchant_id,
            settlement_id,
            "transfers",
            f"Merchant settlement from {shell_merchant_id} routes into {settlement_id}",
            suspicious=True,
        )
        payout_count = rng.randint(*config.payout_waves_range)
        for index in range(1, payout_count + 1):
            payout_id = f"payout_{index:03d}"
            amount = float(rng.randint(500, 980))
            settlement_tick = min(config.max_time_ticks, index)
            payouts[payout_id] = HiddenPayout(
                payout_id=payout_id,
                source_entity_id=settlement_id,
                destination_entity_id=cashout_id,
                amount=amount,
                settlement_tick=settlement_tick,
                destination_type=entities[cashout_id].entity_type,
                fraudulent=True,
                status="pending",
                route_entity_ids=[orchestrator_id, shell_merchant_id, settlement_id, cashout_id],
                risk_clue="Merchant-linked settlement funds are scheduled to cash out soon.",
            )
            payout_focus_ids.append(payout_id)
        reserve_source_id = rng.choice(account_ids)

    return (
        motif_family,
        ring_member_ids,
        payouts,
        payout_focus_ids,
        reserve_source_id is not None and reserve_route_enabled,
        reserve_source_id,
    )


def _build_benign_case(
    rng: random.Random,
    config: TrackConfig,
    entities: dict[str, HiddenEntity],
    account_ids: list[str],
    merchant_ids: list[str],
    links: dict[str, HiddenLink],
) -> tuple[str, set[str], dict[str, HiddenPayout], list[str]]:
    family = rng.choice(BENIGN_FAMILIES)
    benign_focus_ids: set[str] = set()
    payouts: dict[str, HiddenPayout] = {}
    payout_focus_ids: list[str] = []
    group_size = max(3, rng.randint(*config.ring_size_range))
    selected_accounts = _choice_without_replacement(rng, account_ids, group_size)

    if family == "seasonal_merchant_spike":
        merchant_id = rng.choice(merchant_ids)
        benign_focus_ids.update(selected_accounts[:3])
        benign_focus_ids.add(merchant_id)
        for account_id in selected_accounts[:3]:
            _mark_entity_suspicious(
                entities[account_id],
                "This account participates in a seasonal payout spike tied to one merchant.",
                ["seasonal_spike", "shared_settlement"],
                "benign_high_risk",
            )
            _add_link(
                links,
                account_id,
                merchant_id,
                "merchant_interaction",
                f"{account_id} has a legitimate seasonal interaction with {merchant_id}",
                suspicious=True,
            )
        payout_id = "payout_001"
        amount = float(rng.randint(360, 820))
        payouts[payout_id] = HiddenPayout(
            payout_id=payout_id,
            source_entity_id=merchant_id,
            destination_entity_id=selected_accounts[0],
            amount=amount,
            settlement_tick=min(config.max_time_ticks, 1),
            destination_type=entities[selected_accounts[0]].entity_type,
            fraudulent=False,
            status="pending",
            route_entity_ids=[merchant_id, selected_accounts[0]],
            risk_clue="The payout is large but consistent with seasonal settlement behavior.",
        )
        payout_focus_ids.append(payout_id)

    elif family == "shared_device_household_cluster":
        focus_accounts = selected_accounts[:4]
        benign_focus_ids.update(focus_accounts)
        for account_id in focus_accounts:
            _mark_entity_suspicious(
                entities[account_id],
                "The account shares artifacts within a household-like cluster.",
                ["shared_artifact", "household_overlap"],
                "benign_high_risk",
            )
        payout_id = "payout_001"
        payouts[payout_id] = HiddenPayout(
            payout_id=payout_id,
            source_entity_id=focus_accounts[0],
            destination_entity_id=focus_accounts[1],
            amount=float(rng.randint(180, 380)),
            settlement_tick=min(config.max_time_ticks, 2),
            destination_type=entities[focus_accounts[1]].entity_type,
            fraudulent=False,
            status="pending",
            route_entity_ids=[focus_accounts[0], focus_accounts[1]],
            risk_clue="The payout sits inside a shared-device cluster but is legitimate.",
        )
        payout_focus_ids.append(payout_id)

    elif family == "payroll_or_marketplace_aggregator":
        hub_id = selected_accounts[0]
        benign_focus_ids.update(selected_accounts[:4])
        _mark_entity_suspicious(
            entities[hub_id],
            "The account acts like an aggregator with many legitimate inbound and outbound flows.",
            ["aggregator_pattern", "velocity_spike"],
            "benign_high_risk",
        )
        for account_id in selected_accounts[1:4]:
            _add_link(
                links,
                hub_id,
                account_id,
                "transfers",
                f"{hub_id} distributes legitimate aggregator payouts to {account_id}",
                suspicious=True,
            )
        payout_id = "payout_001"
        payouts[payout_id] = HiddenPayout(
            payout_id=payout_id,
            source_entity_id=hub_id,
            destination_entity_id=selected_accounts[1],
            amount=float(rng.randint(420, 720)),
            settlement_tick=min(config.max_time_ticks, 1),
            destination_type=entities[selected_accounts[1]].entity_type,
            fraudulent=False,
            status="pending",
            route_entity_ids=[hub_id, selected_accounts[1]],
            risk_clue="The payout is high-velocity but fits an aggregator pattern.",
        )
        payout_focus_ids.append(payout_id)

    else:
        merchant_id = rng.choice(merchant_ids)
        benign_focus_ids.update(selected_accounts[:3])
        benign_focus_ids.add(merchant_id)
        for account_id in selected_accounts[:3]:
            _mark_entity_suspicious(
                entities[account_id],
                "This account is tied to a concentrated refund or reimbursement burst.",
                ["refund_spike", "merchant_overlap"],
                "benign_high_risk",
            )
        payout_id = "payout_001"
        payouts[payout_id] = HiddenPayout(
            payout_id=payout_id,
            source_entity_id=merchant_id,
            destination_entity_id=selected_accounts[0],
            amount=float(rng.randint(250, 540)),
            settlement_tick=min(config.max_time_ticks, 2),
            destination_type=entities[selected_accounts[0]].entity_type,
            fraudulent=False,
            status="pending",
            route_entity_ids=[merchant_id, selected_accounts[0]],
            risk_clue="The payout resembles a refund wave rather than a fraud cash-out.",
        )
        payout_focus_ids.append(payout_id)

    return family, set(), payouts, payout_focus_ids


def _add_confounders(
    rng: random.Random,
    config: TrackConfig,
    entities: dict[str, HiddenEntity],
    account_ids: list[str],
    merchant_ids: list[str],
    links: dict[str, HiddenLink],
    excluded_ids: set[str],
) -> set[str]:
    confounder_ids: set[str] = set()
    confounder_count = rng.randint(*config.confounder_range)
    available_accounts = [account_id for account_id in account_ids if account_id not in excluded_ids]
    for account_id in _choice_without_replacement(rng, available_accounts, confounder_count):
        confounder_ids.add(account_id)
        merchant_id = rng.choice(merchant_ids)
        _mark_entity_suspicious(
            entities[account_id],
            "This entity shares one or two suspicious-looking signals but lacks the hidden fraud route.",
            ["shared_artifact", "surface_overlap"],
            "benign_high_risk",
        )
        _add_link(
            links,
            account_id,
            merchant_id,
            "merchant_interaction",
            f"{account_id} overlaps with merchant {merchant_id} on recent activity",
            suspicious=True,
        )
    return confounder_ids


def _choose_alert_family(rng: random.Random, payouts: dict[str, HiddenPayout], motif_family: str) -> str:
    if payouts and rng.random() < 0.45:
        return "payout_anomaly"
    if "merchant" in motif_family and rng.random() < 0.5:
        return "merchant_velocity_anomaly"
    return rng.choice([family for family in ALERT_FAMILIES if family != "payout_anomaly"])


def _build_seed_alert(
    rng: random.Random,
    alert_family: str,
    entities: dict[str, HiddenEntity],
    payouts: dict[str, HiddenPayout],
    case_truth: str,
    ring_member_ids: set[str],
    benign_confounder_ids: set[str],
) -> SeedAlert:
    severity_band = "high" if case_truth == "fraud" else "medium"
    all_focus_ids = list(ring_member_ids or benign_confounder_ids)
    if not all_focus_ids:
        all_focus_ids = [next(iter(entities))]
    seed_entity_ids = [rng.choice(all_focus_ids)]
    seed_payout_id: str | None = None

    if alert_family == "payout_anomaly" and payouts:
        seed_payout_id = sorted(payouts)[0]
        seed_entity_ids = [payouts[seed_payout_id].source_entity_id]
        narrative = (
            f"An unusual payout linked to {seed_entity_ids[0]} is scheduled unusually soon for this local network."
        )
    elif alert_family == "shared_device_anomaly":
        narrative = f"{seed_entity_ids[0]} is linked through a shared device pattern spanning multiple case entities."
    elif alert_family == "shared_identity_anomaly":
        narrative = f"{seed_entity_ids[0]} participates in a dense shared-identity overlap cluster."
    elif alert_family == "merchant_velocity_anomaly":
        narrative = f"{seed_entity_ids[0]} appears in merchant-linked activity with unusual settlement velocity."
    else:
        narrative = f"{seed_entity_ids[0]} appears in a transfer pattern that looks inconsistent with its baseline behavior."

    return SeedAlert(
        alert_id=f"alert_{uuid4().hex[:8]}",
        alert_type=alert_family,
        severity_band=severity_band,
        narrative=narrative,
        seed_entity_ids=seed_entity_ids,
        seed_payout_id=seed_payout_id,
    )


def _adjacent_links(world: HiddenWorld, entity_id: str, relation_type: str | None = None) -> list[HiddenLink]:
    links = [
        link
        for link in world.links.values()
        if entity_id in {link.source_entity_id, link.target_entity_id}
    ]
    if relation_type is not None:
        links = [link for link in links if link.relation_type == relation_type]
    return sorted(links, key=lambda link: (not link.suspicious, link.link_id))


def _add_visible_entity(world: HiddenWorld, entity_id: str, new_entities: list[str]) -> None:
    if entity_id not in world.visible_entity_ids:
        world.visible_entity_ids.add(entity_id)
        new_entities.append(entity_id)


def _add_visible_link(world: HiddenWorld, link_id: str, new_links: list[str]) -> None:
    if link_id not in world.visible_link_ids:
        world.visible_link_ids.add(link_id)
        new_links.append(link_id)


def _add_visible_payout(world: HiddenWorld, payout_id: str, new_payouts: list[str]) -> None:
    if payout_id not in world.visible_payout_ids:
        world.visible_payout_ids.add(payout_id)
        new_payouts.append(payout_id)


def _initialize_visible_slice(world: HiddenWorld, rng: random.Random, config: TrackConfig) -> None:
    visible_target = rng.randint(*config.visible_seed_range)
    seed_entity_ids = list(world.seed_alert.seed_entity_ids)
    for entity_id in seed_entity_ids:
        world.visible_entity_ids.add(entity_id)
    if world.seed_alert.seed_payout_id is not None:
        world.visible_payout_ids.add(world.seed_alert.seed_payout_id)
        payout = world.payouts[world.seed_alert.seed_payout_id]
        world.visible_entity_ids.add(payout.source_entity_id)

    frontier = list(seed_entity_ids)
    while frontier and len(world.visible_entity_ids) < visible_target:
        current = frontier.pop(0)
        for link in _adjacent_links(world, current):
            neighbor_id = (
                link.target_entity_id if link.source_entity_id == current else link.source_entity_id
            )
            world.visible_link_ids.add(link.link_id)
            if neighbor_id not in world.visible_entity_ids:
                world.visible_entity_ids.add(neighbor_id)
                frontier.append(neighbor_id)
            if len(world.visible_entity_ids) >= visible_target:
                break

    for payout_id, payout in sorted(world.payouts.items()):
        if payout.source_entity_id in world.visible_entity_ids and rng.random() < 0.5:
            world.visible_payout_ids.add(payout_id)


def build_world(task_name: TrackName, seed: int) -> HiddenWorld:
    rng = random.Random(seed)
    config = TRACK_CONFIGS[task_name]
    entities, account_ids, merchant_ids, device_ids, identity_ids = _create_background_entities(rng)
    links = _create_background_links(rng, entities, account_ids, merchant_ids, device_ids, identity_ids)

    case_truth = "fraud" if rng.random() < config.fraud_probability else "benign"
    if case_truth == "fraud":
        (
            motif_family,
            ring_member_ids,
            payouts,
            _payout_focus_ids,
            reserve_route_enabled,
            reserve_source_id,
        ) = _build_fraud_case(rng, config, entities, account_ids, merchant_ids, links)
        confounder_ids = _add_confounders(
            rng,
            config,
            entities,
            account_ids,
            merchant_ids,
            links,
            excluded_ids=ring_member_ids,
        )
        if reserve_route_enabled and reserve_source_id is not None and payouts:
            first_payout_id = sorted(payouts)[0]
            payouts[first_payout_id].reserve_source_id = reserve_source_id
            _mark_entity_suspicious(
                entities[reserve_source_id],
                "This reserve node is not obviously central until the ring adapts after intervention.",
                ["reserve_overlap"],
                "reserve_mule",
            )
    else:
        motif_family, ring_member_ids, payouts, _payout_focus_ids = _build_benign_case(
            rng, config, entities, account_ids, merchant_ids, links
        )
        reserve_route_enabled = False
        confounder_ids = _add_confounders(
            rng,
            config,
            entities,
            account_ids,
            merchant_ids,
            links,
            excluded_ids=set(),
        )

    alert_family = _choose_alert_family(rng, payouts, motif_family)
    seed_alert = _build_seed_alert(
        rng,
        alert_family,
        entities,
        payouts,
        case_truth,
        ring_member_ids,
        confounder_ids,
    )

    world = HiddenWorld(
        task_name=task_name,
        seed=seed,
        case_id=f"{task_name}-{seed}-{uuid4().hex[:8]}",
        max_steps=config.max_steps,
        max_time_ticks=config.max_time_ticks,
        cost_budget=config.cost_budget,
        case_truth=case_truth,
        motif_family=motif_family,
        alert_family=alert_family,
        seed_alert=seed_alert,
        entities=entities,
        links=links,
        payouts=payouts,
        ring_member_ids=set(sorted(ring_member_ids)),
        benign_confounder_ids=confounder_ids,
        benign_harm_normalizer=max(
            300.0,
            sum(payout.amount for payout in payouts.values()) if payouts else 300.0,
        ),
        reserve_route_enabled=reserve_route_enabled,
    )

    _initialize_visible_slice(world, rng, config)
    return world


def visible_entities(world: HiddenWorld) -> list[VisibleEntity]:
    entities: list[VisibleEntity] = []
    for entity_id in sorted(world.visible_entity_ids):
        hidden = world.entities[entity_id]
        detail_summary = hidden.detail_summary if entity_id in world.inspected_entities else None
        known_status = "profiled" if entity_id in world.inspected_entities else hidden.known_status
        entities.append(
            VisibleEntity(
                entity_id=hidden.entity_id,
                entity_type=hidden.entity_type,
                display_name=hidden.display_name,
                risk_flags=hidden.risk_flags,
                activity_summary=hidden.activity_summary,
                known_status=known_status,
                detail_summary=detail_summary,
            )
        )
    return entities


def visible_links(world: HiddenWorld) -> list[VisibleLink]:
    links: list[VisibleLink] = []
    for link_id in sorted(world.visible_link_ids):
        hidden = world.links[link_id]
        if hidden.source_entity_id in world.visible_entity_ids and hidden.target_entity_id in world.visible_entity_ids:
            links.append(
                VisibleLink(
                    link_id=hidden.link_id,
                    source_entity_id=hidden.source_entity_id,
                    target_entity_id=hidden.target_entity_id,
                    relation_type=hidden.relation_type,
                    summary=hidden.summary,
                )
            )
    return links


def visible_payouts(world: HiddenWorld) -> list[VisiblePayout]:
    payouts: list[VisiblePayout] = []
    for payout_id in sorted(world.visible_payout_ids):
        hidden = world.payouts[payout_id]
        inspected = payout_id in world.inspected_payouts
        payouts.append(
            VisiblePayout(
                payout_id=hidden.payout_id,
                amount_band=_amount_band(hidden.amount),
                settlement_tick_band=_settlement_tick_band(world.time_tick, hidden.settlement_tick),
                source_entity_id=hidden.source_entity_id,
                destination_type=hidden.destination_type,
                amount=hidden.amount if inspected else None,
                settlement_tick=hidden.settlement_tick if inspected else None,
                destination_entity_id=hidden.destination_entity_id if inspected else None,
                holdable=hidden.holdable if inspected else None,
                status=hidden.status if inspected else None,
                detail_summary=hidden.risk_clue if inspected else None,
            )
        )
    return payouts


def active_interventions(world: HiddenWorld) -> list[ActiveIntervention]:
    interventions: list[ActiveIntervention] = []
    for entity_id in sorted(world.frozen_entity_ids):
        interventions.append(
            ActiveIntervention(
                intervention_type="freeze_entity",
                target_id=entity_id,
                status="active",
                created_at_step=world.step_count,
                created_at_time_tick=world.time_tick,
            )
        )
    for payout in sorted(world.payouts.values(), key=lambda payout: payout.payout_id):
        if payout.status in {"blocked_pending", "blocked_realized", "held_benign"}:
            interventions.append(
                ActiveIntervention(
                    intervention_type="hold_payout",
                    target_id=payout.payout_id,
                    status=payout.status,
                    created_at_step=world.step_count,
                    created_at_time_tick=world.time_tick,
                )
            )
    return interventions


def build_observation(
    world: HiddenWorld,
    last_action_result: ActionResult | None,
    reward: float,
    done: bool,
    metadata: dict[str, Any] | None = None,
) -> FraudRingInvestigatorArenaObservation:
    return FraudRingInvestigatorArenaObservation(
        task_id=TRACK_ID_TO_TASK_ALIAS[world.task_name],
        task_name=world.task_name,
        case_id=world.case_id,
        step_count=world.step_count,
        steps_remaining=max(world.max_steps - world.step_count, 0),
        time_tick=world.time_tick,
        time_ticks_remaining=max(world.max_time_ticks - world.time_tick, 0),
        investigation_cost_used=world.investigation_cost_used,
        seed_alert=world.seed_alert,
        visible_entities=visible_entities(world),
        visible_links=visible_links(world),
        visible_payouts=visible_payouts(world),
        active_interventions=active_interventions(world),
        blocked_value_realized=sum(
            payout.amount
            for payout in world.payouts.values()
            if payout.fraudulent and payout.status == "blocked_realized"
        ),
        escaped_value_realized=sum(
            payout.amount
            for payout in world.payouts.values()
            if payout.fraudulent and payout.status == "escaped"
        ),
        last_action_result=last_action_result,
        reward=reward,
        done=done,
        metadata=metadata or {},
    )


def reveal_entity_slice(
    world: HiddenWorld,
    entity_id: str,
    relation_type: str | None,
    limit: int,
) -> tuple[list[str], list[str]]:
    new_entities: list[str] = []
    new_links: list[str] = []
    for link in _adjacent_links(world, entity_id, relation_type):
        neighbor_id = link.target_entity_id if link.source_entity_id == entity_id else link.source_entity_id
        _add_visible_link(world, link.link_id, new_links)
        _add_visible_entity(world, neighbor_id, new_entities)
        if len(new_entities) >= limit:
            break
    return new_entities, new_links


def reveal_payouts_for_entities(
    world: HiddenWorld,
    entity_ids: Iterable[str],
    limit: int,
) -> list[str]:
    new_payouts: list[str] = []
    entity_id_set = set(entity_ids)
    for payout in sorted(world.payouts.values(), key=lambda payout: payout.payout_id):
        if payout.source_entity_id in entity_id_set or payout.destination_entity_id in entity_id_set:
            _add_visible_payout(world, payout.payout_id, new_payouts)
            if len(new_payouts) >= limit:
                break
    return new_payouts
