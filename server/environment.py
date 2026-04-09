from __future__ import annotations

try:
    from .fraud_ring_investigator_arena_environment import (
        grade_easy_single_ring_v1,
        grade_hard_reserve_ring_v1,
        grade_medium_confounded_ring_v1,
    )
except ImportError:
    from server.fraud_ring_investigator_arena_environment import (
        grade_easy_single_ring_v1,
        grade_hard_reserve_ring_v1,
        grade_medium_confounded_ring_v1,
    )


def grade_easy(*args, **kwargs):
    return grade_easy_single_ring_v1(*args, **kwargs)


def grade_medium(*args, **kwargs):
    return grade_medium_confounded_ring_v1(*args, **kwargs)


def grade_hard(*args, **kwargs):
    return grade_hard_reserve_ring_v1(*args, **kwargs)
