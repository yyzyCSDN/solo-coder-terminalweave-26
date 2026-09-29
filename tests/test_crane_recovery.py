from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.core.errors import SafetyConstraintError
from terminalweave.core.intervals import TimeInterval
from terminalweave.cranes.rail_limits import CraneRailAuthority, RailSegment
from terminalweave.cranes.recovery import (
    ActiveLift,
    CraneRecoveryPlanner,
    NO_CONFLICT_FREE_SLOT,
    PREDECESSOR_UNSCHEDULED,
    RAIL_OCCUPIED,
    RecoveryCrane,
    RecoveryTask,
    UNSAFE_ENVELOPE,
)
from terminalweave.cranes.safety import CraneEnvelope
from terminalweave.cranes.workqueue import CraneWorkQueue, WorkMove

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)


def _envelope():
    return CraneEnvelope(
        max_weight_t=40,
        max_outreach_m=20,
        max_wind_kn=30,
        twin_lift_max_weight_t=50,
    )


def _planner():
    return CraneRecoveryPlanner(
        [
            RecoveryCrane('C1', 0, 30, _envelope(), 12, 25),
            RecoveryCrane('C2', 1, 60, _envelope(), 12, 25),
        ],
        CraneRailAuthority(
            [RailSegment(0, 100, 40), RailSegment(100, 120, 40, closed=True)]
        ),
    )


def _task(move, center, **kwargs):
    values = {
        'earliest_start': D,
        'latest_finish': D + timedelta(hours=8),
        'gross_weight_t': 20,
        'outreach_m': 10,
        'wind_kn': 10,
        'rail_center_m': center,
    }
    values.update(kwargs)
    return RecoveryTask(move, **values)


def test_recovery_preserves_active_and_replans_pending():
    moves = [
        WorkMove('A', 10, 1, 1),
        WorkMove('B', 30, 1, 2),
        WorkMove('C', 31, 1, 2, frozenset(['B'])),
        WorkMove('P', 20, 1, 1, frozenset(['A'])),
        WorkMove('G', 50, 1, 1),
        WorkMove('DANGER', 40, 1, 1),
        WorkMove('AFTER_DANGER', 40, 1, 1, frozenset(['DANGER'])),
        WorkMove('CLOSED_RAIL', 40, 1, 1),
    ]
    queue = CraneWorkQueue(moves)
    queue.claim('A')

    active = [
        ActiveLift(
            'A',
            'C1',
            TimeInterval(D + timedelta(minutes=30), D + timedelta(hours=1, minutes=30)),
            10,
            10,
            20,
            10,
            10,
        )
    ]
    tasks = [
        _task(moves[1], 50),
        _task(moves[2], 50),
        _task(moves[3], 10),
        _task(
            moves[4],
            10,
            earliest_start=D + timedelta(hours=1),
            latest_finish=D + timedelta(hours=1, minutes=15),
        ),
        _task(moves[5], 50, gross_weight_t=100),
        _task(moves[6], 50),
        _task(moves[7], 110),
    ]
    restart_at = D + timedelta(hours=1)

    planner = _planner()
    plan = planner.replan(queue, tasks, active, restart_at)
    planner.apply_replan(queue, plan)
    by_id = {item.move_id: item for item in plan.all_lifts}
    reasons = {item.move_id: item.reasons for item in plan.unscheduled}

    assert queue.status('A') == 'in_progress'
    assert queue.status('B') == 'pending'
    assert [item.move_id for item in plan.active] == ['A']
    assert by_id['A'].crane_id == 'C1'
    assert by_id['A'].interval.start == restart_at
    assert by_id['A'].interval.end == D + timedelta(hours=1, minutes=30)
    assert by_id['B'].crane_id == 'C2'
    assert by_id['B'].interval.start >= restart_at
    assert by_id['C'].interval.start >= by_id['B'].interval.end
    assert by_id['P'].interval.start >= by_id['A'].interval.end

    for left in plan.all_lifts:
        for right in plan.all_lifts:
            if left.move_id >= right.move_id:
                continue
            if left.interval.overlaps(right.interval):
                assert left.crane_id != right.crane_id
                assert abs(left.rail_center_m - right.rail_center_m) >= 12
                assert abs(left.bay - right.bay) >= 2

    assert reasons['DANGER'][0].startswith(UNSAFE_ENVELOPE)
    assert reasons['AFTER_DANGER'] == (PREDECESSOR_UNSCHEDULED,)
    assert reasons['CLOSED_RAIL'][0].startswith(RAIL_OCCUPIED)
    assert reasons['G'][0].startswith(NO_CONFLICT_FREE_SLOT)
    assert {item.move_id for item in plan.unscheduled} == {
        'G',
        'DANGER',
        'AFTER_DANGER',
        'CLOSED_RAIL',
    }


def test_recovery_refuses_to_replan_started_move():
    moves = [WorkMove('A', 10, 1, 1), WorkMove('B', 20, 1, 1)]
    queue = CraneWorkQueue(moves)
    queue.claim('A')
    active = [
        ActiveLift(
            'A',
            'C1',
            TimeInterval(D, D + timedelta(hours=2)),
            10,
            10,
            20,
            10,
            10,
        )
    ]
    tasks = [_task(moves[0], 20)]

    with pytest.raises(Exception, match='only unstarted moves'):
        _planner().replan(queue, tasks, active, D)


def test_recovery_can_move_pending_lift_to_another_crane():
    moves = [WorkMove('B', 30, 1, 1)]
    queue = CraneWorkQueue(moves)
    task = _task(moves[0], 50, eligible_crane_ids=frozenset(['C2']))

    plan = _planner().replan(queue, [task], [], D)

    assert len(plan.scheduled) == 1
    assert plan.scheduled[0].crane_id == 'C2'
    assert plan.unscheduled == ()


def test_recovery_rejects_unsafe_active_lift():
    moves = [WorkMove('A', 10, 1, 1)]
    queue = CraneWorkQueue(moves)
    queue.claim('A')
    active = [
        ActiveLift(
            'A',
            'C1',
            TimeInterval(D, D + timedelta(hours=2)),
            10,
            10,
            100,
            10,
            10,
        )
    ]

    with pytest.raises(SafetyConstraintError):
        _planner().replan(queue, [], active, D)


def test_work_queue_reports_pending_and_uses_replan_order():
    moves = [WorkMove('A', 10, 1, 1), WorkMove('B', 20, 1, 1)]
    queue = CraneWorkQueue(moves)
    queue.claim('A')
    queue.apply_replan(('B',))

    assert queue.active_ids == {'A'}
    assert queue.pending_ids == {'B'}
    assert queue.next(10).move_id == 'B'
