from datetime import datetime, timedelta, timezone

from terminalweave.cranes.rail_limits import CraneRailAuthority, RailSegment
from terminalweave.cranes.recovery import (
    ActiveLift,
    CraneRailProfile,
    CraneRecoveryPlanner,
    RecoveryTask,
)
from terminalweave.cranes.safety import CraneEnvelope, LiftPlan, LiftSafetyChecker
from terminalweave.cranes.scheduler import Crane
from terminalweave.cranes.workqueue import CraneWorkQueue, WorkMove
from terminalweave.core.intervals import TimeInterval

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)


def _move(move_id, bay=10, predecessor_ids=frozenset(), priority=0):
    return WorkMove(move_id, bay, 1, priority, predecessor_ids)


def _plan(move_id, weight_t=30.0):
    return LiftPlan(f'container-{move_id}', weight_t, 30.0, 5.0)


def _task(
    move,
    rail_position_m=50.0,
    earliest=D,
    latest=None,
    duration=timedelta(hours=1),
    weight_t=30.0,
):
    return RecoveryTask(
        move,
        earliest,
        latest or D + timedelta(hours=8),
        duration,
        _plan(move.move_id, weight_t),
        rail_position_m,
    )


def _fixtures():
    cranes = [
        Crane('C1', 0, 30),
        Crane('C2', 1, 30),
    ]
    envelope = CraneEnvelope(50.0, 100.0, 30.0, 40.0)
    checkers = {crane.crane_id: LiftSafetyChecker(envelope) for crane in cranes}
    authority = CraneRailAuthority([RailSegment(0.0, 100.0, 100.0)])
    profiles = {
        crane.crane_id: CraneRailProfile(crane.crane_id, 60.0, 20.0)
        for crane in cranes
    }
    return cranes, checkers, authority, profiles


def test_replan_keeps_started_lift_fixed_and_reschedules_unstarted_moves():
    started = _move('A')
    after_a = _move('B', predecessor_ids=frozenset({'A'}), priority=2)
    independent = _move('C', bay=40, priority=1)
    queue = CraneWorkQueue([started, after_a, independent])
    queue.claim('A')

    active = ActiveLift(
        'A',
        'C1',
        TimeInterval(D - timedelta(hours=1), D + timedelta(hours=2)),
        10.0,
        _plan('A'),
    )
    cranes, checkers, authority, profiles = _fixtures()
    planner = CraneRecoveryPlanner(cranes, checkers, authority, profiles)

    result = planner.replan(
        queue,
        [
            _task(after_a, rail_position_m=10.0),
            _task(independent, rail_position_m=80.0),
        ],
        D,
        [active],
    )

    assert result.fully_recovered
    assert result.active_lifts == (active,)
    assert {a.move_id for a in result.assignments} == {'B', 'C'}
    assert {a.move_id for a in result.assignments if a.interval.start == D} == {'C'}
    b = next(a for a in result.assignments if a.move_id == 'B')
    assert b.interval.start == D + timedelta(hours=2)


def test_replan_reports_rail_mutual_exclusion_reason():
    started = _move('A')
    blocked = _move('R', bay=40)
    queue = CraneWorkQueue([started, blocked])
    queue.claim('A')
    active = ActiveLift(
        'A',
        'C1',
        TimeInterval(D - timedelta(hours=1), D + timedelta(hours=2)),
        10.0,
        _plan('A'),
    )
    cranes, checkers, authority, profiles = _fixtures()
    planner = CraneRecoveryPlanner(cranes, checkers, authority, profiles)

    result = planner.replan(
        queue,
        [_task(blocked, rail_position_m=10.0, latest=D + timedelta(hours=1))],
        D,
        [active],
    )

    assert not result.assignments
    failure = result.failures[0]
    assert failure.move_id == 'R'
    assert 'rail_occupied' in {r.code for r in failure.reasons}
    assert {next(iter(r.blocking_move_ids)) for r in failure.reasons if r.code == 'rail_occupied'} == {'A'}


def test_replan_reports_safety_envelope_and_dependency_reasons():
    heavy = _move('H', bay=50)
    dependent = _move('D', bay=50, predecessor_ids=frozenset({'H'}))
    queue = CraneWorkQueue([heavy, dependent])
    cranes, checkers, authority, profiles = _fixtures()
    planner = CraneRecoveryPlanner(cranes, checkers, authority, profiles)

    result = planner.replan(
        queue,
        [
            _task(heavy, rail_position_m=80.0, weight_t=60.0),
            _task(dependent, rail_position_m=80.0),
        ],
        D,
        [],
    )

    reasons = {failure.move_id: {reason.code for reason in failure.reasons} for failure in result.failures}
    assert reasons['H'] == {'lift_envelope'}
    assert reasons['D'] == {'predecessor_unscheduled'}


def test_replan_reports_dependency_cycle():
    first = _move('P', predecessor_ids=frozenset({'Q'}))
    second = _move('Q', predecessor_ids=frozenset({'P'}))
    queue = CraneWorkQueue([first, second])
    cranes, checkers, authority, profiles = _fixtures()
    planner = CraneRecoveryPlanner(cranes, checkers, authority, profiles)

    result = planner.replan(queue, [_task(first), _task(second)], D, [])

    assert {failure.move_id for failure in result.failures} == {'P', 'Q'}
    assert all(failure.reasons[0].code == 'dependency_cycle' for failure in result.failures)


def test_replan_checks_state_of_started_lift_after_sudden_stop():
    started = _move('A')
    pending = _move('B', bay=40)
    queue = CraneWorkQueue([started, pending])
    queue.claim('A')
    active = ActiveLift(
        'A',
        'C1',
        TimeInterval(D - timedelta(hours=1), D + timedelta(hours=2)),
        10.0,
        _plan('A'),
    )
    cranes, checkers, _, profiles = _fixtures()
    authority = CraneRailAuthority([
        RailSegment(0.0, 15.0, 100.0, closed=True),
        RailSegment(15.0, 100.0, 100.0),
    ])
    planner = CraneRecoveryPlanner(cranes, checkers, authority, profiles)

    result = planner.replan(queue, [_task(pending, rail_position_m=80.0)], D, [active])

    assert not result.fully_recovered
    assert result.active_issues[0].move_id == 'A'
    assert result.active_issues[0].code == 'rail_closed'


def test_queue_exposes_recovery_replan():
    started = _move('A')
    pending = _move('B', bay=40)
    queue = CraneWorkQueue([started, pending])
    queue.claim('A')
    active = ActiveLift(
        'A',
        'C1',
        TimeInterval(D - timedelta(hours=1), D + timedelta(hours=2)),
        10.0,
        _plan('A'),
    )
    cranes, checkers, authority, profiles = _fixtures()

    result = queue.replan_after_stop(
        [_task(pending, rail_position_m=80.0)],
        D,
        [active],
        cranes,
        checkers,
        authority,
        profiles,
    )

    assert [a.move_id for a in result.assignments] == ['B']
