from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ConflictError, SafetyConstraintError, ValidationError
from terminalweave.core.intervals import TimeInterval
from terminalweave.cranes.rail_limits import CranePosition, CraneRailAuthority
from terminalweave.cranes.safety import CraneEnvelope, LiftPlan, LiftSafetyChecker
from terminalweave.cranes.workqueue import CraneWorkQueue, WorkMove

IN_PROGRESS = 'in_progress'
SCHEDULED = 'scheduled'
UNSCHEDULED = 'unscheduled'

PREDECESSOR_UNSCHEDULED = 'predecessor_unscheduled'
NO_CONFLICT_FREE_SLOT = 'no_conflict_free_slot'
UNSAFE_ENVELOPE = 'unsafe_envelope'
RAIL_OCCUPIED = 'rail_occupied'
NOT_READY_BY_RESTART = 'not_ready_by_restart'
WINDOW_TOO_SHORT = 'window_too_short'
UNKNOWN_CRANE = 'unknown_crane'
NO_ELIGIBLE_CRANE = 'no_eligible_crane'


@dataclass(frozen=True)
class RecoveryCrane:
    crane_id: str
    rail_index: int
    moves_per_hour: float
    envelope: CraneEnvelope
    footprint_m: float
    wheel_load_t: float
    min_separation_bays: int = 2
    available_at: datetime | None = None

    def __post_init__(self) -> None:
        if (
            self.rail_index < 0
            or self.moves_per_hour <= 0
            or self.min_separation_bays < 0
            or self.footprint_m <= 0
            or self.wheel_load_t <= 0
        ):
            raise ValidationError('invalid recovery crane')
        if self.available_at is not None:
            object.__setattr__(self, 'available_at', ensure_utc(self.available_at))


@dataclass(frozen=True)
class RecoveryTask:
    move: WorkMove
    earliest_start: datetime
    latest_finish: datetime
    gross_weight_t: float
    outreach_m: float
    wind_kn: float
    rail_center_m: float
    eligible_crane_ids: frozenset[str] = field(default_factory=frozenset)
    twin_lift: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'earliest_start', ensure_utc(self.earliest_start))
        object.__setattr__(self, 'latest_finish', ensure_utc(self.latest_finish))
        if self.latest_finish <= self.earliest_start:
            raise ValidationError('invalid recovery task window')
        if self.gross_weight_t <= 0 or self.outreach_m < 0 or self.wind_kn < 0:
            raise ValidationError('invalid recovery lift')

    @property
    def move_id(self) -> str:
        return self.move.move_id

    @property
    def bay(self) -> int:
        return self.move.bay

    @property
    def tier(self) -> int:
        return self.move.tier

    @property
    def priority(self) -> int:
        return self.move.priority

    @property
    def predecessor_ids(self) -> frozenset[str]:
        return self.move.predecessor_ids

    def lift_plan(self) -> LiftPlan:
        return LiftPlan(
            self.move_id,
            self.gross_weight_t,
            self.outreach_m,
            self.wind_kn,
            self.twin_lift,
        )


@dataclass(frozen=True)
class ActiveLift:
    move_id: str
    crane_id: str
    interval: TimeInterval
    rail_center_m: float
    bay: int
    gross_weight_t: float
    outreach_m: float
    wind_kn: float
    twin_lift: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'interval',
            TimeInterval(self.interval.start, self.interval.end),
        )
        if self.gross_weight_t <= 0 or self.outreach_m < 0 or self.wind_kn < 0:
            raise ValidationError('invalid active lift')

    def lift_plan(self) -> LiftPlan:
        return LiftPlan(
            self.move_id,
            self.gross_weight_t,
            self.outreach_m,
            self.wind_kn,
            self.twin_lift,
        )


@dataclass(frozen=True)
class PlannedRecoveryLift:
    move_id: str
    crane_id: str
    rail_index: int
    bay: int
    rail_center_m: float
    interval: TimeInterval
    state: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'interval',
            TimeInterval(self.interval.start, self.interval.end),
        )


@dataclass(frozen=True)
class UnscheduledRecoveryTask:
    move_id: str
    state: str
    reasons: tuple[str, ...]
    blocking_move_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryPlan:
    as_of: datetime
    active: tuple[PlannedRecoveryLift, ...]
    scheduled: tuple[PlannedRecoveryLift, ...]
    unscheduled: tuple[UnscheduledRecoveryTask, ...]

    @property
    def all_lifts(self) -> tuple[PlannedRecoveryLift, ...]:
        return self.active + self.scheduled


class CraneRecoveryPlanner:

    def __init__(
        self,
        cranes: list[RecoveryCrane],
        rail_authority: CraneRailAuthority | None = None,
    ) -> None:
        if not cranes:
            raise ValidationError('at least one crane is required')
        ids = [c.crane_id for c in cranes]
        if len(ids) != len(set(ids)):
            raise ValidationError('duplicate crane id')
        self.cranes = {c.crane_id: c for c in cranes}
        self.rail_authority = rail_authority

    def _eligible_cranes(self, task: RecoveryTask) -> list[RecoveryCrane]:
        if not task.eligible_crane_ids:
            return sorted(self.cranes.values(), key=lambda c: (c.rail_index, c.crane_id))
        return [
            self.cranes[crane_id]
            for crane_id in sorted(task.eligible_crane_ids)
            if crane_id in self.cranes
        ]

    def _duration(self, crane: RecoveryCrane) -> timedelta:
        # A WorkMove represents one container lift in the recovery queue.
        return timedelta(hours=1 / crane.moves_per_hour)

    def _safety_error(self, crane: RecoveryCrane, task: RecoveryTask) -> str:
        try:
            LiftSafetyChecker(crane.envelope).validate(task.lift_plan())
        except SafetyConstraintError as exc:
            return f'{UNSAFE_ENVELOPE}:{crane.crane_id}:{exc}'
        return ''

    def _rail_error(self, crane: RecoveryCrane, center_m: float) -> str:
        if self.rail_authority is None:
            return ''
        try:
            self.rail_authority.validate(
                CranePosition(crane.crane_id, center_m, crane.wheel_load_t, crane.footprint_m)
            )
        except ConflictError as exc:
            return f'{RAIL_OCCUPIED}:{crane.crane_id}:{exc}'
        return ''

    def _overlaps(
        self,
        crane: RecoveryCrane,
        bay: int,
        center_m: float,
        interval: TimeInterval,
        fixed: list[PlannedRecoveryLift],
    ) -> tuple[bool, tuple[str, ...]]:
        blockers: list[str] = []
        for item in fixed:
            existing = self.cranes[item.crane_id]
            same_crane = crane.crane_id == item.crane_id
            physical_rail_overlap = abs(center_m - item.rail_center_m) < (
                crane.footprint_m + existing.footprint_m
            ) / 2
            too_close = abs(bay - item.bay) < max(
                crane.min_separation_bays,
                existing.min_separation_bays,
            )
            crossing = (
                crane.rail_index < existing.rail_index and bay > item.bay
            ) or (
                crane.rail_index > existing.rail_index and bay < item.bay
            )
            if interval.overlaps(item.interval) and (
                same_crane or physical_rail_overlap or too_close or crossing
            ):
                blockers.append(item.move_id)
        return bool(blockers), tuple(sorted(blockers))

    def _validate_active(
        self,
        active_lifts: list[ActiveLift],
        as_of: datetime,
    ) -> list[PlannedRecoveryLift]:
        fixed: list[PlannedRecoveryLift] = []
        for active in active_lifts:
            crane = self.cranes.get(active.crane_id)
            if crane is None:
                raise ValidationError(f'unknown active crane: {active.crane_id}')
            if active.interval.start > as_of:
                raise ValidationError(f'active lift has not started: {active.move_id}')
            if active.interval.end <= as_of:
                raise ValidationError(f'active lift has finished: {active.move_id}')
            try:
                LiftSafetyChecker(crane.envelope).validate(active.lift_plan())
            except SafetyConstraintError as exc:
                raise SafetyConstraintError(
                    f'active lift {active.move_id} violates crane envelope: {exc}'
                ) from exc
            rail_error = self._rail_error(crane, active.rail_center_m)
            if rail_error:
                raise ConflictError(
                    f'active lift {active.move_id} has invalid rail position: {rail_error}'
                )
            fixed.append(
                PlannedRecoveryLift(
                    active.move_id,
                    active.crane_id,
                    crane.rail_index,
                    active.bay,
                    active.rail_center_m,
                    TimeInterval(max(active.interval.start, as_of), active.interval.end),
                    IN_PROGRESS,
                )
            )
        for index, candidate in enumerate(fixed):
            candidate_crane = self.cranes[candidate.crane_id]
            for existing in fixed[index + 1 :]:
                overlaps, _ = self._overlaps(
                    candidate_crane,
                    candidate.bay,
                    candidate.rail_center_m,
                    candidate.interval,
                    [existing],
                )
                if overlaps:
                    raise ConflictError(
                        f'active lifts {candidate.move_id} and {existing.move_id} conflict'
                    )
        return fixed

    @staticmethod
    def _predecessor_finish(
        planned: list[PlannedRecoveryLift],
        predecessor_ids: frozenset[str],
        as_of: datetime,
    ) -> datetime:
        finishes = [item.interval.end for item in planned if item.move_id in predecessor_ids]
        return max([as_of, *finishes]) if finishes else as_of

    @staticmethod
    def _ensure_acyclic(queue: CraneWorkQueue) -> None:
        remaining = set(queue.moves)
        completed: set[str] = set()
        while remaining:
            ready = {
                move_id
                for move_id in remaining
                if queue.moves[move_id].predecessor_ids - completed <= set()
            }
            if not ready:
                raise ConflictError('work queue contains cycle')
            completed.update(ready)
            remaining.difference_update(ready)

    def _candidate_starts(
        self,
        earliest: datetime,
        latest_start: datetime,
        fixed: list[PlannedRecoveryLift],
    ) -> list[datetime]:
        starts = [earliest]
        for item in fixed:
            starts.extend(
                boundary
                for boundary in (item.interval.start, item.interval.end)
                if earliest <= boundary <= latest_start
            )
        return sorted(set(starts))

    def _place_task(
        self,
        task: RecoveryTask,
        predecessor_finish: datetime,
        as_of: datetime,
        fixed: list[PlannedRecoveryLift],
    ) -> tuple[PlannedRecoveryLift | None, tuple[str, ...], tuple[str, ...]]:
        if task.eligible_crane_ids and any(
            crane_id not in self.cranes for crane_id in task.eligible_crane_ids
        ):
            unknown = sorted(task.eligible_crane_ids - self.cranes.keys())
            return None, tuple(f'{UNKNOWN_CRANE}:{x}' for x in unknown), ()

        eligible = self._eligible_cranes(task)
        if not eligible:
            return None, (NO_ELIGIBLE_CRANE,), ()

        candidates: list[tuple[datetime, int, str, PlannedRecoveryLift]] = []
        reasons: list[str] = []
        blockers: set[str] = set()

        for crane in eligible:
            duration = self._duration(crane)
            earliest = max(
                task.earliest_start,
                predecessor_finish,
                as_of,
                crane.available_at or as_of,
            )

            safety_error = self._safety_error(crane, task)
            if safety_error:
                reasons.append(safety_error)
                continue
            rail_error = self._rail_error(crane, task.rail_center_m)
            if rail_error:
                reasons.append(rail_error)
                continue
            if earliest + duration > task.latest_finish:
                if earliest > task.earliest_start:
                    reasons.append(f'{NOT_READY_BY_RESTART}:{crane.crane_id}')
                reasons.append(f'{WINDOW_TOO_SHORT}:{crane.crane_id}')
                continue

            latest_start = task.latest_finish - duration
            crane_blockers: set[str] = set()
            for cursor in self._candidate_starts(earliest, latest_start, fixed):
                interval = TimeInterval(cursor, cursor + duration)
                overlaps, blocking_ids = self._overlaps(
                    crane, task.bay, task.rail_center_m, interval, fixed
                )
                if not overlaps:
                    candidates.append(
                        (
                            interval.end,
                            crane.rail_index,
                            crane.crane_id,
                            PlannedRecoveryLift(
                                task.move_id,
                                crane.crane_id,
                                crane.rail_index,
                                task.bay,
                                task.rail_center_m,
                                interval,
                                SCHEDULED,
                            ),
                        )
                    )
                    break
                crane_blockers.update(blocking_ids)

            if not crane_blockers:
                continue
            blockers.update(crane_blockers)
            reasons.append(f'{NO_CONFLICT_FREE_SLOT}:{crane.crane_id}')

        if candidates:
            return min(candidates, key=lambda x: (x[0], x[1], x[2]))[3], (), ()

        unique_reasons = tuple(dict.fromkeys(reasons))
        return None, unique_reasons, tuple(sorted(blockers))

    def replan(
        self,
        queue: CraneWorkQueue,
        tasks: list[RecoveryTask],
        active_lifts: list[ActiveLift],
        as_of: datetime,
    ) -> RecoveryPlan:
        as_of = ensure_utc(as_of)
        self._ensure_acyclic(queue)

        task_map = {task.move_id: task for task in tasks}
        if len(task_map) != len(tasks):
            raise ValidationError('duplicate recovery task')
        unknown = task_map.keys() - queue.moves.keys()
        if unknown:
            raise ValidationError(f'unknown tasks: {sorted(unknown)}')

        active_ids = {lift.move_id for lift in active_lifts}
        if len(active_ids) != len(active_lifts):
            raise ValidationError('duplicate active lift')
        if active_ids - queue.claimed:
            raise ValidationError('active lifts must be claimed in the work queue')
        unaccounted_claims = queue.claimed - active_ids
        if unaccounted_claims:
            raise ValidationError(
                f'claimed tasks without active lift state: {sorted(unaccounted_claims)}'
            )

        started_or_done = queue.completed | queue.claimed
        pending_ids = set(queue.moves) - started_or_done
        already_handled = task_map.keys() & started_or_done
        if already_handled:
            raise ValidationError(
                f'only unstarted moves can be replanned: {sorted(already_handled)}'
            )
        missing = pending_ids - task_map.keys()
        if missing:
            raise ValidationError(
                f'missing recovery tasks for pending moves: {sorted(missing)}'
            )

        fixed = self._validate_active(active_lifts, as_of)
        scheduled: list[PlannedRecoveryLift] = []
        unscheduled: list[UnscheduledRecoveryTask] = []
        remaining = set(task_map)
        scheduled_ids: set[str] = set()

        while remaining:
            # Only completion, a fixed in-progress lift, or a successful replan makes
            # successors eligible; failed tasks intentionally keep their successors out.
            satisfiable = set(queue.completed) | active_ids | scheduled_ids
            ready = [
                task_map[move_id]
                for move_id in remaining
                if task_map[move_id].predecessor_ids <= satisfiable
            ]
            if not ready:
                for move_id in sorted(remaining):
                    blocked_by = tuple(
                        sorted(task_map[move_id].predecessor_ids - satisfiable)
                    )
                    unscheduled.append(
                        UnscheduledRecoveryTask(
                            move_id,
                            UNSCHEDULED,
                            (PREDECESSOR_UNSCHEDULED,),
                            blocked_by,
                        )
                    )
                break

            ready.sort(
                key=lambda t: (
                    -t.priority,
                    t.earliest_start,
                    t.latest_finish,
                    t.bay,
                    -t.tier,
                    t.move_id,
                )
            )
            for task in ready:
                predecessor_finish = self._predecessor_finish(
                    fixed + scheduled, task.predecessor_ids, as_of
                )
                planned, reasons, blockers = self._place_task(
                    task, predecessor_finish, as_of, fixed + scheduled
                )
                remaining.remove(task.move_id)
                if planned is None:
                    unscheduled.append(
                        UnscheduledRecoveryTask(task.move_id, UNSCHEDULED, reasons, blockers)
                    )
                else:
                    scheduled.append(planned)
                    scheduled_ids.add(task.move_id)

        return RecoveryPlan(
            as_of,
            tuple(sorted(fixed, key=lambda x: (x.interval.start, x.rail_index, x.move_id))),
            tuple(sorted(scheduled, key=lambda x: (x.interval.start, x.rail_index, x.move_id))),
            tuple(sorted(unscheduled, key=lambda x: x.move_id)),
        )

    def apply_replan(self, queue: CraneWorkQueue, plan: RecoveryPlan) -> None:
        ordered = tuple(
            [item.move_id for item in plan.scheduled]
            + [item.move_id for item in plan.unscheduled]
        )
        queue.apply_replan(ordered)
