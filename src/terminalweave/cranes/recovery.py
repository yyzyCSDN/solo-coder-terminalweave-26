from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ConflictError, SafetyConstraintError, ValidationError
from terminalweave.core.intervals import TimeInterval
from terminalweave.cranes.rail_limits import CranePosition, CraneRailAuthority
from terminalweave.cranes.safety import LiftPlan, LiftSafetyChecker
from terminalweave.cranes.scheduler import Crane, ScheduledCraneJob
from terminalweave.cranes.workqueue import CraneWorkQueue, WorkMove


@dataclass(frozen=True)
class CraneRailProfile:
    crane_id: str
    wheel_load_t: float
    footprint_m: float

    def __post_init__(self) -> None:
        if self.wheel_load_t <= 0 or self.footprint_m <= 0:
            raise ValidationError('invalid crane rail profile')


@dataclass(frozen=True)
class RecoveryTask:
    move: WorkMove
    earliest_start: datetime
    latest_finish: datetime
    duration: timedelta
    lift_plan: LiftPlan
    rail_position_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'earliest_start', ensure_utc(self.earliest_start))
        object.__setattr__(self, 'latest_finish', ensure_utc(self.latest_finish))
        if self.duration.total_seconds() <= 0:
            raise ValidationError('recovery task duration must be positive')
        if self.latest_finish <= self.earliest_start:
            raise ValidationError('recovery task window is invalid')

    @property
    def move_id(self) -> str:
        return self.move.move_id


@dataclass(frozen=True)
class ActiveLift:
    move_id: str
    crane_id: str
    interval: TimeInterval
    rail_position_m: float
    lift_plan: LiftPlan


@dataclass(frozen=True)
class ReplanReason:
    code: str
    message: str
    crane_id: str | None = None
    blocking_move_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ActiveLiftIssue:
    move_id: str
    code: str
    message: str
    blocking_move_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ReplanAssignment:
    move_id: str
    crane_id: str
    rail_index: int
    bay: int
    interval: TimeInterval
    rail_position_m: float


@dataclass(frozen=True)
class ReplanFailure:
    move_id: str
    reasons: tuple[ReplanReason, ...]


@dataclass(frozen=True)
class ReplanResult:
    assignments: tuple[ReplanAssignment, ...]
    failures: tuple[ReplanFailure, ...]
    active_lifts: tuple[ActiveLift, ...]
    active_issues: tuple[ActiveLiftIssue, ...]

    @property
    def fully_recovered(self) -> bool:
        return not self.failures and not self.active_issues


@dataclass(frozen=True)
class _Commitment:
    job: ScheduledCraneJob
    rail_position_m: float
    wheel_load_t: float
    footprint_m: float


class CraneRecoveryPlanner:
    """Rebuild the unstarted crane plan around fixed, already-started lifts."""

    def __init__(
        self,
        cranes: Sequence[Crane],
        safety_checkers: Mapping[str, LiftSafetyChecker],
        rail_authority: CraneRailAuthority,
        rail_profiles: Mapping[str, CraneRailProfile],
        increment: timedelta = timedelta(minutes=5),
        max_exact_tasks: int = 12,
    ):
        if not cranes:
            raise ValidationError('at least one crane is required')
        if increment.total_seconds() <= 0:
            raise ValidationError('increment must be positive')
        if max_exact_tasks < 1:
            raise ValidationError('max_exact_tasks must be positive')
        self.cranes = {c.crane_id: c for c in cranes}
        if len(self.cranes) != len(list(cranes)):
            raise ValidationError('duplicate crane id')
        if set(safety_checkers) != set(self.cranes):
            raise ValidationError('every crane requires a safety checker')
        if set(rail_profiles) != set(self.cranes):
            raise ValidationError('every crane requires a rail profile')
        for crane_id, profile in rail_profiles.items():
            if profile.crane_id != crane_id:
                raise ValidationError('rail profile does not match crane id')
        self.safety_checkers = dict(safety_checkers)
        self.rail_authority = rail_authority
        self.rail_profiles = dict(rail_profiles)
        self.increment = increment
        self.max_exact_tasks = max_exact_tasks

    def replan(
        self,
        queue: CraneWorkQueue,
        tasks: Sequence[RecoveryTask],
        restart_at: datetime,
        active_lifts: Sequence[ActiveLift] | None = None,
    ) -> ReplanResult:
        restart_at = ensure_utc(restart_at)
        active_lifts = tuple(active_lifts or ())
        self._validate_state(queue, tasks, active_lifts, restart_at)

        tasks_by_id = {t.move_id: t for t in tasks}
        active_by_id = {a.move_id: a for a in active_lifts}
        active_issues = self._active_issues(queue, active_lifts)

        commitments = [self._active_commitment(queue, lift) for lift in active_lifts]
        assignments: dict[str, ReplanAssignment] = {}
        failures: dict[str, list[ReplanReason]] = {}

        for move_id in self._dependency_cycles(queue, set(tasks_by_id)):
            failures[move_id] = [self._reason(
                'dependency_cycle',
                f'move {move_id} is part of a predecessor cycle',
            )]

        candidate_ids = set(tasks_by_id) - set(failures)
        assignments, infeasible = self._search_assignments(
            queue,
            tasks_by_id,
            restart_at,
            active_by_id,
            commitments,
            candidate_ids,
        )
        commitments += [
            self._assignment_commitment(assignment, tasks_by_id[move_id])
            for move_id, assignment in sorted(assignments.items())
        ]
        for move_id in sorted(infeasible):
            blocked = sorted(
                p for p in queue.moves[move_id].predecessor_ids
                if p not in queue.completed
                and p not in active_by_id
                and p not in assignments
            )
            if blocked:
                failures[move_id] = [self._reason(
                    'predecessor_unscheduled',
                    f'move is blocked by unscheduled predecessors: {blocked}',
                    blockers=blocked,
                )]
                continue
            _, reasons = self._find_assignment(
                queue,
                tasks_by_id[move_id],
                restart_at,
                active_by_id,
                assignments,
                commitments,
            )
            failures[move_id] = reasons

        ordered_assignments = tuple(sorted(
            assignments.values(),
            key=lambda a: (a.interval.start, a.rail_index, a.move_id),
        ))
        ordered_failures = tuple(
            ReplanFailure(move_id, tuple(self._dedupe_reasons(failures[move_id])))
            for move_id in sorted(failures)
        )
        return ReplanResult(
            assignments=ordered_assignments,
            failures=ordered_failures,
            active_lifts=tuple(sorted(active_lifts, key=lambda a: (a.interval.start, a.move_id))),
            active_issues=tuple(active_issues),
        )

    def _validate_state(
        self,
        queue: CraneWorkQueue,
        tasks: Sequence[RecoveryTask],
        active_lifts: Sequence[ActiveLift],
        restart_at: datetime,
    ) -> None:
        tasks_by_id: dict[str, RecoveryTask] = {}
        pending = {
            move_id for move_id in queue.moves
            if move_id not in queue.completed | queue.claimed
        }
        for task in tasks:
            move_id = task.move_id
            if move_id in tasks_by_id:
                raise ValidationError(f'duplicate recovery task: {move_id}')
            if queue.moves.get(move_id) != task.move:
                raise ValidationError(f'recovery task does not match queued move: {move_id}')
            if move_id not in pending:
                raise ValidationError(f'move is not unstarted: {move_id}')
            tasks_by_id[move_id] = task
        if set(tasks_by_id) != pending:
            missing = sorted(pending - set(tasks_by_id))
            raise ValidationError(f'missing recovery tasks: {missing}')

        active_by_id: dict[str, ActiveLift] = {}
        for lift in active_lifts:
            if lift.move_id in active_by_id:
                raise ValidationError(f'duplicate active lift: {lift.move_id}')
            if lift.move_id not in queue.claimed:
                raise ValidationError(f'active lift is not claimed: {lift.move_id}')
            if lift.crane_id not in self.cranes:
                raise ValidationError(f'unknown crane: {lift.crane_id}')
            if lift.interval.start > restart_at or lift.interval.end <= restart_at:
                raise ValidationError(f'active lift does not straddle restart: {lift.move_id}')
            predecessors = queue.moves[lift.move_id].predecessor_ids
            if not predecessors <= queue.completed:
                raise ValidationError(f'active lift has incomplete predecessors: {lift.move_id}')
            active_by_id[lift.move_id] = lift
        if set(active_by_id) != queue.claimed:
            missing = sorted(queue.claimed - set(active_by_id))
            raise ValidationError(f'missing active lift states: {missing}')

    def _dependency_cycles(self, queue: CraneWorkQueue, pending: set[str]) -> set[str]:
        colors = {move_id: 0 for move_id in pending}
        cycles: set[str] = set()
        stack: list[str] = []

        def visit(move_id: str) -> None:
            colors[move_id] = 1
            stack.append(move_id)
            for predecessor in sorted(queue.moves[move_id].predecessor_ids):
                if predecessor not in colors:
                    continue
                if colors[predecessor] == 0:
                    visit(predecessor)
                elif colors[predecessor] == 1:
                    start = stack.index(predecessor)
                    cycles.update(stack[start:])
            colors[move_id] = 2
            stack.pop()

        for move_id in sorted(pending):
            if colors[move_id] == 0:
                visit(move_id)
        return cycles

    def _search_assignments(
        self,
        queue: CraneWorkQueue,
        tasks: Mapping[str, RecoveryTask],
        restart_at: datetime,
        active_lifts: Mapping[str, ActiveLift],
        initial_commitments: Sequence[_Commitment],
        candidate_ids: set[str],
    ) -> tuple[dict[str, ReplanAssignment], set[str]]:
        if len(candidate_ids) <= self.max_exact_tasks:
            return self._search_assignments_exact(
                queue,
                tasks,
                restart_at,
                active_lifts,
                initial_commitments,
                candidate_ids,
            )
        return self._search_assignments_greedy(
            queue,
            tasks,
            restart_at,
            active_lifts,
            initial_commitments,
            candidate_ids,
        )

    def _search_assignments_greedy(
        self,
        queue: CraneWorkQueue,
        tasks: Mapping[str, RecoveryTask],
        restart_at: datetime,
        active_lifts: Mapping[str, ActiveLift],
        initial_commitments: Sequence[_Commitment],
        candidate_ids: set[str],
    ) -> tuple[dict[str, ReplanAssignment], set[str]]:
        assignments: dict[str, ReplanAssignment] = {}
        commitments = list(initial_commitments)
        infeasible: set[str] = set()
        unresolved = set(candidate_ids)
        while unresolved:
            ready_ids = [
                move_id for move_id in unresolved
                if not (queue.moves[move_id].predecessor_ids & infeasible)
                and all(
                    predecessor in queue.completed
                    or predecessor in active_lifts
                    or predecessor in assignments
                    for predecessor in queue.moves[move_id].predecessor_ids
                )
            ]
            if not ready_ids:
                infeasible.update(unresolved)
                break
            move_id = min(
                ready_ids,
                key=lambda m: (
                    -queue.moves[m].priority,
                    tasks[m].latest_finish,
                    tasks[m].duration,
                    m,
                ),
            )
            assignment, _ = self._find_assignment(
                queue,
                tasks[move_id],
                restart_at,
                active_lifts,
                assignments,
                commitments,
            )
            unresolved.discard(move_id)
            if assignment is None:
                infeasible.add(move_id)
                continue
            assignments[move_id] = assignment
            commitments.append(self._assignment_commitment(assignment, tasks[move_id]))
        return assignments, infeasible

    def _search_assignments_exact(
        self,
        queue: CraneWorkQueue,
        tasks: Mapping[str, RecoveryTask],
        restart_at: datetime,
        active_lifts: Mapping[str, ActiveLift],
        initial_commitments: Sequence[_Commitment],
        candidate_ids: set[str],
    ) -> tuple[dict[str, ReplanAssignment], set[str]]:
        assignments: dict[str, ReplanAssignment] = {}
        commitments: list[_Commitment] = list(initial_commitments)
        infeasible: set[str] = set()
        best_assignments: dict[str, ReplanAssignment] = {}
        best_infeasible: set[str] = set(candidate_ids)
        best_score = 0

        def remember(remaining: set[str]) -> None:
            nonlocal best_score
            score = len(assignments)
            if score > best_score:
                best_score = score
                best_assignments.clear()
                best_assignments.update(assignments)
                best_infeasible.clear()
                best_infeasible.update(infeasible)
                best_infeasible.update(remaining)

        def ready(move_id: str) -> bool:
            return all(
                predecessor in queue.completed
                or predecessor in active_lifts
                or predecessor in assignments
                for predecessor in queue.moves[move_id].predecessor_ids
                if predecessor not in infeasible
            ) and not (queue.moves[move_id].predecessor_ids & infeasible)

        def search() -> bool:
            unresolved = candidate_ids - set(assignments) - infeasible
            if not unresolved:
                best_score = len(assignments)
                best_assignments.clear()
                best_assignments.update(assignments)
                best_infeasible.clear()
                best_infeasible.update(infeasible)
                return True
            ready_ids = [move_id for move_id in unresolved if ready(move_id)]
            if not ready_ids:
                remember(unresolved)
                return False

            move_id = min(
                ready_ids,
                key=lambda m: (
                    -queue.moves[m].priority,
                    tasks[m].latest_finish,
                    tasks[m].duration,
                    m,
                ),
            )
            options, _ = self._find_options(
                queue,
                tasks[move_id],
                restart_at,
                active_lifts,
                assignments,
                commitments,
            )
            if not options:
                infeasible.add(move_id)
                if search():
                    return True
                infeasible.discard(move_id)
                remember(unresolved)
                return False

            for option in options:
                assignments[move_id] = option
                commitment = self._assignment_commitment(option, tasks[move_id])
                commitments.append(commitment)
                if search():
                    return True
                assignments.pop(move_id)
                commitments.remove(commitment)
            remember(unresolved)
            return False

        search()
        return best_assignments, best_infeasible

    def _find_assignment(
        self,
        queue: CraneWorkQueue,
        task: RecoveryTask,
        restart_at: datetime,
        active_lifts: Mapping[str, ActiveLift],
        assignments: Mapping[str, ReplanAssignment],
        commitments: Sequence[_Commitment],
    ) -> tuple[ReplanAssignment | None, list[ReplanReason]]:
        options, reasons = self._find_options(
            queue,
            task,
            restart_at,
            active_lifts,
            assignments,
            commitments,
        )
        return (options[0], []) if options else (None, reasons)

    def _find_options(
        self,
        queue: CraneWorkQueue,
        task: RecoveryTask,
        restart_at: datetime,
        active_lifts: Mapping[str, ActiveLift],
        assignments: Mapping[str, ReplanAssignment],
        commitments: Sequence[_Commitment],
    ) -> tuple[list[ReplanAssignment], list[ReplanReason]]:
        earliest = max(restart_at, task.earliest_start)
        for predecessor in task.move.predecessor_ids:
            if predecessor in active_lifts:
                earliest = max(earliest, active_lifts[predecessor].interval.end)
            elif predecessor in assignments:
                earliest = max(earliest, assignments[predecessor].interval.end)

        reasons: list[ReplanReason] = []
        if earliest + task.duration > task.latest_finish:
            return [], [self._reason(
                'outside_window',
                f'earliest start {earliest.isoformat()} with duration {task.duration} '
                f'misses latest finish {task.latest_finish.isoformat()}',
            )]

        candidates: list[tuple[datetime, int, str, ReplanAssignment]] = []
        for crane in sorted(self.cranes.values(), key=lambda c: (c.rail_index, c.crane_id)):
            static_reasons = self._static_reasons(crane, task)
            if static_reasons:
                reasons.extend(static_reasons)
                continue

            cursor = earliest
            feasible_here = False
            slot_checked = False
            while cursor + task.duration <= task.latest_finish:
                slot_checked = True
                interval = TimeInterval(cursor, cursor + task.duration)
                candidate = ScheduledCraneJob(
                    crane.crane_id,
                    crane.rail_index,
                    task.move_id,
                    task.move.bay,
                    interval,
                    1,
                )
                conflicts, blockers = self._candidate_conflicts(
                    crane,
                    task.rail_position_m,
                    candidate,
                    commitments,
                )
                if not conflicts:
                    assignment = ReplanAssignment(
                        task.move_id,
                        crane.crane_id,
                        crane.rail_index,
                        task.move.bay,
                        interval,
                        task.rail_position_m,
                    )
                    candidates.append((interval.end, crane.rail_index, crane.crane_id, assignment))
                    feasible_here = True
                    break
                reasons.extend(conflicts)
                cursor = max(
                    [cursor + self.increment]
                    + [commitment.job.interval.end for commitment in blockers]
                )
            if not feasible_here and not slot_checked:
                reasons.append(self._reason(
                    'outside_window',
                    f'crane {crane.crane_id} has no slot before {task.latest_finish.isoformat()}',
                    crane.crane_id,
                ))

        if not candidates:
            return [], self._dedupe_reasons(reasons) or [self._reason(
                'no_feasible_assignment',
                'no crane, time, safety, and rail combination satisfies the move',
            )]

        options = [
            assignment
            for _, _, _, assignment in sorted(
                candidates,
                key=lambda x: (x[0], x[1], x[2]),
            )
        ]
        return options, []

    def _static_reasons(self, crane: Crane, task: RecoveryTask) -> list[ReplanReason]:
        reasons: list[ReplanReason] = []
        try:
            self.safety_checkers[crane.crane_id].validate(task.lift_plan)
        except SafetyConstraintError as exc:
            reasons.append(self._reason(
                'lift_envelope',
                str(exc),
                crane.crane_id,
            ))
        except ValidationError as exc:
            reasons.append(self._reason(
                'invalid_lift_plan',
                str(exc),
                crane.crane_id,
            ))
        reasons.extend(self._rail_reasons(crane, task.rail_position_m, task.move_id))
        return reasons

    def _rail_reasons(self, crane: Crane, position_m: float, owner: str) -> list[ReplanReason]:
        profile = self.rail_profiles[crane.crane_id]
        left = position_m - profile.footprint_m / 2
        right = position_m + profile.footprint_m / 2
        covered = sorted((
            segment for segment in self.rail_authority.segments
            if segment.start_m < right and left < segment.end_m
        ), key=lambda segment: segment.start_m)
        reasons: list[ReplanReason] = []
        if not covered:
            reasons.append(self._reason(
                'rail_outside_range',
                f'crane footprint {left:.3f}-{right:.3f} is outside configured rail',
                crane.crane_id,
            ))
            return reasons
        covered_until = left
        for segment in covered:
            if segment.start_m > covered_until:
                reasons.append(self._reason(
                    'rail_outside_range',
                    f'crane footprint {left:.3f}-{right:.3f} crosses unsupported rail',
                    crane.crane_id,
                ))
                break
            covered_until = max(covered_until, min(right, segment.end_m))
        if covered_until < right:
            reasons.append(self._reason(
                'rail_outside_range',
                f'crane footprint {left:.3f}-{right:.3f} extends past configured rail',
                crane.crane_id,
            ))
        if any(segment.closed for segment in covered):
            reasons.append(self._reason(
                'rail_closed',
                'crane footprint intersects closed rail',
                crane.crane_id,
            ))
        if any(profile.wheel_load_t > segment.max_wheel_load_t for segment in covered):
            reasons.append(self._reason(
                'rail_load_limit',
                f'wheel load {profile.wheel_load_t} exceeds a rail segment limit',
                crane.crane_id,
            ))
        try:
            self.rail_authority.validate(CranePosition(
                crane.crane_id,
                position_m,
                profile.wheel_load_t,
                profile.footprint_m,
            ))
        except ConflictError as exc:
            if not reasons:
                reasons.append(self._reason('rail_invalid', str(exc), crane.crane_id))
        return reasons

    def _candidate_conflicts(
        self,
        crane: Crane,
        rail_position_m: float,
        candidate: ScheduledCraneJob,
        commitments: Sequence[_Commitment],
    ) -> tuple[list[ReplanReason], list[_Commitment]]:
        reasons: list[ReplanReason] = []
        blocking_commitments: list[_Commitment] = []
        profile = self.rail_profiles[crane.crane_id]
        left = rail_position_m - profile.footprint_m / 2
        right = rail_position_m + profile.footprint_m / 2
        for commitment in commitments:
            if not candidate.interval.overlaps(commitment.job.interval):
                continue
            blockers = frozenset({commitment.job.job_id})
            if candidate.crane_id == commitment.job.crane_id:
                blocking_commitments.append(commitment)
                reasons.append(self._reason(
                    'crane_busy',
                    f'crane is occupied by {commitment.job.job_id}',
                    crane.crane_id,
                    blockers,
                ))
            elif self._interferes(candidate, commitment.job):
                blocking_commitments.append(commitment)
                reasons.append(self._reason(
                    'crane_envelope',
                    f'crane safety envelope conflicts with {commitment.job.job_id}',
                    crane.crane_id,
                    blockers,
                ))
            occupied_left = commitment.rail_position_m - commitment.footprint_m / 2
            occupied_right = commitment.rail_position_m + commitment.footprint_m / 2
            if left < occupied_right and occupied_left < right:
                if commitment not in blocking_commitments:
                    blocking_commitments.append(commitment)
                reasons.append(self._reason(
                    'rail_occupied',
                    f'rail footprint is occupied by {commitment.job.job_id}',
                    crane.crane_id,
                    blockers,
                ))
        return reasons, blocking_commitments

    def _interferes(self, candidate: ScheduledCraneJob, existing: ScheduledCraneJob) -> bool:
        candidate_crane = self.cranes[candidate.crane_id]
        existing_crane = self.cranes[existing.crane_id]
        if abs(candidate.bay - existing.bay) < max(
            candidate_crane.min_separation_bays,
            existing_crane.min_separation_bays,
        ):
            return True
        if candidate_crane.rail_index < existing_crane.rail_index and candidate.bay > existing.bay:
            return True
        if candidate_crane.rail_index > existing_crane.rail_index and candidate.bay < existing.bay:
            return True
        return False

    def _active_issues(self, queue: CraneWorkQueue, active_lifts: Sequence[ActiveLift]) -> list[ActiveLiftIssue]:
        issues: list[ActiveLiftIssue] = []
        for lift in active_lifts:
            crane = self.cranes[lift.crane_id]
            try:
                self.safety_checkers[crane.crane_id].validate(lift.lift_plan)
            except (SafetyConstraintError, ValidationError) as exc:
                code = 'lift_envelope' if isinstance(exc, SafetyConstraintError) else 'invalid_lift_plan'
                issues.append(ActiveLiftIssue(lift.move_id, code, str(exc)))
            for reason in self._rail_reasons(crane, lift.rail_position_m, lift.move_id):
                issues.append(ActiveLiftIssue(
                    lift.move_id,
                    reason.code,
                    reason.message,
                    reason.blocking_move_ids,
                ))

        def add_pair_issue(first_id: str, second_id: str, code: str, message: str) -> None:
            issues.append(ActiveLiftIssue(
                first_id,
                code,
                message.format(move_id=second_id),
                frozenset({second_id}),
            ))
            issues.append(ActiveLiftIssue(
                second_id,
                code,
                message.format(move_id=first_id),
                frozenset({first_id}),
            ))

        for index, first in enumerate(active_lifts):
            first_job = self._active_job(queue, first)
            for second in active_lifts[index + 1:]:
                if not first.interval.overlaps(second.interval):
                    continue
                if first.crane_id == second.crane_id:
                    add_pair_issue(
                        first.move_id,
                        second.move_id,
                        'active_crane_busy',
                        'crane is also occupied by {move_id}',
                    )
                elif self._interferes(first_job, self._active_job(queue, second)):
                    add_pair_issue(
                        first.move_id,
                        second.move_id,
                        'active_crane_envelope',
                        'crane safety envelope conflicts with active lift {move_id}',
                    )
                first_profile = self.rail_profiles[first.crane_id]
                second_profile = self.rail_profiles[second.crane_id]
                first_left = first.rail_position_m - first_profile.footprint_m / 2
                first_right = first.rail_position_m + first_profile.footprint_m / 2
                second_left = second.rail_position_m - second_profile.footprint_m / 2
                second_right = second.rail_position_m + second_profile.footprint_m / 2
                if first_left < second_right and second_left < first_right:
                    add_pair_issue(
                        first.move_id,
                        second.move_id,
                        'active_rail_conflict',
                        'rail footprint overlaps active lift {move_id}',
                    )
        return sorted(issues, key=lambda i: (i.move_id, i.code, sorted(i.blocking_move_ids)))

    def _active_job(self, queue: CraneWorkQueue, lift: ActiveLift) -> ScheduledCraneJob:
        crane = self.cranes[lift.crane_id]
        return ScheduledCraneJob(
            lift.crane_id,
            crane.rail_index,
            lift.move_id,
            queue.moves[lift.move_id].bay,
            lift.interval,
            1,
        )

    def _active_commitment(self, queue: CraneWorkQueue, lift: ActiveLift) -> _Commitment:
        profile = self.rail_profiles[lift.crane_id]
        return _Commitment(
            self._active_job(queue, lift),
            lift.rail_position_m,
            profile.wheel_load_t,
            profile.footprint_m,
        )

    def _assignment_commitment(self, assignment: ReplanAssignment, task: RecoveryTask) -> _Commitment:
        profile = self.rail_profiles[assignment.crane_id]
        return _Commitment(
            ScheduledCraneJob(
                assignment.crane_id,
                assignment.rail_index,
                assignment.move_id,
                assignment.bay,
                assignment.interval,
                1,
            ),
            task.rail_position_m,
            profile.wheel_load_t,
            profile.footprint_m,
        )

    def _reason(
        self,
        code: str,
        message: str,
        crane_id: str | None = None,
        blockers: Sequence[str] | frozenset[str] = (),
    ) -> ReplanReason:
        return ReplanReason(code, message, crane_id, frozenset(blockers))

    def _dedupe_reasons(self, reasons: Sequence[ReplanReason]) -> list[ReplanReason]:
        seen: set[tuple[object, ...]] = set()
        out: list[ReplanReason] = []
        for reason in reasons:
            key = (
                reason.code,
                reason.message,
                reason.crane_id,
                tuple(sorted(reason.blocking_move_ids)),
            )
            if key not in seen:
                seen.add(key)
                out.append(reason)
        return out
