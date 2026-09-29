from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, ConflictError
from terminalweave.core.intervals import TimeInterval

@dataclass(frozen=True)
class Crane:
    crane_id: str
    rail_index: int
    moves_per_hour: float
    min_separation_bays: int = 2

    def __post_init__(self) -> None:
        if self.rail_index < 0 or self.moves_per_hour <= 0 or self.min_separation_bays < 0:
            raise ValidationError('invalid crane configuration')

@dataclass(frozen=True)
class CraneJob:
    job_id: str
    call_id: str
    bay: int
    moves: int
    earliest_start: datetime
    latest_finish: datetime
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'earliest_start', ensure_utc(self.earliest_start))
        object.__setattr__(self, 'latest_finish', ensure_utc(self.latest_finish))
        if self.bay < 1 or self.moves <= 0 or self.latest_finish <= self.earliest_start:
            raise ValidationError('invalid crane job')

@dataclass(frozen=True)
class ScheduledCraneJob:
    crane_id: str
    rail_index: int
    job_id: str
    bay: int
    interval: TimeInterval
    moves: int

def crane_jobs_interfere(
    candidate: ScheduledCraneJob,
    existing: ScheduledCraneJob,
    cranes: dict[str, Crane],
) -> bool:
    if not candidate.interval.overlaps(existing.interval):
        return False
    c = cranes[candidate.crane_id]
    e = cranes[existing.crane_id]
    if abs(candidate.bay - existing.bay) < max(c.min_separation_bays, e.min_separation_bays):
        return True
    if candidate.rail_index < existing.rail_index and candidate.bay > existing.bay:
        return True
    if candidate.rail_index > existing.rail_index and candidate.bay < existing.bay:
        return True
    return False


def scheduled_jobs_conflict(
    candidate: ScheduledCraneJob,
    existing: ScheduledCraneJob,
    cranes: dict[str, Crane],
) -> bool:
    return (
        candidate.crane_id == existing.crane_id
        and candidate.interval.overlaps(existing.interval)
    ) or crane_jobs_interfere(candidate, existing, cranes)


class CraneScheduler:

    def __init__(self, cranes: list[Crane]):
        if not cranes:
            raise ValidationError('at least one crane is required')
        if len({c.crane_id for c in cranes}) != len(cranes):
            raise ValidationError('duplicate crane id')
        self.cranes = {c.crane_id: c for c in cranes}
        self._scheduled: list[ScheduledCraneJob] = []

    @property
    def scheduled(self) -> tuple[ScheduledCraneJob, ...]:
        return tuple(sorted(self._scheduled, key=lambda x: (x.interval.start, x.rail_index)))

    def _duration(self, crane: Crane, job: CraneJob) -> timedelta:
        return timedelta(hours=job.moves / crane.moves_per_hour)

    def can_schedule(self, crane: Crane, job: CraneJob, start: datetime) -> bool:
        start = ensure_utc(start)
        finish = start + self._duration(crane, job)
        if start < job.earliest_start or finish > job.latest_finish:
            return False
        candidate = ScheduledCraneJob(crane.crane_id, crane.rail_index, job.job_id, job.bay, TimeInterval(start, finish), job.moves)
        for existing in self._scheduled:
            if scheduled_jobs_conflict(candidate, existing, self.cranes):
                return False
        return True

    def schedule(self, job: CraneJob, increment: timedelta=timedelta(minutes=5)) -> ScheduledCraneJob:
        if increment.total_seconds() <= 0:
            raise ValidationError('increment must be positive')
        candidates: list[tuple[datetime, float, Crane]] = []
        for crane in self.cranes.values():
            cursor = job.earliest_start
            duration = self._duration(crane, job)
            while cursor + duration <= job.latest_finish:
                if self.can_schedule(crane, job, cursor):
                    candidates.append((cursor + duration, -crane.moves_per_hour, crane))
                    break
                cursor += increment
        if not candidates:
            raise ConflictError('no crane can execute job within its window')
        finish, _, crane = min(candidates, key=lambda x: (x[0], x[1], x[2].rail_index))
        start = finish - self._duration(crane, job)
        scheduled = ScheduledCraneJob(crane.crane_id, crane.rail_index, job.job_id, job.bay, TimeInterval(start, finish), job.moves)
        self._scheduled.append(scheduled)
        return scheduled

    def remove_job(self, job_id: str) -> bool:
        before = len(self._scheduled)
        self._scheduled = [s for s in self._scheduled if s.job_id != job_id]
        return len(self._scheduled) != before

    def crane_workload(self, crane_id: str, interval: TimeInterval) -> float:
        seconds = 0.0
        for scheduled in self._scheduled:
            if scheduled.crane_id != crane_id:
                continue
            overlap = scheduled.interval.intersection(interval)
            if overlap:
                seconds += overlap.duration.total_seconds()
        return min(1.0, seconds / interval.duration.total_seconds())
