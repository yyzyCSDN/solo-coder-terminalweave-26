from __future__ import annotations
from dataclasses import dataclass
from terminalweave.core.errors import ValidationError

@dataclass(frozen=True)
class WorkMove:
    move_id: str
    bay: int
    tier: int
    priority: int
    predecessor_ids: frozenset[str] = frozenset()
    kind: str = 'discharge'

    def __post_init__(self) -> None:
        if self.bay < 1 or self.tier < 1:
            raise ValidationError('bay and tier must be positive')

class CraneWorkQueue:

    def __init__(self, moves: list[WorkMove]):
        if len({m.move_id for m in moves}) != len(moves):
            raise ValidationError('duplicate move id')
        ids = {m.move_id for m in moves}
        for m in moves:
            unknown = m.predecessor_ids - ids
            if unknown:
                raise ValidationError(f'unknown predecessors: {sorted(unknown)}')
        self.moves = {m.move_id: m for m in moves}
        self.completed: set[str] = set()
        self.claimed: set[str] = set()

    def ready(self) -> list[WorkMove]:
        return [m for m in self.moves.values() if m.move_id not in self.completed | self.claimed and m.predecessor_ids <= self.completed]

    def next(self, current_bay: int) -> WorkMove | None:
        ready = self.ready()
        if not ready:
            return None
        return min(ready, key=lambda m: (-m.priority, abs(m.bay - current_bay), -m.tier, m.move_id))

    def claim(self, move_id: str) -> WorkMove:
        move = self.moves[move_id]
        if move_id in self.completed or move_id in self.claimed:
            raise ValidationError('move is not claimable')
        if not move.predecessor_ids <= self.completed:
            raise ValidationError('move predecessors are incomplete')
        self.claimed.add(move_id)
        return move

    def complete(self, move_id: str) -> None:
        if move_id not in self.claimed:
            raise ValidationError('move was not claimed')
        self.claimed.remove(move_id)
        self.completed.add(move_id)

    def release(self, move_id: str) -> None:
        self.claimed.discard(move_id)

    def replan_after_stop(
        self,
        tasks: list,
        restart_at,
        active_lifts: list,
        cranes: list,
        safety_checkers: dict,
        rail_authority,
        rail_profiles: dict,
        increment=None,
        max_exact_tasks=None,
    ):
        """Replan unstarted moves while treating every claimed move as fixed."""
        from datetime import timedelta
        from terminalweave.cranes.recovery import CraneRecoveryPlanner
        planner = CraneRecoveryPlanner(
            cranes,
            safety_checkers,
            rail_authority,
            rail_profiles,
            increment or timedelta(minutes=5),
            max_exact_tasks if max_exact_tasks is not None else 12,
        )
        return planner.replan(self, tasks, restart_at, active_lifts)

    def replan(
        self,
        tasks: list,
        restart_at,
        active_lifts: list,
        cranes: list,
        safety_checkers: dict,
        rail_authority,
        rail_profiles: dict,
        increment=None,
        max_exact_tasks=None,
    ):
        return self.replan_after_stop(
            tasks,
            restart_at,
            active_lifts,
            cranes,
            safety_checkers,
            rail_authority,
            rail_profiles,
            increment,
            max_exact_tasks,
        )

    def progress(self) -> float:
        return len(self.completed) / len(self.moves) if self.moves else 1.0
