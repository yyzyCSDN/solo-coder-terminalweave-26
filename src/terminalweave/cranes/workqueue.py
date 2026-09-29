from __future__ import annotations
from dataclasses import dataclass
from terminalweave.core.errors import NotFoundError, ValidationError

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
        self.replan_order: tuple[str, ...] = ()

    @property
    def pending_ids(self) -> set[str]:
        return set(self.moves) - self.completed - self.claimed

    @property
    def active_ids(self) -> set[str]:
        return set(self.claimed)

    def apply_replan(self, ordered_pending_ids: tuple[str, ...] | list[str]) -> None:
        ordered = tuple(ordered_pending_ids)
        if len(ordered) != len(set(ordered)):
            raise ValidationError('duplicate move in replan order')
        unknown = set(ordered) - self.moves.keys()
        if unknown:
            raise ValidationError(f'unknown moves: {sorted(unknown)}')
        pending = self.pending_ids
        if set(ordered) != pending:
            raise ValidationError('replan order must contain every pending move exactly once')
        self.replan_order = ordered

    def _ordered_pending(self, moves: list[WorkMove]) -> list[WorkMove]:
        if not self.replan_order:
            return moves
        rank = {move_id: index for index, move_id in enumerate(self.replan_order)}
        return sorted(moves, key=lambda move: rank.get(move.move_id, len(rank)))

    def status(self, move_id: str) -> str:
        if move_id not in self.moves:
            raise NotFoundError('unknown move')
        if move_id in self.completed:
            return 'completed'
        if move_id in self.claimed:
            return 'in_progress'
        return 'pending'

    def ready(self) -> list[WorkMove]:
        return [m for m in self.moves.values() if m.move_id not in self.completed | self.claimed and m.predecessor_ids <= self.completed]

    def next(self, current_bay: int) -> WorkMove | None:
        ready = self._ordered_pending(self.ready())
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

    def progress(self) -> float:
        return len(self.completed) / len(self.moves) if self.moves else 1.0
