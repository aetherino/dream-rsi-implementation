"""Policies see current cache metadata only, never future trace data.

New policies implement choose() and optionally observe_insert/hit/evict hooks.
Hooks and victim selection are timed by the engine. Candidate execution is
in-process and trusted; this is NOT an isolation boundary for generated code.
"""
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class Entry:
    block_id: int  # Opaque ID; its numerical value must not be used as a signal.
    depth: int
    inserted_at: float
    last_access: float
    frequency: int
    insertion_order: int
    access_order: int
    # Metadata of the most recent request touching this prefix, including shared prefixes.
    task_chat: bool = False
    task_qa: bool = False
    task_unknown: bool = True
    turn_index: int = 0


class Policy:
    name = "custom"

    def observe_insert(self, entry: Entry) -> None:
        pass

    def observe_hit(self, entry: Entry) -> None:
        pass

    def observe_evict(self, entry: Entry) -> None:
        pass

    def choose(self, eligible: Sequence[Entry], now: float) -> int:
        raise NotImplementedError


class LRU(Policy):
    name = "lru"

    def choose(self, eligible: Sequence[Entry], now: float) -> int:
        return min(eligible, key=lambda e: (e.last_access, e.access_order)).block_id


class LFU(Policy):
    name = "lfu"

    def choose(self, eligible: Sequence[Entry], now: float) -> int:
        return min(eligible, key=lambda e: (e.frequency, e.last_access, e.access_order)).block_id


class FIFO(Policy):
    name = "fifo"

    def choose(self, eligible: Sequence[Entry], now: float) -> int:
        return min(eligible, key=lambda e: e.insertion_order).block_id


BASELINES = {p.name: p for p in (LRU, LFU, FIFO)}
