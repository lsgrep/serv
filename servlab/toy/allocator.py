"""Paged KV allocation, the idea PagedAttention is named after.

Classic serving reserved a contiguous `max_model_len` slab per sequence, so a
request that generated 20 tokens still held 2048 tokens' worth of memory. Paging
hands out fixed-size blocks on demand, so waste is bounded by *one partly-filled
block per sequence* instead of the whole reservation.

That single change is most of the throughput difference between a 2022 server
and a 2024 one, and it is small enough to implement here in 80 lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OutOfBlocks(Exception):
    """Raised when the KV pool is exhausted. In a real engine this is not an
    error — it is the signal to preempt someone."""


@dataclass
class Block:
    id: int
    ref_count: int = 0


@dataclass
class BlockAllocator:
    """A free list of fixed-size KV blocks.

    `block_size` is tokens per block (vLLM defaults to 16). `num_blocks` comes
    straight from the napkin math: KV budget / (block_size * kv_bytes_per_token).
    """

    num_blocks: int
    block_size: int = 16
    free: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)  # seq_id -> [block ids]

    def __post_init__(self):
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if not self.free:
            self.free = list(range(self.num_blocks))

    # -- accounting -------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    @property
    def usage(self) -> float:
        """0..1, the same quantity vLLM exports as `kv_cache_usage_perc`."""
        return self.num_used / self.num_blocks

    def blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)  # ceil

    def blocks_for(self, seq_id) -> list:
        return self.tables.get(seq_id, [])

    def slots_left(self, seq_id, n_tokens) -> int:
        """Free slots in the last block of a sequence — the internal
        fragmentation that paging deliberately accepts."""
        allocated = len(self.blocks_for(seq_id)) * self.block_size
        return allocated - n_tokens

    # -- mutation ---------------------------------------------------------
    def can_allocate(self, n_tokens: int) -> bool:
        return self.blocks_needed(n_tokens) <= self.num_free

    def allocate(self, seq_id, n_tokens: int) -> list:
        """Reserve blocks for a sequence's prompt."""
        if seq_id in self.tables:
            raise ValueError(f"{seq_id} already allocated")
        need = self.blocks_needed(n_tokens)
        if need > self.num_free:
            raise OutOfBlocks(f"need {need} blocks, {self.num_free} free")
        ids = [self.free.pop() for _ in range(need)]
        self.tables[seq_id] = ids
        return ids

    def append_token(self, seq_id, n_tokens_after) -> bool:
        """Grow a sequence by one token. Returns True if a new block was taken.

        This is where decode meets memory pressure: most steps are free, and
        every `block_size`-th step needs a block that may not exist.
        """
        if seq_id not in self.tables:
            raise KeyError(seq_id)
        if self.blocks_needed(n_tokens_after) <= len(self.tables[seq_id]):
            return False
        if not self.free:
            raise OutOfBlocks(f"{seq_id} needs a new block, pool is full")
        self.tables[seq_id].append(self.free.pop())
        return True

    def free_seq(self, seq_id):
        for b in self.tables.pop(seq_id, []):
            self.free.append(b)

    def fragmentation(self, lengths) -> float:
        """Fraction of allocated slots that hold no token.

        Bounded by (block_size - 1) / block_size *per sequence* — compare that
        with the contiguous-reservation waste of (max_len - len) / max_len.
        """
        allocated = sum(len(v) for v in self.tables.values()) * self.block_size
        used = sum(lengths.get(s, 0) for s in self.tables)
        return 0.0 if allocated == 0 else (allocated - used) / allocated
