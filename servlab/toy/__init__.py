"""A toy inference engine: paged KV blocks, continuous batching, preemption.

Small enough to read in one sitting, faithful enough that the metrics it emits
have the same names and the same shapes as vLLM's — so lab 1's dashboard code
plots lab 3's toy engine without changes.
"""

from .allocator import Block, BlockAllocator, OutOfBlocks
from .scheduler import Request, Scheduler, SchedulerConfig

__all__ = ["Block", "BlockAllocator", "OutOfBlocks", "Request", "Scheduler", "SchedulerConfig"]
