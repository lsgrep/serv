"""Continuous batching with admission control and preemption.

Static batching waits for every sequence in a batch to finish before starting
the next batch, so one 500-token generation holds eight 20-token ones hostage.
Continuous batching admits a new request the step after a slot frees.

The loop below is the whole idea, and it is about forty lines:

    every step:
        admit from the waiting queue while there are seat and blocks
        for each running sequence, append one token
        if a sequence needs a block and none is free -> preempt someone
        retire finished sequences and free their blocks

Everything a production scheduler adds — chunked prefill, priority, prefix
cache reuse, swapping to host memory — is a refinement of these four lines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .allocator import BlockAllocator, OutOfBlocks


@dataclass
class Request:
    id: str
    prompt_len: int
    max_tokens: int
    arrival: float = 0.0
    generated: int = 0
    state: str = "waiting"  # waiting | running | finished | aborted
    start_time: float = 0.0  # first scheduled (prefill) — gives TTFT
    finish_time: float = 0.0
    preemptions: int = 0
    prefill_done: bool = False

    @property
    def n_tokens(self) -> int:
        return self.prompt_len + self.generated

    @property
    def done(self) -> bool:
        return self.generated >= self.max_tokens

    @property
    def ttft(self):
        return (self.start_time - self.arrival) if self.start_time else None

    @property
    def e2e(self):
        return (self.finish_time - self.arrival) if self.finish_time else None


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 32
    # Prefill tokens per step. vLLM's `--max-num-batched-tokens`; the knob that
    # trades TTFT for TPOT, because a big prefill stalls everyone's decode.
    max_num_batched_tokens: int = 2048
    # Refuse to admit if it would leave the pool under this fraction free.
    # Without a watermark, the engine admits a request and immediately preempts
    # someone to make room for it — thrash, not progress.
    watermark: float = 0.01
    policy: str = "fcfs"  # fcfs | sjf (shortest expected output first)
    preemption: str = "recompute"  # recompute | swap


@dataclass
class StepOutput:
    step: int
    prefilled: list = field(default_factory=list)
    decoded: list = field(default_factory=list)
    finished: list = field(default_factory=list)
    preempted: list = field(default_factory=list)
    prefill_tokens: int = 0


class Scheduler:
    """The engine's brain, with no model attached.

    Drive it with `step()` from a real engine (see `engine.py`) or from
    `simulate()` for a GPU-free reproduction of the same dynamics.
    """

    def __init__(self, allocator: BlockAllocator, config: SchedulerConfig = None):
        self.allocator = allocator
        self.config = config or SchedulerConfig()
        self.waiting = []
        self.running = []
        self.finished = []
        self.total_preemptions = 0
        self.total_gen_tokens = 0
        self.total_prompt_tokens = 0
        self.step_count = 0

    # -- queue ------------------------------------------------------------
    def add(self, req: Request):
        req.state = "waiting"
        self.waiting.append(req)
        return req

    def _pick_waiting(self):
        if self.config.policy == "sjf":
            self.waiting.sort(key=lambda r: r.max_tokens)
        return self.waiting

    def _watermark_blocks(self) -> int:
        return int(self.allocator.num_blocks * self.config.watermark)

    # -- the loop ---------------------------------------------------------
    def step(self, now=0.0) -> StepOutput:
        out = StepOutput(step=self.step_count)

        # 1. admit
        queue = self._pick_waiting()
        while queue and len(self.running) < self.config.max_num_seqs:
            req = queue[0]
            need = self.allocator.blocks_needed(req.n_tokens)
            if need > self.allocator.num_free - self._watermark_blocks():
                break  # head-of-line blocked: FCFS means we do not skip ahead
            if out.prefill_tokens + req.prompt_len > self.config.max_num_batched_tokens and out.prefilled:
                break  # this step's prefill budget is spent; next step will take it
            queue.pop(0)
            self.allocator.allocate(req.id, req.n_tokens)
            req.state = "running"
            if not req.start_time:
                req.start_time = now
            req.prefill_done = True
            self.running.append(req)
            out.prefilled.append(req)
            out.prefill_tokens += req.prompt_len
            self.total_prompt_tokens += req.prompt_len

        # 2. decode one token for everything already resident
        for req in list(self.running):
            if req in out.prefilled:
                continue  # its first token came out of prefill this step
            if req.state != "running":
                continue  # preempted earlier in this same step
            while True:
                try:
                    self.allocator.append_token(req.id, req.n_tokens + 1)
                    break
                except OutOfBlocks:
                    victim = self._preempt(now, avoid=req)
                    if victim is None:
                        break
                    out.preempted.append(victim)
                    if victim is req:
                        # Nothing left to evict but ourselves: back to the queue.
                        break
            if req.state != "running":
                continue
            req.generated += 1
            self.total_gen_tokens += 1
            out.decoded.append(req)

        # 3. retire
        for req in list(self.running):
            if req.done:
                req.state = "finished"
                req.finish_time = now
                self.allocator.free_seq(req.id)
                self.running.remove(req)
                self.finished.append(req)
                out.finished.append(req)

        self.step_count += 1
        return out

    def _preempt(self, now, avoid=None):
        """Evict the newest running request — the one that has invested the
        least work, so recomputation costs the least. (vLLM does the same.)

        `recompute` throws its KV away and re-prefills later; `swap` would copy
        it to host memory. Recompute is usually cheaper than a PCIe round trip,
        which is why the swap path is rarely the default any more.
        """
        if not self.running:
            return None
        # Newest first, but do not evict the request we are trying to serve
        # unless it is the only one left.
        victim = next((r for r in reversed(self.running) if r is not avoid), self.running[-1])
        self.allocator.free_seq(victim.id)
        self.running.remove(victim)
        victim.preemptions += 1
        victim.state = "waiting"
        if self.config.preemption == "recompute":
            # Its generated tokens are gone; the work is done twice. This is why
            # a preemption storm burns throughput without failing any request.
            victim.generated = 0
        self.waiting.insert(0, victim)
        self.total_preemptions += 1
        return victim

    # -- observability ----------------------------------------------------
    def stats(self, t=None) -> dict:
        """Same keys `servlab.prometheus.vllm_row` produces, so the toy engine
        and a real vLLM plot with identical code."""
        return {
            "t": t,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "swapped": 0,
            "kv_usage": self.allocator.usage * 100.0,
            "preemptions": self.total_preemptions,
            "prompt_tokens": self.total_prompt_tokens,
            "gen_tokens": self.total_gen_tokens,
            "finished": len(self.finished),
        }

    @property
    def empty(self) -> bool:
        return not self.waiting and not self.running


def simulate(*, rps=2.0, duration=60.0, prompt_len=256, output_len=128, num_blocks=512,
             block_size=16, step_time=0.02, config=None, seed=0, jitter=0.3):
    """Discrete-event run of the scheduler with a virtual clock — no GPU needed.

    Every step costs `step_time` seconds regardless of batch size, which is a
    deliberate simplification: decode is memory bound, so step time is nearly
    flat in batch size until the KV term grows. It is close enough that the
    queueing behaviour — including the death spiral — is real.

    Returns `(rows, requests)`: rows are dashboard-shaped, requests carry
    per-request TTFT/E2E.
    """
    rng = random.Random(seed)
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    sched = Scheduler(alloc, config or SchedulerConfig())

    arrivals, t = [], 0.0
    i = 0
    while t < duration:
        t += rng.expovariate(rps)
        if t >= duration:
            break
        plen = max(8, int(prompt_len * rng.uniform(1 - jitter, 1 + jitter)))
        olen = max(4, int(output_len * rng.uniform(1 - jitter, 1 + jitter)))
        arrivals.append(Request(id=f"r{i}", prompt_len=plen, max_tokens=olen, arrival=t))
        i += 1

    rows, requests = [], list(arrivals)
    now, pending = 0.0, list(arrivals)
    # Give the queue a bounded chance to drain after arrivals stop, so a
    # saturated run still terminates instead of spinning forever.
    horizon = duration * 3
    while now < horizon and (pending or not sched.empty):
        while pending and pending[0].arrival <= now:
            sched.add(pending.pop(0))
        sched.step(now=now)
        rows.append(sched.stats(t=now))
        now += step_time
    return rows, requests
