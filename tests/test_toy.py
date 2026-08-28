import pytest

from servlab.toy import BlockAllocator, OutOfBlocks, Request, Scheduler, SchedulerConfig
from servlab.toy.scheduler import simulate


def test_blocks_are_allocated_by_ceiling():
    a = BlockAllocator(num_blocks=10, block_size=16)
    a.allocate("s", 17)
    assert len(a.blocks_for("s")) == 2
    assert a.num_used == 2
    assert a.slots_left("s", 17) == 15  # internal fragmentation, bounded by one block


def test_append_only_takes_a_block_at_a_boundary():
    a = BlockAllocator(num_blocks=10, block_size=16)
    a.allocate("s", 16)
    assert a.append_token("s", 17) is True   # crosses into a second block
    assert a.append_token("s", 18) is False  # still inside it


def test_exhaustion_raises_rather_than_over_committing():
    a = BlockAllocator(num_blocks=2, block_size=16)
    a.allocate("s1", 32)
    with pytest.raises(OutOfBlocks):
        a.allocate("s2", 1)
    a.free_seq("s1")
    a.allocate("s2", 1)  # freeing returns the blocks to the pool


def test_paged_fragmentation_is_bounded_by_one_block_per_sequence():
    a = BlockAllocator(num_blocks=64, block_size=16)
    for i in range(4):
        a.allocate(f"s{i}", 20)
    waste = a.fragmentation({f"s{i}": 20 for i in range(4)})
    # 4 sequences x 32 slots allocated for 20 tokens each
    assert waste == pytest.approx((128 - 80) / 128)
    assert waste < 15 / 16


def test_continuous_batching_admits_as_slots_free():
    sched = Scheduler(BlockAllocator(num_blocks=64, block_size=16),
                      SchedulerConfig(max_num_seqs=2))
    for i in range(4):
        sched.add(Request(id=f"r{i}", prompt_len=16, max_tokens=2))
    out = sched.step()
    assert len(out.prefilled) == 2 and len(sched.waiting) == 2
    for _ in range(6):
        sched.step()
    assert len(sched.finished) == 4  # later arrivals ran without waiting for a batch


def test_watermark_keeps_the_pool_from_being_admitted_to_death():
    sched = Scheduler(BlockAllocator(num_blocks=8, block_size=16),
                      SchedulerConfig(max_num_seqs=32, watermark=0.25))
    for i in range(8):
        sched.add(Request(id=f"r{i}", prompt_len=16, max_tokens=8))
    sched.step()
    assert sched.allocator.num_free >= 2  # 25% of 8 blocks held back
    assert sched.waiting  # the rest are queued rather than admitted-then-evicted


def test_preemption_recomputes_and_is_counted():
    sched = Scheduler(BlockAllocator(num_blocks=4, block_size=4),
                      SchedulerConfig(max_num_seqs=8, watermark=0.0))
    for i in range(4):
        sched.add(Request(id=f"r{i}", prompt_len=4, max_tokens=40))
    for _ in range(30):
        sched.step()
    assert sched.total_preemptions > 0
    victim = next(r for r in sched.waiting + sched.running if r.preemptions)
    # recompute mode throws the generated tokens away — that is the cost
    assert victim.generated < victim.max_tokens


def test_everything_finishes_when_there_is_room():
    sched = Scheduler(BlockAllocator(num_blocks=256, block_size=16),
                      SchedulerConfig(max_num_seqs=8))
    for i in range(8):
        sched.add(Request(id=f"r{i}", prompt_len=32, max_tokens=10))
    for _ in range(200):
        sched.step()
        if sched.empty:
            break
    assert len(sched.finished) == 8
    assert sched.allocator.num_free == sched.allocator.num_blocks  # no leaks


def test_stats_use_the_same_keys_as_the_vllm_scrape():
    from servlab.prometheus import vllm_row

    sched = Scheduler(BlockAllocator(num_blocks=16))
    toy_keys = set(sched.stats(t=0).keys())
    assert toy_keys <= set(vllm_row("").keys())


def test_simulation_reproduces_the_death_spiral():
    calm, calm_reqs = simulate(rps=1.0, duration=25, num_blocks=256, seed=1)
    busy, busy_reqs = simulate(rps=14.0, duration=25, num_blocks=256, seed=1)

    def p50_ttft(reqs):
        vals = sorted(r.ttft for r in reqs if r.ttft is not None)
        return vals[len(vals) // 2] if vals else None

    assert max(r["waiting"] for r in calm) <= 2
    assert max(r["waiting"] for r in busy) > 20        # queue grows without bound
    assert busy[-1]["preemptions"] > calm[-1]["preemptions"]
    assert p50_ttft(busy_reqs) > 10 * p50_ttft(calm_reqs)        # latency, not errors, is how it fails
