import pytest

from servlab import napkin as nk


def test_llama_70b_kv_matches_the_drill():
    # 2 x 80 layers x 8 KV heads x 128 dim x 2 bytes = 327,680 B ~ 0.32 MB/token
    assert nk.kv_bytes_per_token("llama-3.3-70b") == 327_680
    # one 32K-context request is 10 GiB of KV — more than many people's weights
    assert nk.kv_bytes("llama-3.3-70b", 32_768) / nk.GIB == pytest.approx(10.0)


def test_concurrent_sessions_can_exceed_the_weights():
    # 100 concurrent 8K sessions on a 70B: KV dwarfs a 70 GB fp8 checkpoint.
    kv = nk.kv_bytes("llama-3.3-70b", 8192, batch=100) / nk.GIB
    assert kv > nk.weight_bytes("llama-3.3-70b", "fp8") / nk.GIB


def test_single_stream_decode_ceiling_on_an_h100():
    # 70 GB of fp8 weights over 3.35 TB/s ~ 48 tok/s, before any KV traffic.
    ceiling = 1 / nk.decode_step_time_s("H100-80GB", "llama-3.3-70b", batch=1, ctx_len=1,
                                        weight_dtype="fp8", kv_dtype="fp8", bw_efficiency=1.0)
    assert 44 < ceiling < 52


def test_tensor_parallelism_buys_latency_and_loses_efficiency():
    rows = nk.tp_scaling("H100-80GB", "llama-3.3-70b", batch=1, ctx_len=4096, weight_dtype="fp8")
    by_tp = {r["tp"]: r for r in rows}
    assert by_tp[8]["step_s"] < by_tp[1]["step_s"]          # it does get faster
    assert by_tp[8]["efficiency"] < by_tp[2]["efficiency"]  # and less efficient each time
    assert by_tp[1]["efficiency"] == pytest.approx(1.0)


def test_tensor_parallelism_pays_only_while_memory_dominates_communication():
    # TP splits the memory term and adds a fixed communication term, so whether
    # it helps is a comparison between those two — not a property of TP.
    def step(model, tp, comm):
        return nk.decode_step_time_tp_s("H100-80GB", model, tp=tp, allreduce_us_per_layer=comm)

    # Big model: the memory term is tens of milliseconds, so even a slow fabric
    # is still a net win — it just wastes most of the theoretical speedup.
    assert step("llama-3.3-70b", 8, 6) < step("llama-3.3-70b", 8, 60) < step("llama-3.3-70b", 1, 0)

    # Small model: the memory term is under a millisecond per card, so the
    # all-reduces cost more than the reads they saved. 8-way TP is now slower
    # than a single GPU. This is the regime where TP is cargo-culted.
    assert step("llama-3.2-1b", 8, 60) > step("llama-3.2-1b", 1, 0)


def test_slow_interconnect_destroys_tp_efficiency_even_when_it_still_helps():
    fast = nk.tp_scaling("H100-80GB", "llama-3.3-70b", tps=(8,), allreduce_us_per_layer=6)[0]
    slow = nk.tp_scaling("H100-80GB", "llama-3.3-70b", tps=(8,), allreduce_us_per_layer=60)[0]
    assert fast["efficiency"] > 0.7
    assert slow["efficiency"] < 0.5     # paying for 8 cards, getting under 4


def test_a_70b_fits_on_one_h100_only_once_quantized():
    assert not nk.fits("H100-80GB", "llama-3.3-70b", weight_dtype="fp16", seq_len=4096)
    assert nk.fits("H100-80GB", "llama-3.3-70b", weight_dtype="fp8", seq_len=4096)
    assert nk.fits_with_tp("H100-80GB", "llama-3.3-70b", weight_dtype="fp16", tp=4, seq_len=4096)


def test_speculative_decoding_collapses_at_low_acceptance():
    assert nk.spec_decode_speedup(0.9) > 2.0
    assert nk.spec_decode_speedup(0.7) > 1.0
    # Below ~0.5 the draft passes cost more than the tokens they save.
    assert nk.spec_decode_speedup(0.4) < 1.05
    with pytest.raises(ValueError):
        nk.spec_decode_speedup(1.5)


def test_newer_cards_are_bandwidth_upgrades_first():
    # The decode-relevant axis is bandwidth, not FLOPs — which is why an L4
    # with more FLOPs than a T4 does not decode much faster.
    h100 = nk.decode_tokens_per_s("H100-80GB", "llama-3.3-70b", batch=8, ctx_len=4096, weight_dtype="fp8")
    h200 = nk.decode_tokens_per_s("H200", "llama-3.3-70b", batch=8, ctx_len=4096, weight_dtype="fp8")
    assert h200 / h100 == pytest.approx(nk.GPUS["H200"].mem_bw_gb_s / nk.GPUS["H100-80GB"].mem_bw_gb_s, rel=0.01)


def test_tpu_v7_is_registered_with_bf16_and_fp8_support():
    tpu = nk.GPUS["TPU-v7"]
    assert tpu.supports_bf16 and tpu.supports_fp8
    assert tpu.vram_gb == 192
