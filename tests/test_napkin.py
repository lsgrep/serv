import math

import pytest

from servlab import napkin as nk


def test_kv_bytes_per_token_matches_hand_calculation():
    # GPT-2: 2 (K,V) x 12 layers x 12 heads x 64 dim x 2 bytes = 36 KiB/token
    assert nk.kv_bytes_per_token("gpt2") == 2 * 12 * 12 * 64 * 2 == 36 * 1024


def test_gqa_discounts_the_kv_cache():
    # Llama-3.1-8B: 32 query heads share 8 KV heads, so KV is 4x smaller than
    # the same model would need with full multi-head attention.
    spec = nk.MODELS["llama-3.1-8b"]
    assert spec.gqa_ratio == 4
    assert nk.kv_bytes_per_token(spec) == 128 * 1024

    mha = nk.ModelSpec(**{**spec.__dict__, "n_kv_heads": spec.n_heads})
    assert nk.kv_bytes_per_token(mha) == 4 * nk.kv_bytes_per_token(spec)


def test_fp8_kv_halves_the_cache():
    assert nk.kv_bytes_per_token("llama-3.1-8b", "fp8") == nk.kv_bytes_per_token("llama-3.1-8b") / 2


def test_8b_fp16_does_not_fit_on_a_t4():
    # The gotcha the labs are built around: weights alone exceed the card.
    assert nk.weight_bytes("llama-3.1-8b") > 14 * nk.GIB
    assert nk.kv_budget_bytes("T4", "llama-3.1-8b") == 0
    assert not nk.fits("T4", "llama-3.1-8b", seq_len=2048)


def test_3b_fp16_leaves_room_for_real_concurrency_on_a_t4():
    seqs = nk.max_concurrent_sequences("T4", "qwen2.5-3b", seq_len=2048)
    assert 50 < seqs < 300  # enough headroom that the lab is about queueing, not OOM


def test_awq_weights_are_more_than_four_bits_per_param():
    spec = nk.MODELS["llama-3.1-8b"]
    awq = nk.awq_weight_bytes(spec)
    assert awq > nk.weight_bytes(spec, "int4")  # scales and zero-points are not free
    assert awq < nk.weight_bytes(spec, "fp16") / 3


def test_batching_amortises_weight_reads():
    # Decode reads the weights once per step regardless of batch size, so
    # tokens/s should climb steeply with batch while KV traffic is small.
    t1 = nk.decode_tokens_per_s("A100-40GB", "llama-3.1-8b", batch=1, ctx_len=512)
    t32 = nk.decode_tokens_per_s("A100-40GB", "llama-3.1-8b", batch=32, ctx_len=512)
    assert t32 > 10 * t1


def test_long_context_erodes_the_batching_win():
    short = nk.decode_tokens_per_s("A100-40GB", "llama-3.1-8b", batch=32, ctx_len=512)
    long = nk.decode_tokens_per_s("A100-40GB", "llama-3.1-8b", batch=32, ctx_len=32768)
    assert long < short / 2


def test_ridge_point_says_batch_one_wastes_the_card():
    assert nk.arithmetic_intensity("llama-3.1-8b", batch=1, ctx_len=1024) < nk.ridge_point("T4") / 10


def test_queueing_blows_up_at_saturation():
    assert nk.mm1_wait_s(rps=1, service_time_s=0.5) < 1
    assert nk.mm1_wait_s(rps=1.9, service_time_s=0.5) > 8
    assert math.isinf(nk.mm1_wait_s(rps=2.0, service_time_s=0.5))


def test_little_law():
    assert nk.little_law_concurrency(rps=10, latency_s=2.5) == 25


def test_dtype_lookup_rejects_nonsense():
    with pytest.raises(ValueError):
        nk.dtype_bytes("fp3")


def test_memory_report_flags_a_model_that_cannot_fit():
    assert "do not fit" in nk.memory_report("T4", "llama-3.1-8b")
    assert "do not fit" not in nk.memory_report("A100-80GB", "llama-3.1-8b")
