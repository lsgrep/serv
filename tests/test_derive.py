import pytest

from servlab import napkin as nk
from servlab.derive import Derivation, Worksheet


def test_a_derivation_renders_givens_working_and_a_result():
    d = nk.derive_kv_per_token(80, 8, 128, "fp16")
    text = str(d)
    assert "GIVEN" in text and "WORKING" in text and "KV PER TOKEN" in text
    # The substitution itself must be visible — that is the entire point.
    assert "2 x 8 x 128 x 2" in text
    assert "4,096 x 80" in text


def test_the_derived_value_matches_the_one_line_function():
    for layers, kv, dim, dtype in ((80, 8, 128, "fp16"), (32, 32, 64, "fp8"), (12, 12, 64, "fp16")):
        spec = nk.ModelSpec("x", layers, kv, kv, dim, kv * dim, 4 * kv * dim, 32000, 1e9)
        assert nk.derive_kv_per_token(layers, kv, dim, dtype).value == \
            nk.kv_bytes_per_token(spec, dtype)


def test_derivations_take_raw_numbers_not_preset_keys():
    # The point of the exercise: a model nobody has a preset for.
    d = nk.derive_kv_per_token(n_layers=57, n_kv_heads=6, head_dim=112, dtype="fp8")
    assert d.value == 2 * 57 * 6 * 112 * 1


def test_four_bit_weights_are_half_a_byte_not_four():
    d = nk.derive_weight_bytes(70.6e9, 4)
    assert d.value == pytest.approx(70.6e9 * 0.5)
    assert nk.derive_weight_bytes(70.6e9, 16).value == pytest.approx(70.6e9 * 2)
    assert nk.derive_weight_bytes(70.6e9, "int4").value == pytest.approx(70.6e9 * 0.5)


def test_capacity_derivation_agrees_with_the_closed_form():
    kv = nk.derive_kv_per_token(80, 8, 128).value
    d = nk.derive_capacity(vram_gb=80, params=70.6e9, kv_per_token=kv, ctx=8192,
                           weight_bits=8, util=0.90, activation_gb=1.0)
    expected = nk.max_concurrent_sequences(
        nk.GPUS["H100-80GB"], nk.MODELS["llama-3.3-70b"], 8192,
        weight_dtype="fp8", kv_dtype="fp16", util=0.90, activation_gb=1.0)
    assert d.value == pytest.approx(expected, rel=0.02)


def test_capacity_warns_rather_than_returning_a_meaningless_number():
    kv = nk.derive_kv_per_token(80, 8, 128).value
    d = nk.derive_capacity(vram_gb=16, params=70.6e9, kv_per_token=kv, ctx=8192)
    assert d.value == 0
    assert any("do not fit" in w for w in d.warnings)


def test_capacity_says_context_and_concurrency_are_the_same_knob():
    kv = nk.derive_kv_per_token(80, 8, 128).value
    d = nk.derive_capacity(vram_gb=80, params=70.6e9, kv_per_token=kv, ctx=8192, weight_bits=8)
    assert any("same knob" in c for c in d.checks)


def test_decode_derivation_shows_the_weight_and_kv_terms_separately():
    kv = nk.derive_kv_per_token(80, 8, 128).value
    d = nk.derive_decode_speed(70.6e9, 3350, batch=32, ctx=8192, kv_per_token=kv,
                               weight_bits=8)
    labels = [s.label for s in d.steps]
    assert "weight bytes read every step" in labels
    assert "KV bytes read every step" in labels
    # And it must say which one currently dominates, since that is the batching answer.
    assert any("dominate" in c for c in d.checks)


def test_decode_derivation_matches_the_closed_form():
    kv = nk.kv_bytes_per_token("llama-3.3-70b", "fp8")
    d = nk.derive_decode_speed(70.6e9, 3350, batch=16, ctx=4096, kv_per_token=kv,
                               weight_bits=8, efficiency=0.7)
    closed = nk.decode_tokens_per_s("H100-80GB", "llama-3.3-70b", batch=16, ctx_len=4096,
                                    weight_dtype="fp8", kv_dtype="fp8")
    assert d.value == pytest.approx(closed, rel=0.01)


def test_prefill_derivation_matches_the_closed_form():
    d = nk.derive_prefill_time(70.6e9, 4096, tflops=989, mfu=0.4, n_gpus=1)
    assert d.value == pytest.approx(nk.prefill_time_s("H100-80GB", "llama-3.3-70b", 4096, mfu=0.4),
                                    rel=0.01)


def test_prefill_derivation_judges_the_answer_against_a_user():
    fast = nk.derive_prefill_time(1e9, 256, tflops=989)
    slow = nk.derive_prefill_time(400e9, 32768, tflops=65)
    assert any("instant" in c for c in fast.checks)
    assert any("notice" in c for c in slow.checks)


def test_cost_derivation_matches_the_closed_form():
    d = nk.derive_cost_per_million(3.50, 2000)
    assert d.value == pytest.approx(nk.cost_per_million_tokens(
        nk.GPUSpec("x", 80, 3350, 989, (9, 0), 3.50), 2000))


def test_worksheet_chains_the_whole_answer():
    ws = nk.worksheet(name="mystery 34B", n_layers=60, n_kv_heads=8, head_dim=128,
                      params=34e9, vram_gb=80, mem_bw_gb_s=3350, tflops=989,
                      ctx=8192, batch=32, weight_bits=8, usd_per_hour=3.50)
    assert isinstance(ws, Worksheet)
    titles = [p.title for p in ws.parts]
    assert any("KV cache" in t for t in titles)
    assert any("concurrency" in t for t in titles)
    assert any("decode" in t for t in titles)
    assert any("first token" in t for t in titles)
    assert any("cost" in t for t in titles)
    text = str(ws)
    assert "mystery 34B" in text
    assert text.count("GIVEN") == len(ws.parts)


def test_worksheet_omits_the_parts_it_was_not_given_inputs_for():
    ws = nk.worksheet(name="partial", n_layers=32, n_kv_heads=8, head_dim=128, params=8e9,
                      vram_gb=24, mem_bw_gb_s=300, ctx=4096)
    titles = [p.title for p in ws.parts]
    assert not any("first token" in t for t in titles)   # no tflops given
    assert not any("cost" in t for t in titles)          # no price given


# -- reading a config ------------------------------------------------------

LLAMA_31_8B = {
    "_name_or_path": "meta-llama/Llama-3.1-8B",
    "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
    "hidden_size": 4096, "intermediate_size": 14336, "vocab_size": 128256,
    "torch_dtype": "bfloat16",
}


def test_from_config_uses_kv_heads_and_says_so():
    d = nk.from_config(LLAMA_31_8B)
    assert d.value == 128 * 1024
    assert any("NOT used in this formula" in g.source for g in d.givens)
    assert any("overstated the cache by 4x" in c for c in d.checks)


def test_from_config_prefers_a_stated_head_dim_over_the_ratio():
    cfg = {**LLAMA_31_8B, "head_dim": 64}      # deliberately not hidden/heads
    d = nk.from_config(cfg)
    assert d.value == 2 * 32 * 8 * 64 * 2
    assert any("stated explicitly" in c for c in d.checks)


def test_from_config_flags_bfloat16_as_the_turing_trap():
    assert any("--dtype half" in c for c in nk.from_config(LLAMA_31_8B).checks)
    no_bf16 = nk.from_config({**LLAMA_31_8B, "torch_dtype": "float16"})
    assert not any("--dtype half" in c for c in no_bf16.checks)


def test_from_config_notices_when_there_is_no_gqa():
    d = nk.from_config({**LLAMA_31_8B, "num_key_value_heads": 32})
    assert any("no GQA" in c for c in d.checks)


def test_from_config_accepts_an_object_not_just_a_dict():
    class Cfg:
        num_hidden_layers = 32
        num_attention_heads = 32
        num_key_value_heads = 8
        hidden_size = 4096

    assert nk.from_config(Cfg()).value == 128 * 1024


def test_config_guide_names_the_common_error():
    guide = nk.config_guide()
    assert "num_key_value_heads" in guide
    assert "4x" in guide


# -- the rendering itself --------------------------------------------------

def test_byte_values_are_shown_in_human_units_too():
    d = nk.derive_weight_bytes(70.6e9, 16)
    assert "GiB" in str(d)


def test_a_derivation_is_usable_as_a_number():
    d = nk.derive_kv_per_token(80, 8, 128)
    assert float(d) == 327_680
    assert d.value * 2 == 655_360


def test_an_empty_derivation_renders_without_crashing():
    assert "EMPTY" in str(Derivation("empty")).upper()
