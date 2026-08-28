import pytest

np = pytest.importorskip("numpy")

from servlab import attention as at  # noqa: E402
from servlab import napkin as nk  # noqa: E402


@pytest.fixture
def weights():
    return at.AttentionWeights(d_model=64, n_heads=8, n_kv_heads=2, seed=0)


@pytest.fixture
def x():
    return np.random.default_rng(0).normal(size=(12, 64))


# -- softmax ---------------------------------------------------------------


def test_naive_softmax_overflows_where_the_stable_one_does_not():
    big = np.array([1000.0, 1001.0, 999.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = at.naive_softmax(big)
    assert not np.all(np.isfinite(naive))
    stable = at.softmax(big)
    assert np.all(np.isfinite(stable))
    assert stable.sum() == pytest.approx(1.0)


def test_online_softmax_matches_the_one_pass_version():
    # The claim behind flash attention: you can stream it and get the same
    # answer, holding a running max and sum instead of the whole row.
    scores = np.random.default_rng(1).normal(size=(3, 17)) * 5
    for block in (1, 4, 8, 32):
        assert np.allclose(at.online_softmax(scores, block_size=block),
                           at.softmax(scores))


# -- masking ---------------------------------------------------------------


def test_causal_mask_is_lower_triangular():
    m = at.causal_mask(4)
    assert m.tolist() == [[True, False, False, False],
                          [True, True, False, False],
                          [True, True, True, False],
                          [True, True, True, True]]


def test_a_single_decode_query_may_attend_to_the_whole_past():
    # One query at position 7 against 8 keys: a row, not a triangle. This shape
    # is why decode needs no mask at all once the cache holds only the past.
    m = at.causal_mask(1, 8)
    assert m.shape == (1, 8)
    assert m.all()


def test_masked_positions_get_no_attention_mass():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(5, 8)) for _ in range(3))
    _, w = at.attention(q, k, v, mask=at.causal_mask(5), return_weights=True)
    assert w[0, 1:].max() < 1e-6      # first token sees only itself
    assert np.allclose(w.sum(axis=-1), 1.0)


# -- flash -----------------------------------------------------------------


def test_flash_attention_matches_naive_exactly_enough():
    rng = np.random.default_rng(2)
    q, k, v = (rng.normal(size=(4, 16, 8)) for _ in range(3))
    assert np.allclose(at.flash_attention(q, k, v, block_size=4),
                       at.attention(q, k, v))


def test_flash_attention_matches_under_a_causal_mask():
    rng = np.random.default_rng(3)
    q, k, v = (rng.normal(size=(12, 8)) for _ in range(3))
    mask = at.causal_mask(12)
    assert np.allclose(at.flash_attention(q, k, v, mask=mask, block_size=5),
                       at.attention(q, k, v, mask=mask))


def test_flash_saves_memory_and_not_arithmetic():
    # The distinction people get wrong: same FLOPs, hugely different footprint.
    naive = at.score_matrix_bytes(8192, n_heads=32)
    tiled = at.flash_workspace_bytes(128, n_heads=32)
    assert naive / tiled > 1000
    assert at.attention_flops(8192, 4096) == at.attention_flops(8192, 4096)


def test_score_matrix_is_quadratic_in_context():
    assert (at.score_matrix_bytes(4096, 32) / at.score_matrix_bytes(2048, 32)) == 4.0


# -- heads -----------------------------------------------------------------


def test_gqa_is_exactly_a_repeat_at_inference_time():
    kv = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    repeated = at.repeat_kv(kv, 8)
    assert repeated.shape == (8, 3, 4)
    # Each KV head is shared by four query heads — nothing approximated.
    assert np.array_equal(repeated[0], kv[0])
    assert np.array_equal(repeated[3], kv[0])
    assert np.array_equal(repeated[4], kv[1])


def test_repeat_kv_is_a_no_op_for_multi_head_attention():
    kv = np.random.default_rng(0).normal(size=(8, 3, 4))
    assert np.array_equal(at.repeat_kv(kv, 8), kv)


def test_gqa_shrinks_the_cache_by_exactly_the_group_ratio():
    mha = at.AttentionWeights(d_model=64, n_heads=8, n_kv_heads=8)
    gqa = at.AttentionWeights(d_model=64, n_heads=8, n_kv_heads=2)
    mqa = at.AttentionWeights(d_model=64, n_heads=8, n_kv_heads=1)
    per_token = lambda w: at.KVCache(w.n_kv_heads, w.d_head).bytes_per_token()  # noqa: E731
    assert per_token(mha) / per_token(gqa) == 4.0
    assert per_token(mha) / per_token(mqa) == 8.0


def test_incompatible_head_counts_are_rejected():
    with pytest.raises(ValueError):
        at.AttentionWeights(d_model=64, n_heads=8, n_kv_heads=3)
    with pytest.raises(ValueError):
        at.repeat_kv(np.zeros((3, 2, 4)), 8)


# -- the KV cache ----------------------------------------------------------


def test_cached_decoding_equals_full_recomputation(weights, x):
    """The correctness proof of the KV cache, and the reason it is sound.

    Because the mask is causal, keys and values for earlier tokens never change,
    so reusing them cannot alter the answer. If this test ever fails, caching is
    not an optimisation — it is a bug.
    """
    full = at.mha_forward(x, weights)

    out, cache = at.prefill(x[:8], weights)
    last = out[-1]
    for t in range(8, 12):
        step, cache = at.decode_step(x[t], weights, cache)
        last = step[0]

    assert cache.length == 12
    assert np.allclose(full[-1], last)


def test_cache_bytes_match_the_formula_used_everywhere_else(weights):
    cache = at.KVCache(weights.n_kv_heads, weights.d_head, dtype_bytes=2)
    cache.append(*[np.zeros((weights.n_kv_heads, 100, weights.d_head))] * 2)
    # 2 (K and V) x kv_heads x head_dim x bytes, per token.
    assert cache.bytes_per_token() == 2 * weights.n_kv_heads * weights.d_head * 2
    assert cache.nbytes == cache.bytes_per_token() * 100


def test_the_toy_cache_agrees_with_the_napkin_formula():
    spec = nk.MODELS["llama-3.3-70b"]
    cache = at.KVCache(spec.n_kv_heads, spec.head_dim, dtype_bytes=2)
    # One layer here; the napkin formula counts all of them.
    assert cache.bytes_per_token() * spec.n_layers == nk.kv_bytes_per_token(spec)


def test_cache_grows_by_one_token_per_decode_step(weights, x):
    _, cache = at.prefill(x[:4], weights)
    assert cache.length == 4
    for expected in (5, 6, 7):
        _, cache = at.decode_step(x[expected - 1], weights, cache)
        assert cache.length == expected


# -- cost ------------------------------------------------------------------


def test_prefill_attention_is_quadratic_and_decode_is_linear():
    assert at.attention_flops(2048, 4096) / at.attention_flops(1024, 4096) == 4.0
    d1 = at.attention_flops(1, 4096, decode=True, ctx_len=1024)
    d2 = at.attention_flops(1, 4096, decode=True, ctx_len=2048)
    assert d2 / d1 == 2.0


def test_attention_is_a_rounding_error_at_short_context_and_dominant_at_long():
    short = at.attention_share(1024, 4096, 32, 8)
    long = at.attention_share(65536, 4096, 32, 8)
    assert short < 0.06
    assert long > 0.7


def test_crossover_is_where_attention_reaches_half_of_a_layer():
    n = at.quadratic_crossover(4096, 32, 8)
    assert 8192 < n < 65536
    assert at.attention_share(n, 4096, 32, 8) >= 0.5
    assert at.attention_share(n - 64, 4096, 32, 8) < 0.5


# -- positions -------------------------------------------------------------


def test_rope_scores_depend_on_distance_not_absolute_position():
    r = at.relative_score_shift(d_head=8, gap=3)
    assert r["spread"] < 1e-9        # same gap, same score, anywhere


def test_a_cached_key_is_only_valid_at_its_own_offset():
    # Why prefix caching is a *prefix* cache: reuse a key at the wrong offset
    # and the geometry — and therefore the score — is wrong.
    r = at.relative_score_shift(d_head=8, gap=3)
    assert not np.isclose(r["wrong_offset_score"], r["same_gap_scores"][0])


def test_rope_rejects_odd_head_dims():
    with pytest.raises(ValueError):
        at.rope(np.zeros((4, 7)))


# -- looking at weights ----------------------------------------------------


def test_entropy_separates_a_peaked_head_from_an_averaging_one():
    peaked = np.array([[0.98, 0.01, 0.01]])
    flat = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert at.attention_entropy(peaked)[0] < at.attention_entropy(flat)[0]
    assert at.attention_entropy(flat)[0] == pytest.approx(np.log(3))


def test_sink_share_measures_mass_on_the_first_token():
    assert at.sink_share(np.array([[0.9, 0.05, 0.05], [0.7, 0.2, 0.1]])) == pytest.approx(0.8)
