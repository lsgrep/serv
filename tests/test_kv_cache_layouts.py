"""The cache layout has changed three times; each change broke code silently.

These fakes stand in for the three representations HuggingFace has shipped, so
CI catches a regression without needing torch.
"""

import pytest

from servlab.toy.engine import format_decode_comparison, iter_kv_layers, measured_kv_bytes


class FakeTensor:
    def __init__(self, n, itemsize=2):
        self._n, self._i = n, itemsize

    def numel(self):
        return self._n

    def element_size(self):
        return self._i


class ModernLayer:
    """transformers 5.x: Cache.layers[i].keys / .values"""

    def __init__(self, n):
        self.keys = FakeTensor(n)
        self.values = FakeTensor(n)


class ModernCache:
    def __init__(self, n_layers, n):
        self.layers = [ModernLayer(n) for _ in range(n_layers)]

    def __iter__(self):
        # A real one yields 3-tuples — the shape that caused "too many values
        # to unpack". If the walker ever falls through to iteration, this bites.
        return iter([(FakeTensor(0), FakeTensor(0), None)] * len(self.layers))


class LegacyCache:
    """transformers 4.36-4.5x: parallel key_cache / value_cache lists"""

    def __init__(self, n_layers, n):
        self.key_cache = [FakeTensor(n) for _ in range(n_layers)]
        self.value_cache = [FakeTensor(n) for _ in range(n_layers)]


def test_modern_cache_is_read_through_layers_not_by_iterating():
    cache = ModernCache(n_layers=4, n=100)
    pairs = list(iter_kv_layers(cache))
    assert len(pairs) == 4
    assert all(k.numel() == 100 for k, _ in pairs)
    # 4 layers x (100 + 100) elements x 2 bytes
    assert measured_kv_bytes(cache) == 4 * 200 * 2


def test_the_three_tuple_iteration_that_used_to_crash():
    # Regression: `for k, v in cache` raised ValueError on transformers 5.x.
    cache = ModernCache(n_layers=2, n=8)
    with pytest.raises(ValueError):
        [(k, v) for k, v in cache]          # the old code path
    assert measured_kv_bytes(cache) == 2 * 16 * 2   # the new one is fine


def test_the_four_x_layout_still_works():
    assert measured_kv_bytes(LegacyCache(n_layers=3, n=50)) == 3 * 100 * 2


def test_the_original_tuple_of_tuples_still_works():
    legacy = tuple((FakeTensor(10), FakeTensor(10)) for _ in range(5))
    assert measured_kv_bytes(legacy) == 5 * 20 * 2


def test_entries_with_extra_elements_take_the_first_two():
    padded = tuple((FakeTensor(10), FakeTensor(10), "something else") for _ in range(2))
    assert measured_kv_bytes(padded) == 2 * 20 * 2


def test_an_empty_cache_weighs_nothing():
    assert measured_kv_bytes(ModernCache(n_layers=0, n=0)) == 0
    assert measured_kv_bytes(()) == 0


# -- the comparison summary ------------------------------------------------

def _summary(cached_ms, naive_ms):
    return {
        "prompt_tokens": 512, "n_new": 16,
        "with KV cache": {"median_ms": cached_ms, "first_ms": cached_ms * 8,
                          "last_ms": cached_ms, "context_first": 513, "context_last": 528},
        "no KV cache": {"median_ms": naive_ms, "first_ms": naive_ms,
                        "last_ms": naive_ms, "context_first": 513, "context_last": 528},
        "speedup_median": naive_ms / cached_ms, "speedup_last": naive_ms / cached_ms,
    }


def test_a_healthy_comparison_reports_the_speedup():
    text = format_decode_comparison(_summary(24.5, 490.0))
    assert "20.0x at the median" in text
    assert "!!" not in text


def test_a_weak_result_says_the_measurement_is_wrong_not_the_theory():
    # 0.5x is what the notebook actually printed with a five-token prompt.
    text = format_decode_comparison(_summary(28.4, 12.9))
    assert "!!" in text
    assert "prompt is too short" in text
