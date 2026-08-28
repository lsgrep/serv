from servlab.stats import RequestResult, percentile, summarize


def r(start=0.0, ttft=0.2, e2e=1.0, out=21, status=200, error="", scheduled=0.0):
    return RequestResult(start=start, first_token=start + ttft, end=start + e2e,
                         output_tokens=out, status=status, error=error, scheduled=scheduled)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.99) == 5
    assert percentile([], 0.5) is None


def test_tpot_excludes_the_first_token():
    res = r(ttft=0.5, e2e=2.5, out=21)
    # 2.0s of decode over 20 tokens after the first
    assert abs(res.tpot - 0.1) < 1e-9


def test_tpot_is_none_for_single_token_outputs():
    assert r(out=1).tpot is None


def test_failed_requests_are_counted_but_excluded_from_latency():
    results = [r() for _ in range(3)] + [r(status=500, error="boom")]
    s = summarize(results, duration=10.0)
    assert (s.n, s.ok, s.failed) == (4, 3, 1)
    assert s.ttft["p50"] == 0.2


def test_goodput_drops_when_latency_breaches_slo_even_though_throughput_holds():
    fast = [r(start=i * 0.1, ttft=0.3) for i in range(10)]
    slow = [r(start=i * 0.1, ttft=9.0, e2e=12.0) for i in range(10)]
    a = summarize(fast, duration=10.0, slo_ttft=1.0)
    b = summarize(slow, duration=10.0, slo_ttft=1.0)
    assert a.request_throughput == b.request_throughput  # same work done
    assert b.goodput == 0.0 and a.goodput > 0            # none of it useful


def test_client_side_lag_is_surfaced():
    late = [RequestResult(start=5.0, scheduled=0.0, first_token=5.1, end=6.0,
                          output_tokens=10, status=200)]
    s = summarize(late, duration=6.0)
    assert s.max_client_queue_delay == 5.0
    assert "client fell behind" in str(s)
