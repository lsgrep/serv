from servlab.prometheus import Snapshot, histogram_quantile, parse_text, vllm_row

# Trimmed from a real vLLM /metrics scrape.
SAMPLE = """
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 12.0
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 47.0
vllm:gpu_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 0.987
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 31.0
vllm:prompt_tokens_total{model_name="Qwen/Qwen2.5-3B-Instruct"} 128000.0
vllm:generation_tokens_total{model_name="Qwen/Qwen2.5-3B-Instruct"} 64000.0
vllm:request_success_total{finished_reason="length",model_name="Qwen/Qwen2.5-3B-Instruct"} 90.0
vllm:request_success_total{finished_reason="stop",model_name="Qwen/Qwen2.5-3B-Instruct"} 10.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 10.0
vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="m"} 50.0
vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="m"} 80.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 100.0
vllm:time_to_first_token_seconds_sum{model_name="m"} 61.0
vllm:time_to_first_token_seconds_count{model_name="m"} 100.0
"""


def test_parses_labels_and_values():
    samples = parse_text(SAMPLE)
    running = [s for s in samples if s.name == "vllm:num_requests_running"]
    assert running[0].value == 12.0
    assert running[0].labels["model_name"] == "Qwen/Qwen2.5-3B-Instruct"


def test_comments_and_blank_lines_are_skipped():
    assert all(not s.name.startswith("#") for s in parse_text(SAMPLE))


def test_value_sums_across_label_sets():
    snap = Snapshot.from_text(SAMPLE)
    # 90 finished on length + 10 on stop
    assert snap.value("vllm:request_success_total") == 100.0
    assert snap.value("vllm:request_success_total", finished_reason="stop") == 10.0


def test_histogram_quantile_interpolates_within_a_bucket():
    buckets = [(0.1, 10.0), (0.5, 50.0), (1.0, 80.0), (float("inf"), 100.0)]
    # p50 -> rank 50 lands exactly on the 0.5 boundary
    assert histogram_quantile(buckets, 0.5) == 0.5
    # p25 -> rank 25, between 10 and 50 in the (0.1, 0.5] bucket
    p25 = histogram_quantile(buckets, 0.25)
    assert 0.1 < p25 < 0.5


def test_quantile_in_the_inf_bucket_reports_the_last_finite_edge():
    buckets = [(0.1, 10.0), (1.0, 80.0), (float("inf"), 100.0)]
    # p99 is off the top of the histogram: the honest answer is "at least 1.0",
    # not a made-up interpolation into infinity.
    assert histogram_quantile(buckets, 0.99) == 1.0


def test_empty_histogram_returns_none():
    assert histogram_quantile([], 0.5) is None
    assert histogram_quantile([(1.0, 0.0)], 0.5) is None


def test_vllm_row_is_dashboard_shaped():
    row = vllm_row(SAMPLE, t=3.0)
    assert row["running"] == 12
    assert row["waiting"] == 47
    assert row["kv_usage"] == 98.7  # exported as a 0..1 fraction despite the name
    assert row["preemptions"] == 31
    assert row["ttft_p50"] == 0.5
    assert row["finished"] == 100


def test_accepts_either_kv_metric_name():
    renamed = SAMPLE.replace("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
    assert vllm_row(renamed)["kv_usage"] == 98.7


def test_missing_metrics_do_not_explode():
    row = vllm_row("# nothing here\n")
    assert row["running"] == 0
    assert row["ttft_p50"] is None
