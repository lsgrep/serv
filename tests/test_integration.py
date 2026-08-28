"""End-to-end checks against fake servers.

The network code is the part most likely to break silently — a stream format
change turns TTFT into None rather than into an error. These tests stand up a
real socket and exercise the whole path.
"""

import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest

from servlab.monitor import MetricsPoller, scrape_row
from servlab.stats import summarize

METRICS = b"""vllm:num_requests_running{model_name="m"} 12.0
vllm:num_requests_waiting{model_name="m"} 47.0
vllm:gpu_cache_usage_perc{model_name="m"} 0.99
vllm:generation_tokens_total{model_name="m"} %d.0
vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="m"} 5.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 10.0
vllm:time_to_first_token_seconds_count{model_name="m"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="m"} 4.0
"""


class _Threaded(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture
def metrics_server():
    counter = {"n": 1000}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            counter["n"] += 250  # a steadily advancing counter, to test rate derivation
            body = METRICS % counter["n"]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = _Threaded(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def openai_server():
    """Minimal OpenAI-compatible SSE endpoint with a deliberate prefill delay."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(payload):
                data = ("data: " + payload + "\n\n").encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            time.sleep(0.05)  # "prefill" — should land in TTFT, not TPOT
            for _ in range(req["max_tokens"]):
                chunk(json.dumps({"choices": [{"delta": {"content": "tok "}}]}))
                time.sleep(0.002)
            chunk("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *a):
            pass

    srv = _Threaded(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_scrape_row_reads_a_live_endpoint(metrics_server):
    row = scrape_row(metrics_server)
    assert row["running"] == 12
    assert row["waiting"] == 47
    assert row["kv_usage"] == 99.0
    assert row["ttft_p50"] == 0.5


def test_poller_collects_rows_and_derives_rates(metrics_server):
    poller = MetricsPoller(metrics_server, interval=0.1).start()
    time.sleep(0.75)
    poller.stop()

    assert len(poller.rows) >= 4
    assert not poller.errors
    # 250 tokens per 0.1s scrape -> ~2500/s. Wide bounds: CI machines are slow.
    assert 500 < poller.rows[-1]["gen_tokens_per_s"] < 20_000


def test_poller_survives_a_dead_server():
    poller = MetricsPoller("http://127.0.0.1:9", interval=0.05).start()
    time.sleep(0.3)
    poller.stop()
    # A refused connection while the server loads weights is normal, not fatal.
    assert poller.errors and not poller.rows


def test_closed_loop_parses_the_stream(openai_server):
    pytest.importorskip("httpx")
    from servlab.loadgen import run_load

    results = run_load(openai_server, "fake", concurrency=4, duration=1.0,
                       prompt_tokens=64, max_tokens=8)
    assert results and all(r.ok for r in results)
    r = results[0]
    assert r.output_tokens == 8
    assert r.ttft >= 0.05          # the simulated prefill shows up here
    assert r.tpot < r.ttft         # ...and not in the per-token time


def test_open_loop_launches_on_schedule_regardless_of_completion(openai_server):
    pytest.importorskip("httpx")
    from servlab.loadgen import run_load

    results = run_load(openai_server, "fake", rps=10.0, duration=1.0,
                       prompt_tokens=32, max_tokens=4)
    # Poisson arrivals at 10/s over 1s: loose bounds, but far from zero and
    # not gated on how fast the server drains them.
    assert 3 <= len(results) <= 25
    assert summarize(results).failed == 0


def test_error_responses_are_recorded_not_raised():
    pytest.importorskip("httpx")
    from servlab.loadgen import run_load

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = b'{"error": "model not found"}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = _Threaded(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        results = run_load(f"http://127.0.0.1:{srv.server_address[1]}", "fake",
                           concurrency=1, n_requests=2, duration=None, max_tokens=4)
    finally:
        srv.shutdown()

    s = summarize(results)
    assert s.failed == len(results)     # a failed request is data, not a crash
    assert results[0].status == 404
