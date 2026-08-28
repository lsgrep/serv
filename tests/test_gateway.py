import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from servlab import gateway as gw


class _Threaded(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve(handler):
    srv = _Threaded(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _openai_handler(status=200, text="hello"):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({
                "model": "fake-1",
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    return H


def test_gateway_calls_an_openai_compatible_endpoint_and_prices_it():
    srv = _serve(_openai_handler())
    try:
        g = gw.Gateway().register(
            "cheap", gw.OpenAICompatible(f"http://127.0.0.1:{srv.server_address[1]}"),
            model="fake-1", price_key="gemini-3.5-flash-lite")
        out = g.complete("hi")
    finally:
        srv.shutdown()

    assert out.text == "hello"
    assert out.input_tokens == 100 and out.output_tokens == 20
    # 100 in @ $0.30/M + 20 out @ $2.50/M
    assert out.cost_usd == pytest.approx(100 * 0.30 / 1e6 + 20 * 2.50 / 1e6)
    assert g.spend() == pytest.approx(out.cost_usd)


def test_fallback_chain_survives_a_dead_primary():
    srv = _serve(_openai_handler(text="from the backup"))
    try:
        g = (gw.Gateway()
             .register("primary", gw.OpenAICompatible("http://127.0.0.1:9"), model="x")
             .register("backup", gw.OpenAICompatible(f"http://127.0.0.1:{srv.server_address[1]}"),
                       model="fake-1"))
        out = g.complete("hi", alias="primary", fallbacks=["backup"])
    finally:
        srv.shutdown()
    assert out.text == "from the backup"
    assert g.log[-1]["alias"] == "backup"


def test_every_route_failing_raises_with_all_the_reasons():
    g = gw.Gateway().register("a", gw.OpenAICompatible("http://127.0.0.1:9"), model="x")
    with pytest.raises(gw.ProviderError) as exc:
        g.complete("hi", alias="a", fallbacks=["nope"])
    assert "not registered" in str(exc.value)


def test_switching_cost_is_dominated_by_people_and_finetunes_not_embeddings():
    # The folk wisdom says embeddings; the arithmetic says otherwise at current
    # prices, and it is better to check than to repeat it.
    r = gw.switching_cost(corpus_tokens=1e9, engineer_days=10, fine_tune_reruns=2,
                          fine_tune_cost_each=8000)
    assert r["terms"]["re-embed corpus"] < r["terms"]["engineering"]
    assert r["dominant_term"] in ("re-run fine-tunes", "engineering")


def test_no_finetunes_means_a_cheap_exit():
    r = gw.switching_cost(corpus_tokens=5e6, engineer_days=5)
    assert r["total_usd"] < 10_000
    assert "consciously" in gw.format_switching_cost(r) or "knowingly" in gw.format_switching_cost(r)
