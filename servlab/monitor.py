"""Background scraper for an OpenAI-compatible server's `/metrics`.

This is the notebook's Grafana. Start it before the load generator, stop it
after, and you have a per-second history of queue depth, KV-cache occupancy and
latency quantiles to plot next to the request-side numbers.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

from .prometheus import Snapshot, vllm_row

# Counters are cumulative; the labs want rates. Derive them here so both the
# live dashboard and the post-hoc plots see the same columns.
_COUNTERS = ("prompt_tokens", "gen_tokens", "finished", "preemptions")


def scrape(url, timeout=2.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local server
        return r.read().decode("utf-8", "replace")


def scrape_row(base_url, t=None) -> dict:
    return vllm_row(scrape(metrics_url(base_url)), t=t if t is not None else time.time())


def metrics_url(base_url) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/metrics") else base + "/metrics"


class MetricsPoller:
    """Poll `/metrics` on a thread and keep the rows.

    Usage:
        poller = MetricsPoller("http://localhost:8000").start()
        ...  run load ...
        df = poller.stop().to_dataframe()
    """

    def __init__(self, base_url="http://localhost:8000", interval=1.0, on_row=None):
        self.url = metrics_url(base_url)
        self.interval = interval
        self.on_row = on_row
        self.rows = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("poller already started")
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="servlab-metrics")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            tick = time.time()
            try:
                row = vllm_row(Snapshot.from_text(scrape(self.url)), t=tick - self._t0)
                self._decorate(row)
                self.rows.append(row)
                if self.on_row:
                    self.on_row(row)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                # A refused connection while the server is still loading weights
                # is normal; record it rather than killing the thread.
                self.errors.append((tick - self._t0, repr(exc)))
            # Keep a steady cadence even if a scrape was slow.
            self._stop.wait(max(0.0, self.interval - (time.time() - tick)))

    def _decorate(self, row):
        prev = self.rows[-1] if self.rows else None
        dt = (row["t"] - prev["t"]) if prev else None
        for c in _COUNTERS:
            rate_key = c + "_per_s"
            if prev and dt and dt > 0 and prev.get(c) is not None and row.get(c) is not None:
                # Counters reset if the server restarts mid-lab; clamp to 0.
                row[rate_key] = max(0.0, (row[c] - prev[c]) / dt)
            else:
                row[rate_key] = 0.0

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 3)
        self._thread = None
        return self

    @property
    def latest(self):
        return self.rows[-1] if self.rows else None

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
