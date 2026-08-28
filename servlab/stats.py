"""Latency statistics, kept torch-free and network-free so they can be tested.

The definitions matter more than the code:

* **TTFT** — time to first token. What a chat user experiences as "did it hang?"
* **TPOT** — time per output token *after* the first. Perceived reading speed.
* **E2E** — the whole request. Dominated by output length, so it is the least
  useful of the three for diagnosis and the one dashboards over-report.
* **Goodput** — throughput counting only requests that met an SLO. A server can
  hold throughput flat while goodput collapses; that gap is the death spiral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    """One request's timeline, from the client's point of view."""

    start: float
    end: float = 0.0
    first_token: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    status: int = 0
    error: str = ""
    text: str = ""
    scheduled: float = None  # when the generator *intended* to send it

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error

    @property
    def ttft(self):
        return (self.first_token - self.start) if self.first_token else None

    @property
    def e2e(self):
        return (self.end - self.start) if self.end else None

    @property
    def tpot(self):
        """Excludes the first token — that one is prefill, not decode."""
        if not (self.first_token and self.end and self.output_tokens > 1):
            return None
        return (self.end - self.first_token) / (self.output_tokens - 1)

    @property
    def queue_delay(self):
        """How late the client was in *sending*. Non-zero means your load
        generator is the bottleneck, not the server — check this before you
        believe any latency number from an open-loop run."""
        return (self.start - self.scheduled) if self.scheduled is not None else 0.0


def percentile(values, q):
    """Linear-interpolation percentile; `q` in 0..1. Returns None if empty."""
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


@dataclass
class LoadSummary:
    n: int = 0
    ok: int = 0
    failed: int = 0
    duration: float = 0.0
    ttft: dict = field(default_factory=dict)
    tpot: dict = field(default_factory=dict)
    e2e: dict = field(default_factory=dict)
    output_tokens: int = 0
    prompt_tokens: int = 0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    goodput: float = 0.0
    slo: dict = field(default_factory=dict)
    max_client_queue_delay: float = 0.0

    def __str__(self):
        def q(d):
            return "  ".join(f"{k}={d[k]*1000:,.0f}ms" for k in ("p50", "p90", "p99") if d.get(k) is not None)

        lines = [
            f"{self.ok}/{self.n} ok  ({self.failed} failed)  in {self.duration:.1f}s",
            f"  TTFT   {q(self.ttft)}",
            f"  TPOT   {q(self.tpot)}",
            f"  E2E    {q(self.e2e)}",
            f"  throughput  {self.request_throughput:.2f} req/s   {self.output_throughput:,.0f} out-tok/s",
        ]
        if self.slo:
            lines.append(
                f"  goodput     {self.goodput:.2f} req/s  "
                f"(SLO ttft<{self.slo.get('ttft', float('inf')):.1f}s, tpot<{self.slo.get('tpot', float('inf'))*1000:.0f}ms)"
            )
        if self.max_client_queue_delay > 0.5:
            lines.append(
                f"  !! client fell behind by up to {self.max_client_queue_delay:.1f}s — "
                "the generator could not keep up, so treat these latencies as a floor"
            )
        return "\n".join(lines)


def _quantiles(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    return {
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p99": percentile(vals, 0.99),
        "max": max(vals),
    }


def summarize(results, duration=None, slo_ttft=None, slo_tpot=None) -> LoadSummary:
    """Roll a list of `RequestResult` into the numbers you would report.

    `duration` defaults to wall-clock span of the run, which is what you want
    for an open-loop test; pass it explicitly if you are timing a fixed window.
    """
    results = list(results)
    ok = [r for r in results if r.ok]
    if duration is None and results:
        duration = max((r.end or r.start) for r in results) - min(r.start for r in results)
    duration = duration or 0.0

    out_tokens = sum(r.output_tokens for r in ok)
    s = LoadSummary(
        n=len(results),
        ok=len(ok),
        failed=len(results) - len(ok),
        duration=duration,
        ttft=_quantiles([r.ttft for r in ok]),
        tpot=_quantiles([r.tpot for r in ok]),
        e2e=_quantiles([r.e2e for r in ok]),
        output_tokens=out_tokens,
        prompt_tokens=sum(r.prompt_tokens for r in ok),
        max_client_queue_delay=max((r.queue_delay for r in results), default=0.0),
    )
    if duration > 0:
        s.request_throughput = len(ok) / duration
        s.output_throughput = out_tokens / duration
    if slo_ttft is not None or slo_tpot is not None:
        s.slo = {
            "ttft": slo_ttft if slo_ttft is not None else float("inf"),
            "tpot": slo_tpot if slo_tpot is not None else float("inf"),
        }
        good = [
            r for r in ok
            if (r.ttft is not None and r.ttft <= s.slo["ttft"])
            and (r.tpot is None or r.tpot <= s.slo["tpot"])
        ]
        s.goodput = len(good) / duration if duration > 0 else 0.0
    return s
