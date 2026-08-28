"""A small Prometheus text-format parser, plus a vLLM-aware view over it.

Why not use a client library: `/metrics` is a few KB of text, the format is
simple, and having the parse in front of you means the metric names stop being
magic. In an interview you want to be able to say what `vllm:num_requests_waiting`
is counting and where it comes from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_LINE = re.compile(
    r"""^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)      # metric name
        (?:\{(?P<labels>.*)\})?                  # optional label set
        \s+(?P<value>[^\s]+)                     # value
        (?:\s+[0-9.eE+-]+)?$                     # optional timestamp
    """,
    re.VERBOSE,
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict
    value: float


def _to_float(raw: str) -> float:
    low = raw.lower()
    if low in ("+inf", "inf"):
        return float("inf")
    if low in ("-inf",):
        return float("-inf")
    if low == "nan":
        return float("nan")
    return float(raw)


def parse_text(text: str):
    """Parse exposition text into samples. HELP/TYPE comments are skipped."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = {}
        if m.group("labels"):
            for k, v in _LABEL.findall(m.group("labels")):
                labels[k] = v.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        try:
            value = _to_float(m.group("value"))
        except ValueError:
            continue
        out.append(Sample(m.group("name"), labels, value))
    return out


def histogram_quantile(buckets, q: float):
    """Prometheus-style quantile over cumulative `(le, count)` pairs.

    Linear interpolation inside the bucket, exactly like `histogram_quantile()`
    in PromQL — which also means the answer is only as good as your bucket
    boundaries. A p99 that sits in the `+Inf` bucket returns the largest finite
    boundary, and you should treat that as "off the top of the histogram",
    not as a measurement.
    """
    buckets = sorted(((float(le), float(c)) for le, c in buckets), key=lambda b: b[0])
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le if prev_le > 0 else None
            if count == prev_count:
                return le
            frac = (rank - prev_count) / (count - prev_count)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_count = le, count
    return buckets[-1][0]


@dataclass
class Snapshot:
    """One scrape, indexed for lookup."""

    samples: list = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> Snapshot:
        return cls(parse_text(text))

    def find(self, name, **labels):
        return [
            s for s in self.samples
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items())
        ]

    def value(self, name, default=None, **labels):
        """Value of a gauge/counter. Sums across label sets (e.g. per-model,
        per-finish-reason) so callers do not have to care about cardinality."""
        found = self.find(name, **labels)
        if not found:
            return default
        if len(found) == 1:
            return found[0].value
        return sum(s.value for s in found)

    def first(self, names, default=None, **labels):
        """First of several candidate names that exists — vLLM has renamed
        metrics across releases, so the labs accept either spelling."""
        for n in names:
            v = self.value(n, default=None, **labels)
            if v is not None:
                return v
        return default

    def histogram(self, name, **labels):
        """Return `(buckets, sum, count)` for a histogram family."""
        buckets = [(s.labels.get("le"), s.value) for s in self.find(name + "_bucket", **labels)]
        buckets = [(le, c) for le, c in buckets if le is not None]
        merged = {}
        for le, c in buckets:
            merged[_to_float(le)] = merged.get(_to_float(le), 0.0) + c
        return (
            sorted(merged.items()),
            self.value(name + "_sum", 0.0, **labels),
            self.value(name + "_count", 0.0, **labels),
        )

    def quantile(self, name, q=0.5, **labels):
        buckets, _, _ = self.histogram(name, **labels)
        return histogram_quantile(buckets, q)

    def mean(self, name, **labels):
        _, total, count = self.histogram(name, **labels)
        return total / count if count else None


# --------------------------------------------------------------------------
# vLLM specifics
# --------------------------------------------------------------------------

# vLLM renamed several metrics between v0 and v1 engines; accept both.
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
SWAPPED = ("vllm:num_requests_swapped",)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
PREFIX_HIT = ("vllm:gpu_prefix_cache_hit_rate",)
PREEMPTIONS = ("vllm:num_preemption_total", "vllm:num_preemptions_total")
PROMPT_TOKENS = ("vllm:prompt_tokens_total",)
GEN_TOKENS = ("vllm:generation_tokens_total",)
TTFT = "vllm:time_to_first_token_seconds"
TPOT = "vllm:time_per_output_token_seconds"
E2E = "vllm:e2e_request_latency_seconds"
QUEUE_TIME = "vllm:request_queue_time_seconds"


def vllm_row(text_or_snapshot, t=None) -> dict:
    """Flatten one scrape into a row for a dataframe / a live plot.

    These seven numbers are the whole dashboard: if you can read them you can
    diagnose a saturated server without any other tooling.
    """
    snap = text_or_snapshot if isinstance(text_or_snapshot, Snapshot) else Snapshot.from_text(text_or_snapshot)
    return {
        "t": t,
        "running": snap.first(RUNNING, 0.0),
        "waiting": snap.first(WAITING, 0.0),
        "swapped": snap.first(SWAPPED, 0.0),
        # vLLM reports this as a 0..1 fraction despite the `_perc` suffix.
        "kv_usage": (snap.first(KV_USAGE, 0.0) or 0.0) * 100.0,
        "preemptions": snap.first(PREEMPTIONS, 0.0),
        "prompt_tokens": snap.first(PROMPT_TOKENS, 0.0),
        "gen_tokens": snap.first(GEN_TOKENS, 0.0),
        "ttft_p50": snap.quantile(TTFT, 0.50),
        "ttft_p99": snap.quantile(TTFT, 0.99),
        "tpot_p50": snap.quantile(TPOT, 0.50),
        "e2e_p99": snap.quantile(E2E, 0.99),
        "queue_p99": snap.quantile(QUEUE_TIME, 0.99),
        "finished": snap.value("vllm:request_success_total", 0.0),
    }
