"""Token economics: API pricing, routing, and the managed-vs-self-host call.

The prices below are **a snapshot you maintain**, not something this library
knows. Vendor pages move monthly and intro pricing expires; every function here
is only as good as `VERIFIED_ON`, and `staleness()` will say so out loud.

The point of putting them in code rather than a spreadsheet is that the
interesting questions are all comparisons under assumptions — blended cost after
routing, break-even against self-hosting, what a 5x volume increase does to the
answer — and those are two lines of arithmetic each once the table exists.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

# Snapshot date for every price in this module. Re-verify before quoting.
VERIFIED_ON = _dt.date(2026, 8, 17)
STALE_AFTER_DAYS = 30

# Cross-provider constants that have held across vendors and generations.
BATCH_DISCOUNT = 0.50          # async/batch tiers run about half price
CACHE_READ_FRACTION = 0.10     # cached input reads bill at ~10% of input
TOKENS_PER_WORD = 1 / 0.75     # 1 token ~ 0.75 words ~ 4 characters
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens.

    `long_ctx_threshold` captures the context cliff some models have: past a
    context length, both rates step up. Missing it is how a quote comes in at
    half the real number for long-document work.
    """

    name: str
    input_per_m: float
    output_per_m: float
    provider: str = ""
    tier: str = ""                    # frontier | mid | cheap
    long_ctx_threshold: int = 0
    long_ctx_input_per_m: float = 0.0
    long_ctx_output_per_m: float = 0.0
    notes: str = ""

    def rates(self, context_tokens=0):
        if self.long_ctx_threshold and context_tokens > self.long_ctx_threshold:
            return self.long_ctx_input_per_m, self.long_ctx_output_per_m
        return self.input_per_m, self.output_per_m


# --------------------------------------------------------------------------
# The table. Edit this; do not trust it.
# --------------------------------------------------------------------------

MODELS = {
    "gemini-3.1-pro": ModelPrice(
        "gemini-3.1-pro", 2.00, 12.00, "google", "frontier",
        long_ctx_threshold=200_000, long_ctx_input_per_m=4.00, long_ctx_output_per_m=18.00,
        notes="context cliff at 200K: rates roughly double"),
    "gemini-3.7-flash": ModelPrice("gemini-3.7-flash", 1.50, 7.50, "google", "mid",
                                   notes="intro pricing; expected to roughly double 2027-01-01"),
    "gemini-3.6-flash": ModelPrice("gemini-3.6-flash", 1.50, 7.50, "google", "mid",
                                   notes="intro pricing; expected to roughly double 2027-01-01"),
    "gemini-3.5-flash": ModelPrice("gemini-3.5-flash", 1.50, 9.00, "google", "mid"),
    "gemini-3.5-flash-lite": ModelPrice("gemini-3.5-flash-lite", 0.30, 2.50, "google", "cheap"),
    "gemini-2.5-flash-lite": ModelPrice("gemini-2.5-flash-lite", 0.10, 0.40, "google", "cheap",
                                        notes="the floor"),
    "gpt-5.6-sol": ModelPrice("gpt-5.6-sol", 5.00, 30.00, "openai", "frontier"),
    "gpt-5.6-terra": ModelPrice("gpt-5.6-terra", 2.00, 12.00, "openai", "mid"),
    "gpt-5.6-luna": ModelPrice("gpt-5.6-luna", 0.20, 1.20, "openai", "cheap",
                               notes="cut ~80% in July 2026"),
    "claude-opus-5": ModelPrice("claude-opus-5", 5.00, 25.00, "anthropic", "frontier"),
    "claude-sonnet-5": ModelPrice("claude-sonnet-5", 2.00, 10.00, "anthropic", "mid",
                                  notes="2/10 now permanent"),
    "claude-haiku-4.5": ModelPrice("claude-haiku-4.5", 1.00, 5.00, "anthropic", "cheap"),
}

# Priced per query rather than per token, and easy to forget in a quote.
GROUNDING_SEARCH_PER_1K_QUERIES = 14.00
GROUNDING_FREE_QUERIES_PER_DAY = 5_000


def price(model) -> ModelPrice:
    return model if isinstance(model, ModelPrice) else MODELS[str(model).lower()]


def staleness(today=None) -> str:
    """Say how old the table is. Call it at the top of any costing notebook."""
    today = today or _dt.date.today()
    age = (today - VERIFIED_ON).days
    line = f"pricing snapshot from {VERIFIED_ON.isoformat()} — {age} days old"
    if age > STALE_AFTER_DAYS:
        return (f"!! {line}. Vendor pages move monthly and intro rates expire; "
                "re-verify before quoting any of these numbers to a customer.")
    return f"-- {line}."


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def tokens_from_words(words) -> float:
    return words * TOKENS_PER_WORD


def tokens_from_chars(chars) -> float:
    return chars / CHARS_PER_TOKEN


def call_cost(model, input_tokens, output_tokens, cached_input_tokens=0,
              thinking_tokens=0, batch=False, context_tokens=None) -> float:
    """USD for one call.

    Two traps this function exists to make unmissable:

    * **Thinking tokens bill as output.** A reasoning model's hidden trace is
      charged at the output rate — 5-6x the input rate — even though nobody
      reads it. It is the usual explanation for a bill that doubled without
      traffic changing.
    * **Cached input is not free**, just cheap: about 10% of the input rate.
    """
    p = price(model)
    ctx = context_tokens if context_tokens is not None else input_tokens + cached_input_tokens
    in_rate, out_rate = p.rates(ctx)
    fresh_in = max(0.0, input_tokens)
    cost = (fresh_in * in_rate
            + cached_input_tokens * in_rate * CACHE_READ_FRACTION
            + (output_tokens + thinking_tokens) * out_rate) / 1e6
    return cost * (BATCH_DISCOUNT if batch else 1.0)


def daily_cost(model, input_tokens_per_day, output_tokens_per_day, **kw) -> float:
    return call_cost(model, input_tokens_per_day, output_tokens_per_day, **kw)


def monthly_cost(model, requests_per_month, input_tokens, output_tokens, days=30, **kw) -> float:
    return call_cost(model, input_tokens, output_tokens, **kw) * requests_per_month


def compare(models, input_tokens, output_tokens, requests_per_month=1, **kw) -> list:
    """Same workload across models, cheapest first. The spread *is* the
    architecture conversation."""
    rows = []
    for m in models:
        p = price(m)
        per_call = call_cost(p, input_tokens, output_tokens, **kw)
        rows.append({"model": p.name, "provider": p.provider, "tier": p.tier,
                     "per_call": per_call, "per_month": per_call * requests_per_month})
    rows.sort(key=lambda r: r["per_call"])
    cheapest = rows[0]["per_call"] or 1e-12
    for r in rows:
        r["vs_cheapest"] = r["per_call"] / cheapest
    return rows


def format_compare(rows) -> str:
    out = [f"{'model':<24}{'per call':>13}{'per month':>15}{'vs cheapest':>13}"]
    for r in rows:
        out.append(f"{r['model']:<24}{'$' + format(r['per_call'], '.5f'):>13}"
                   f"{'$' + format(r['per_month'], ',.0f'):>15}{r['vs_cheapest']:>12.1f}x")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


@dataclass
class RoutingPlan:
    cheap_model: str
    strong_model: str
    escalation_rate: float = 0.15
    double_charge: bool = True   # an escalated request paid for the cheap try too

    def cost_per_call(self, input_tokens, output_tokens, **kw) -> float:
        cheap = call_cost(self.cheap_model, input_tokens, output_tokens, **kw)
        strong = call_cost(self.strong_model, input_tokens, output_tokens, **kw)
        e = self.escalation_rate
        if self.double_charge:
            # The realistic accounting: you pay the cheap model, judge the
            # answer, then pay the strong one. Ignoring the first call is the
            # most common way a routing estimate comes in optimistic.
            return cheap + e * strong
        return (1 - e) * cheap + e * strong


def routing_curve(cheap_model, strong_model, input_tokens, output_tokens,
                  requests_per_month, rates=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0), **kw) -> list:
    """Blended cost against escalation rate.

    The shape to internalise: the curve is nearly flat at low escalation and
    the endpoint is the all-frontier bill. Most of the saving survives a
    surprisingly high escalation rate, which is the argument for shipping the
    router even when you cannot predict the rate.
    """
    all_strong = call_cost(strong_model, input_tokens, output_tokens, **kw) * requests_per_month
    rows = []
    for r in rates:
        plan = RoutingPlan(cheap_model, strong_model, escalation_rate=r)
        m = plan.cost_per_call(input_tokens, output_tokens, **kw) * requests_per_month
        rows.append({"escalation_rate": r, "per_month": m,
                     "vs_all_strong": m / all_strong if all_strong else 0.0,
                     "saved_per_month": all_strong - m})
    return rows


def breakeven_escalation(cheap_model, strong_model, input_tokens, output_tokens, **kw) -> float:
    """Escalation rate at which routing stops saving anything.

    Past this, you are paying twice for most requests and should just send
    everything to the strong model. Knowing this number stops a router being
    defended past the point where it helps.
    """
    cheap = call_cost(cheap_model, input_tokens, output_tokens, **kw)
    strong = call_cost(strong_model, input_tokens, output_tokens, **kw)
    if strong <= 0:
        return 0.0
    return max(0.0, min(1.0, (strong - cheap) / strong))


# --------------------------------------------------------------------------
# Self-hosting
# --------------------------------------------------------------------------

HOURS_PER_MONTH = 730


@dataclass
class SelfHostPlan:
    """The fully-loaded cost, which is the only version worth comparing.

    Compute alone always flatters self-hosting. The engineer who keeps the
    fleet alive is usually the largest line item at small scale, and the
    quality risk you now own does not appear on any invoice.
    """

    gpus_per_node: int = 8
    usd_per_gpu_hour: float = 2.50        # reserved / neocloud rate
    nodes: int = 1
    platform_fte: float = 0.75
    fte_loaded_monthly: float = 30_000.0
    utilization: float = 1.0              # you pay for reserved capacity idle or not
    extras_monthly: float = 0.0           # egress, storage, observability

    @property
    def compute_monthly(self) -> float:
        return self.nodes * self.gpus_per_node * self.usd_per_gpu_hour * HOURS_PER_MONTH

    @property
    def people_monthly(self) -> float:
        return self.platform_fte * self.fte_loaded_monthly

    @property
    def total_monthly(self) -> float:
        return self.compute_monthly + self.people_monthly + self.extras_monthly

    def __str__(self):
        return (f"  compute   {self.nodes} node(s) x {self.gpus_per_node} GPU "
                f"@ ${self.usd_per_gpu_hour:.2f}/hr   ${self.compute_monthly:>10,.0f}/mo\n"
                f"  people    {self.platform_fte:.2f} FTE loaded            "
                f"     ${self.people_monthly:>10,.0f}/mo\n"
                f"  extras                                     ${self.extras_monthly:>10,.0f}/mo\n"
                f"  TOTAL                                      ${self.total_monthly:>10,.0f}/mo")


def nodes_needed(output_tokens_per_day, tok_s_per_node, peak_factor=2.5) -> float:
    """Size for peak, not average.

    Daily volume divided by 86,400 is the average rate; traffic is not flat, so
    a peak factor of 2-3x is the difference between a sized fleet and a fleet
    that browns out every afternoon.
    """
    if tok_s_per_node <= 0:
        return float("inf")
    avg = output_tokens_per_day / 86_400
    return avg * peak_factor / tok_s_per_node


def managed_vs_selfhost(model, input_tokens_per_day, output_tokens_per_day, plan: SelfHostPlan,
                        batch=False, cached_fraction=0.0, days=30) -> dict:
    """The comparison, with the verdict and the trigger that would flip it."""
    cached = input_tokens_per_day * cached_fraction
    fresh = input_tokens_per_day - cached
    per_day = call_cost(model, fresh, output_tokens_per_day,
                        cached_input_tokens=cached, batch=batch)
    managed = per_day * days
    self_host = plan.total_monthly
    ratio = managed / self_host if self_host else float("inf")
    return {
        "managed_monthly": managed,
        "selfhost_monthly": self_host,
        "selfhost_compute_only": plan.compute_monthly,
        "ratio": ratio,
        "verdict": "managed" if managed < self_host else "self-host",
        "crossover_volume_multiple": self_host / managed if managed else float("inf"),
    }


def format_verdict(result, model_name="") -> str:
    v = result["verdict"]
    label = f"managed ({model_name})" if model_name else "managed"
    lines = [
        f"{label:<34}{'$' + format(result['managed_monthly'], ',.0f'):>12}/mo",
        f"{'self-hosted, fully loaded':<34}{'$' + format(result['selfhost_monthly'], ',.0f'):>12}/mo"
        f"   (compute alone ${result['selfhost_compute_only']:,.0f})",
        "",
        f"call: {v}.",
    ]
    if v == "managed":
        lines.append(
            f"Volume would have to grow ~{result['crossover_volume_multiple']:.1f}x before "
            "self-hosting breaks even on cost alone — and that ignores the quality and "
            "availability risk you take on with it.")
        lines.append(
            "Self-host earlier only if data residency mandates it, or the customer "
            "already owns the GPUs and the people.")
    else:
        lines.append(
            f"Managed costs {1 / result['crossover_volume_multiple']:.1f}x the self-host "
            "number. Confirm the eval passes on the open model before moving — "
            "the cost case is necessary, not sufficient.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Workload presets from the prep sheet, so the drills are reproducible
# --------------------------------------------------------------------------


@dataclass
class Workload:
    name: str
    input_tokens_per_day: float
    output_tokens_per_day: float
    notes: str = ""
    requests_per_month: float = 0.0
    input_per_request: float = 0.0
    output_per_request: float = 0.0
    tags: tuple = field(default_factory=tuple)


WORKLOADS = {
    "internal-assistant": Workload(
        "internal-assistant", 20e6, 5e6,
        "low quality bar, small volume — the answer is the cheapest tier, immediately"),
    "doc-pipeline": Workload(
        "doc-pipeline", 2e9, 500e6,
        "high volume, batchable, a 70B open model passes evals — the real comparison"),
    "chat-product": Workload(
        "chat-product", 10e6 * 1000 / 30, 10e6 * 300 / 30,
        "10M chats/month at 1K in / 300 out — the per-tier spread is the architecture",
        requests_per_month=10e6, input_per_request=1000, output_per_request=300),
}
