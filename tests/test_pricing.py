import datetime as dt

import pytest

from servlab import pricing as pr


def test_chat_workload_reproduces_the_tier_spread():
    # 10M chats/month at 1K in / 300 out — the drill you may be asked live.
    rows = pr.compare(["gemini-3.1-pro", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
                      1000, 300, requests_per_month=10e6)
    by_name = {r["model"]: r for r in rows}
    assert by_name["gemini-3.1-pro"]["per_month"] == pytest.approx(56_000)
    assert by_name["gemini-3.6-flash"]["per_month"] == pytest.approx(37_500)
    assert by_name["gemini-3.5-flash-lite"]["per_month"] == pytest.approx(10_500)
    # The 5x spread between tiers is the whole architecture conversation.
    assert by_name["gemini-3.1-pro"]["vs_cheapest"] > 5


def test_batch_is_half_price():
    full = pr.call_cost("gemini-3.5-flash-lite", 1e6, 1e6)
    assert pr.call_cost("gemini-3.5-flash-lite", 1e6, 1e6, batch=True) == pytest.approx(full / 2)


def test_cached_input_bills_at_a_tenth():
    fresh = pr.call_cost("gemini-3.1-pro", 100_000, 0)
    cached = pr.call_cost("gemini-3.1-pro", 0, 0, cached_input_tokens=100_000)
    assert cached == pytest.approx(fresh * 0.10)


def test_thinking_tokens_bill_as_output():
    # The usual explanation for a bill that doubled with no traffic change.
    quiet = pr.call_cost("gemini-3.1-pro", 1000, 300)
    thinking = pr.call_cost("gemini-3.1-pro", 1000, 300, thinking_tokens=2000)
    assert thinking > 2 * quiet
    assert thinking - quiet == pytest.approx(2000 * 12.00 / 1e6)


def test_context_cliff_applies_past_the_threshold():
    below = pr.call_cost("gemini-3.1-pro", 100_000, 1000)
    above = pr.call_cost("gemini-3.1-pro", 300_000, 1000)
    # 3x the input tokens but more than 3x the cost: both rates stepped up.
    assert above > 3 * below


def test_routing_saves_most_of_the_bill_even_at_high_escalation():
    curve = pr.routing_curve("gemini-3.5-flash-lite", "gemini-3.1-pro", 1000, 300, 10e6)
    at_zero = next(r for r in curve if r["escalation_rate"] == 0)
    at_thirty = next(r for r in curve if r["escalation_rate"] == 0.3)
    assert at_zero["vs_all_strong"] < 0.25
    assert at_thirty["vs_all_strong"] < 0.6      # still half the all-frontier bill


def test_routing_stops_paying_once_escalation_is_high_enough():
    # Past break-even you are paying twice for most requests; just use the
    # strong model. Knowing this number stops a router being over-defended.
    be = pr.breakeven_escalation("gemini-3.5-flash-lite", "gemini-3.1-pro", 1000, 300)
    assert 0 < be < 1
    plan = pr.RoutingPlan("gemini-3.5-flash-lite", "gemini-3.1-pro", escalation_rate=1.0)
    all_strong = pr.call_cost("gemini-3.1-pro", 1000, 300)
    assert plan.cost_per_call(1000, 300) > all_strong


def test_selfhost_people_cost_is_not_optional():
    plan = pr.SelfHostPlan(nodes=1, platform_fte=0.75, fte_loaded_monthly=30_000)
    assert plan.people_monthly == 22_500
    assert plan.total_monthly > plan.compute_monthly


def test_doc_pipeline_verdict_is_managed_and_names_the_flip_condition():
    w = pr.WORKLOADS["doc-pipeline"]
    plan = pr.SelfHostPlan(nodes=4)
    result = pr.managed_vs_selfhost("gemini-3.5-flash-lite", w.input_tokens_per_day,
                                    w.output_tokens_per_day, plan, batch=True)
    assert result["verdict"] == "managed"
    assert result["crossover_volume_multiple"] > 1
    assert "residency" in pr.format_verdict(result)


def test_small_workload_never_justifies_self_hosting():
    w = pr.WORKLOADS["internal-assistant"]
    monthly = pr.daily_cost("gemini-3.5-flash-lite", w.input_tokens_per_day,
                            w.output_tokens_per_day) * 30
    assert monthly < 600           # one engineer-day costs more than the year
    assert monthly < pr.SelfHostPlan(nodes=1).people_monthly


def test_size_for_peak_not_average():
    avg_only = 500e6 / 86_400 / 3500
    with_peak = pr.nodes_needed(500e6, tok_s_per_node=3500, peak_factor=2.5)
    assert with_peak == pytest.approx(avg_only * 2.5)


def test_staleness_warns_when_the_table_is_old():
    assert "!!" not in pr.staleness(pr.VERIFIED_ON + dt.timedelta(days=5))
    assert "!!" in pr.staleness(pr.VERIFIED_ON + dt.timedelta(days=90))
