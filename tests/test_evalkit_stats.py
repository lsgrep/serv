import pytest

from servlab import evalkit as ek


def test_sixty_prompts_cannot_detect_a_three_point_difference():
    # The humility number to say out loud before anyone celebrates a small win.
    lo, hi = ek.accuracy_ci(0.80, 60)
    assert hi - lo > 0.15
    assert ek.min_samples_for_detectable_difference(0.80, 0.03) > 1000
    assert ek.min_samples_for_detectable_difference(0.80, 0.10) < 300


def test_bootstrap_ci_brackets_the_mean():
    scores = [1.0] * 80 + [0.0] * 20
    lo, hi = ek.bootstrap_ci(scores, n_resamples=500)
    assert lo < 0.8 < hi
    assert hi - lo < 0.25


def test_kappa_punishes_a_judge_that_always_says_yes():
    human = [1] * 90 + [0] * 10
    lazy = [1] * 100
    cal = ek.calibrate_judge(human, lazy)
    assert cal.agreement == pytest.approx(0.90)   # looks great
    assert cal.kappa == pytest.approx(0.0)        # is worthless
    assert "NOT usable" in cal.verdict


def test_a_good_judge_is_marked_usable():
    human = [1] * 50 + [0] * 50
    judge = [1] * 48 + [0] * 2 + [1] * 3 + [0] * 47
    cal = ek.calibrate_judge(human, judge)
    assert cal.agreement >= 0.90
    assert cal.kappa > 0.8
    assert "usable" in cal.verdict


def test_false_passes_are_counted_separately_from_false_fails():
    cal = ek.calibrate_judge([0, 0, 1, 1], [1, 0, 0, 1])
    assert cal.false_pass == 1     # judge waved a bad answer through
    assert cal.false_fail == 1


def test_groundedness_flags_invented_content():
    src = ["refunds are issued within five to ten business days of approval"]
    grounded = ek.groundedness("refunds are issued within five to ten business days", src)
    invented = ek.groundedness("refunds are issued instantly by wire to any bank account", src)
    assert grounded > 0.8
    assert invented < 0.2


def test_citation_validity_catches_invented_references():
    out = ek.citation_validity("See [refunds#0] and [handbook#9].", ["refunds#0", "sla#1"])
    assert out["invalid"] == ["handbook#9"]
    assert out["valid_rate"] == pytest.approx(0.5)


def test_refusal_metrics_separate_the_two_ways_to_be_wrong():
    cases = [{"answerable": True}, {"answerable": False}, {"answerable": False}]
    answers = ["the refund window is 30 days", "I don't know", "it is 42 days"]
    m = ek.refusal_correctness(cases, answers)
    assert m["correct_refusal_rate"] == pytest.approx(0.5)
    assert m["hallucinated_on_unanswerable"] == 1
    assert m["wrongly_refused"] == 0


def test_gate_blocks_a_real_regression():
    gate = ek.regression_gate({"accuracy": 0.90}, {"accuracy": 0.80}, {"accuracy": 0.02}, n=2000)
    assert not gate.passed
    assert "accuracy dropped" in gate.reasons[0]


def test_gate_says_when_a_failure_is_within_sampling_noise():
    # A gate that fires on noise gets switched off within a month.
    gate = ek.regression_gate({"accuracy": 0.86}, {"accuracy": 0.82}, {"accuracy": 0.02}, n=200)
    assert not gate.passed
    assert "within 2 SE" in gate.reasons[0]


def test_gate_ignores_improvements_and_non_numeric_fields():
    gate = ek.regression_gate({"accuracy": 0.80, "note": "x"}, {"accuracy": 0.95, "note": "y"})
    assert gate.passed
    assert "note" not in gate.deltas
