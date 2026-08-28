"""Quantisation evaluation: does the smaller model still behave like the big one?

Two families of measurement, and you want both:

* **Behavioural** — task accuracy on a fixed prompt set. What a product cares
  about, but noisy at small N and blind to drift that does not change answers.
* **Distributional** — how far the next-token distribution moved from the fp16
  reference (KL divergence, top-1 agreement). Sensitive, cheap, and needs no
  labels. A quant that keeps top-1 agreement above ~0.95 on your traffic is very
  unlikely to change task behaviour.

A benchmark score alone will not tell you whether INT4 is safe to ship; the pair
will.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    prompt: str
    answer: str
    tags: tuple = ()


def arithmetic_cases(n=40, seed=0, max_operand=99):
    """Deterministic, checkable, and sensitive to quantisation damage: 2-step
    arithmetic is one of the first things a bad quant loses."""
    import random

    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        a, b, c = (rng.randint(2, max_operand) for _ in range(3))
        cases.append(EvalCase(
            prompt=f"Compute ({a} + {b}) * {c}. Reply with only the number.",
            answer=str((a + b) * c),
            tags=("arithmetic",),
        ))
    return cases


def format_cases(n=20, seed=0):
    """Instruction-following that is machine-checkable without a judge."""
    import random

    rng = random.Random(seed)
    words = ["cache", "kernel", "latency", "throughput", "tensor", "batch", "token", "queue"]
    cases = []
    for _ in range(n):
        w = rng.choice(words)
        cases.append(EvalCase(
            prompt=f"Reply with exactly the word '{w}' in uppercase, nothing else.",
            answer=w.upper(),
            tags=("format",),
        ))
    return cases


_NUM = re.compile(r"-?\d[\d,]*")


def score_case(case: EvalCase, output: str) -> bool:
    """Lenient on formatting, strict on content — otherwise you measure the
    chat template rather than the model."""
    out = (output or "").strip()
    if "arithmetic" in case.tags:
        nums = [m.group(0).replace(",", "") for m in _NUM.finditer(out)]
        return bool(nums) and nums[-1] == case.answer
    return case.answer.lower() in out.lower()


@dataclass
class EvalResult:
    name: str = ""
    n: int = 0
    correct: int = 0
    by_tag: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def __str__(self):
        tags = "  ".join(f"{k}={v['correct']}/{v['n']}" for k, v in sorted(self.by_tag.items()))
        return f"{self.name:<24} {self.accuracy:6.1%}  ({self.correct}/{self.n})   {tags}"


def run_eval(cases, generate, name="model") -> EvalResult:
    """`generate(prompt) -> str`. Keeping the callable abstract means the same
    eval runs against a local pipeline, a vLLM endpoint, or an API."""
    res = EvalResult(name=name, n=len(cases))
    for case in cases:
        out = generate(case.prompt)
        ok = score_case(case, out)
        res.correct += ok
        res.outputs.append({"prompt": case.prompt, "expected": case.answer, "got": out, "ok": ok})
        for tag in case.tags or ("untagged",):
            slot = res.by_tag.setdefault(tag, {"n": 0, "correct": 0})
            slot["n"] += 1
            slot["correct"] += ok
    return res


def openai_generator(base_url, model, max_tokens=32, temperature=0.0):
    """Greedy generation against any OpenAI-compatible server."""
    import json
    import urllib.request

    def generate(prompt):
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - local server
            body = json.loads(r.read())
        return body["choices"][0]["message"]["content"]

    return generate


# --------------------------------------------------------------------------
# Distributional drift
# --------------------------------------------------------------------------


def kl_divergence(p, q, eps=1e-9) -> float:
    """KL(p || q) in nats, over aligned probability vectors."""
    return sum(pi * math.log(max(pi, eps) / max(qi, eps)) for pi, qi in zip(p, q) if pi > 0)


def top1_agreement(ref_tokens, cand_tokens) -> float:
    """Fraction of positions where the quantised model's argmax matches the
    reference's. The single most legible quantisation-damage number."""
    pairs = list(zip(ref_tokens, cand_tokens))
    if not pairs:
        return 0.0
    return sum(a == b for a, b in pairs) / len(pairs)


def perplexity(logprobs) -> float:
    """exp of mean negative log-likelihood. Same text, same tokenizer, or the
    comparison is meaningless."""
    lps = [lp for lp in logprobs if lp is not None]
    if not lps:
        return float("nan")
    return math.exp(-sum(lps) / len(lps))


def compare_table(results, latencies=None, sizes_gb=None, usd_per_hour=None) -> str:
    """The lab-5 deliverable: quality, speed and cost in one block, because any
    one of them alone justifies the wrong decision."""
    lines = [f"{'variant':<20}{'accuracy':>10}{'tok/s':>10}{'p50 TTFT':>12}{'VRAM':>10}{'$/1M tok':>12}"]
    for r in results:
        lat = (latencies or {}).get(r.name, {})
        tps = lat.get("output_throughput")
        ttft = lat.get("ttft_p50")
        size = (sizes_gb or {}).get(r.name)
        cost = ""
        if usd_per_hour and tps:
            cost = f"${usd_per_hour / 3600 / tps * 1e6:,.2f}"
        lines.append(
            f"{r.name:<20}{r.accuracy:>9.1%}"
            f"{(f'{tps:,.0f}' if tps else '-'):>10}"
            f"{(f'{ttft * 1000:,.0f}ms' if ttft else '-'):>12}"
            f"{(f'{size:.1f}G' if size else '-'):>10}"
            f"{cost or '-':>12}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# How many samples do you actually need?
# --------------------------------------------------------------------------


def accuracy_stderr(accuracy, n) -> float:
    """Standard error of a proportion: sqrt(p(1-p)/n).

    At n=60 and p=0.8 this is about 5 points, so a 95% interval spans ~20
    points. Saying that out loud before anyone celebrates a 3-point improvement
    is the cheapest credibility in the room.
    """
    if n <= 0:
        return float("nan")
    return math.sqrt(max(accuracy * (1 - accuracy), 0.0) / n)


def accuracy_ci(accuracy, n, z=1.96):
    se = accuracy_stderr(accuracy, n)
    return (max(0.0, accuracy - z * se), min(1.0, accuracy + z * se))


def min_samples_for_detectable_difference(baseline, delta, power_z=2.8) -> int:
    """Roughly how many cases to detect a `delta` change from `baseline`.

    Two-proportion rule of thumb; `power_z` ~2.8 covers 95% confidence at 80%
    power. Use it to answer "is our eval big enough?" with a number instead of
    a shrug — and to set expectations before a week is spent on a difference
    the harness could never have seen.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    p = min(max(baseline, 1e-6), 1 - 1e-6)
    return int(math.ceil(2 * (power_z ** 2) * p * (1 - p) / (delta ** 2)))


def bootstrap_ci(scores, n_resamples=2000, alpha=0.05, seed=0):
    """Percentile bootstrap over per-case scores.

    Works for any metric you can compute per case, not just proportions, and it
    is the honest way to put error bars on an LLM-judge score.
    """
    import random

    vals = [float(s) for s in scores if s is not None]
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_resamples):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


# --------------------------------------------------------------------------
# Calibrating an LLM judge against humans
# --------------------------------------------------------------------------


def cohens_kappa(a_labels, b_labels) -> float:
    """Agreement corrected for chance.

    Raw agreement flatters a judge on skewed data: if 90% of answers are good,
    a judge that says "good" every time scores 90%. Kappa subtracts that.
    Rough reading: >0.8 strong, 0.6-0.8 usable with care, <0.6 means the judge
    is producing noise you would be automating.
    """
    pairs = [(a, b) for a, b in zip(a_labels, b_labels) if a is not None and b is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    expected = sum(
        (sum(a == lbl for a, _ in pairs) / n) * (sum(b == lbl for _, b in pairs) / n)
        for lbl in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


@dataclass
class JudgeCalibration:
    n: int = 0
    agreement: float = 0.0
    kappa: float = 0.0
    false_pass: int = 0     # judge said good, human said bad — the dangerous one
    false_fail: int = 0     # judge said bad, human said good — merely annoying
    verdict: str = ""

    def __str__(self):
        return (f"judge vs human on {self.n} cases: agreement {self.agreement:.1%}, "
                f"kappa {self.kappa:.2f}\n"
                f"  false passes {self.false_pass}   false fails {self.false_fail}\n"
                f"  {self.verdict}")


def calibrate_judge(human_labels, judge_labels, target_agreement=0.90) -> JudgeCalibration:
    """Check the judge before trusting a single number it produces.

    An uncalibrated LLM judge is a vibe check with a spreadsheet attached. The
    step everyone skips: label 50-100 cases by hand once, measure agreement and
    kappa, and only then let the judge run unattended. False passes matter more
    than false fails — a judge that waves bad answers through is worse than no
    judge, because it manufactures confidence.
    """
    pairs = [(h, j) for h, j in zip(human_labels, judge_labels) if h is not None and j is not None]
    n = len(pairs)
    agree = sum(h == j for h, j in pairs) / n if n else 0.0
    cal = JudgeCalibration(
        n=n,
        agreement=agree,
        kappa=cohens_kappa([h for h, _ in pairs], [j for _, j in pairs]),
        false_pass=sum(1 for h, j in pairs if j and not h),
        false_fail=sum(1 for h, j in pairs if h and not j),
    )
    if cal.agreement >= target_agreement and cal.kappa >= 0.6:
        cal.verdict = "usable — run it unattended, re-calibrate when the prompt or model changes"
    elif cal.kappa < 0.4:
        cal.verdict = ("NOT usable — near chance agreement. Fix the rubric before the judge: "
                       "ambiguous criteria produce disagreement no model can resolve.")
    else:
        cal.verdict = ("borderline — tighten the rubric, add few-shot examples of edge cases, "
                       "or keep humans in the loop on the failures only")
    return cal


# --------------------------------------------------------------------------
# Groundedness
# --------------------------------------------------------------------------


def groundedness(answer, sources, ngram=4) -> float:
    """Fraction of the answer's n-grams that appear in the retrieved sources.

    A cheap, model-free proxy for "did it make this up". It is a *lower bound*
    on faithfulness — a correct paraphrase scores low — so use it as a screen
    that flags candidates for a judge or a human, not as the verdict.
    """
    a = tokenize_words(answer)
    if len(a) < ngram:
        return 1.0
    src = " ".join(tokenize_words(" ".join(sources)))
    grams = [" ".join(a[i:i + ngram]) for i in range(len(a) - ngram + 1)]
    if not grams:
        return 1.0
    return sum(1 for g in grams if g in src) / len(grams)


def tokenize_words(text) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def citation_validity(answer, valid_ids) -> dict:
    """Do the cited ids exist? An invented citation is the worst failure mode
    there is — it looks like evidence."""
    cited = set(re.findall(r"\[([A-Za-z0-9_#\-\.]+)\]", answer or ""))
    valid = set(valid_ids)
    invalid = sorted(cited - valid)
    return {"cited": sorted(cited), "invalid": invalid,
            "valid_rate": (len(cited & valid) / len(cited)) if cited else 1.0,
            "has_citations": bool(cited)}


def refusal_correctness(cases, answers, refusal_markers=("i don't know", "not covered",
                                                         "no information", "cannot find")) -> dict:
    """Did it refuse when it should have, and answer when it could?

    `cases` carry an `answerable` flag. The unanswerable questions are the ones
    worth over-weighting in a golden set: confident answers to questions the
    corpus cannot support are how a RAG system loses a customer's trust in one
    screenshot.
    """
    refused_ok = wrongly_refused = hallucinated = answered_ok = 0
    for case, answer in zip(cases, answers):
        refused = any(m in (answer or "").lower() for m in refusal_markers)
        if case.get("answerable", True):
            answered_ok += not refused
            wrongly_refused += refused
        else:
            refused_ok += refused
            hallucinated += not refused
    unanswerable = refused_ok + hallucinated
    return {
        "correct_refusal_rate": refused_ok / unanswerable if unanswerable else 1.0,
        "hallucinated_on_unanswerable": hallucinated,
        "wrongly_refused": wrongly_refused,
        "answered_when_possible": answered_ok,
    }


# --------------------------------------------------------------------------
# The regression gate
# --------------------------------------------------------------------------


@dataclass
class GateResult:
    passed: bool = True
    reasons: list = field(default_factory=list)
    deltas: dict = field(default_factory=dict)

    def __str__(self):
        head = "PASS" if self.passed else "FAIL"
        lines = [f"regression gate: {head}"]
        for k, d in sorted(self.deltas.items()):
            lines.append(f"  {k:<28}{d['baseline']:>8.3f} -> {d['candidate']:>8.3f}"
                         f"   ({d['delta']:+.3f})")
        lines.extend(f"  ! {r}" for r in self.reasons)
        return "\n".join(lines)


def regression_gate(baseline: dict, candidate: dict, tolerances=None, n=None) -> GateResult:
    """Block a change that makes a metric worse by more than its tolerance.

    This is the artifact that converts "it feels worse lately" into a red build.
    Set tolerances above the noise floor — `accuracy_stderr` tells you where
    that is — or the gate will fail on sampling noise and be switched off within
    a month, which is worse than not having it.
    """
    tolerances = tolerances or {}
    result = GateResult()
    for key, base in baseline.items():
        if key not in candidate or not isinstance(base, (int, float)):
            continue
        cand = candidate[key]
        tol = tolerances.get(key, 0.02)
        delta = cand - base
        result.deltas[key] = {"baseline": base, "candidate": cand, "delta": delta,
                              "tolerance": tol}
        if delta < -tol:
            result.passed = False
            noise = ""
            if n:
                se = accuracy_stderr(base, n)
                if abs(delta) < 2 * se:
                    noise = (f" (within 2 SE = {2 * se:.3f} at n={n} — widen the eval "
                             "before treating this as real)")
            result.reasons.append(f"{key} dropped {abs(delta):.3f}, tolerance {tol:.3f}{noise}")
    return result
