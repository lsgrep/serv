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
