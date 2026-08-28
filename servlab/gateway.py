"""A thin model gateway, and the exit cost it is there to keep small.

The argument this module encodes: **your eval suite is your exit option.** If a
model swap is a measured afternoon rather than a quarter of unquantified risk,
vendor choice stays an economic decision instead of becoming a hostage
situation. That is worth saying while wearing any vendor's badge, and it is more
persuasive when you can put a number on the exit.

`switching_cost()` does that, term by term. Run it before repeating the folk
wisdom: at current embedding prices, re-embedding even a billion-token corpus is
a couple of hundred dollars of API spend. The expensive terms are re-running
fine-tunes (which are non-portable by construction) and the engineering days to
re-ingest and re-validate. "Embeddings are the lock-in" is true about the
*pipeline*, not the bill.

The client here is deliberately small — one interface, three provider shapes,
no dependencies. Real gateways add retries, budgets, and observability; the
point of this one is that the abstraction is thirty lines, so "we can't afford
portability" is rarely true.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import pricing


class ProviderError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    raw: dict = field(default_factory=dict)


def _post(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"{exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(str(exc)) from exc


class OpenAICompatible:
    """Covers far more than OpenAI: vLLM, SGLang, Together, Groq, Fireworks,
    and Gemini's OpenAI-compatibility endpoint all speak this shape. Defaulting
    to it is most of portability for free."""

    name = "openai-compatible"

    def __init__(self, base_url="http://localhost:8000", api_key_env="OPENAI_API_KEY", model=None):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model

    def complete(self, prompt, model=None, max_tokens=256, temperature=0.0, system=None):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        key = os.environ.get(self.api_key_env, "")
        body = _post(f"{self.base_url}/v1/chat/completions",
                     {"model": model or self.model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
                     {"Authorization": f"Bearer {key}"} if key else {})
        usage = body.get("usage", {})
        return Completion(
            text=body["choices"][0]["message"]["content"],
            model=body.get("model", model or self.model or ""), provider=self.name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0), raw=body)


class AnthropicMessages:
    name = "anthropic"

    def __init__(self, base_url="https://api.anthropic.com", api_key_env="ANTHROPIC_API_KEY",
                 version="2023-06-01", model=None):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.version = version
        self.model = model

    def complete(self, prompt, model=None, max_tokens=256, temperature=0.0, system=None):
        payload = {"model": model or self.model, "max_tokens": max_tokens,
                   "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        if system:
            payload["system"] = system     # a top-level field here, a message elsewhere
        body = _post(f"{self.base_url}/v1/messages", payload,
                     {"x-api-key": os.environ.get(self.api_key_env, ""),
                      "anthropic-version": self.version})
        usage = body.get("usage", {})
        return Completion(
            text="".join(b.get("text", "") for b in body.get("content", [])),
            model=body.get("model", model or self.model or ""), provider=self.name,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0), raw=body)


class VertexGemini:
    """`generateContent`. Note the shape differences that a gateway absorbs:
    `contents`/`parts` instead of `messages`, `systemInstruction` as its own
    field, and token counts under `usageMetadata`."""

    name = "vertex"

    def __init__(self, project=None, location="us-central1", model="gemini-2.5-flash-lite",
                 token_env="GOOGLE_ACCESS_TOKEN"):
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.location = location
        self.model = model
        self.token_env = token_env

    def complete(self, prompt, model=None, max_tokens=256, temperature=0.0, system=None):
        m = model or self.model
        url = (f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project}"
               f"/locations/{self.location}/publishers/google/models/{m}:generateContent")
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        body = _post(url, payload, {"Authorization": f"Bearer {os.environ.get(self.token_env, '')}"})
        cand = (body.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        usage = body.get("usageMetadata", {})
        return Completion(text=text, model=m, provider=self.name,
                          input_tokens=usage.get("promptTokenCount", 0),
                          output_tokens=usage.get("candidatesTokenCount", 0), raw=body)


@dataclass
class Route:
    provider: object
    model: str
    price_key: str = ""      # key into servlab.pricing.MODELS, for cost accounting


class Gateway:
    """One `complete()` for every provider, with fallback and cost accounting.

    The fallback chain is the part that earns its keep operationally — provider
    incidents are not rare — and it is the same mechanism that makes a model
    swap a config change. Build it on day one and portability is free; retrofit
    it later and it is a quarter.
    """

    def __init__(self, routes=None, default=None):
        self.routes = dict(routes or {})
        self.default = default or (next(iter(self.routes)) if self.routes else None)
        self.log = []

    def register(self, alias, provider, model, price_key=""):
        self.routes[alias] = Route(provider, model, price_key)
        self.default = self.default or alias
        return self

    def complete(self, prompt, alias=None, fallbacks=(), **kw) -> Completion:
        chain = [alias or self.default, *fallbacks]
        errors = []
        for name in chain:
            route = self.routes.get(name)
            if route is None:
                errors.append(f"{name}: not registered")
                continue
            t0 = time.perf_counter()
            try:
                out = route.provider.complete(prompt, model=route.model, **kw)
            except ProviderError as exc:
                errors.append(f"{name}: {exc}")
                continue
            out.latency_s = time.perf_counter() - t0
            if route.price_key:
                out.cost_usd = pricing.call_cost(route.price_key, out.input_tokens, out.output_tokens)
            self.log.append({"alias": name, "model": out.model, "latency_s": out.latency_s,
                             "cost_usd": out.cost_usd, "input_tokens": out.input_tokens,
                             "output_tokens": out.output_tokens})
            return out
        raise ProviderError("every route failed:\n  " + "\n  ".join(errors))

    def spend(self) -> float:
        return sum(e["cost_usd"] for e in self.log)

    def summary(self) -> str:
        if not self.log:
            return "no calls"
        by = {}
        for e in self.log:
            s = by.setdefault(e["alias"], {"n": 0, "cost": 0.0, "latency": 0.0})
            s["n"] += 1
            s["cost"] += e["cost_usd"]
            s["latency"] += e["latency_s"]
        lines = [f"{'route':<20}{'calls':>7}{'cost':>12}{'mean latency':>15}"]
        for alias, s in sorted(by.items()):
            lines.append(f"{alias:<20}{s['n']:>7}{'$' + format(s['cost'], '.4f'):>12}"
                         f"{s['latency'] / s['n']:>14.2f}s")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# What would it actually cost to leave?
# --------------------------------------------------------------------------

LOCKIN_VECTORS = {
    "embeddings": "Switching embedding models invalidates every vector, so the corpus is "
                  "re-embedded and every index rebuilt. The API spend for that is minor; "
                  "the cost is the ingestion pipeline re-run and the revalidation. Keep the "
                  "raw text and chunk boundaries and it stays a batch job instead of a project.",
    "fine-tunes": "Non-portable by construction. A fine-tune on vendor A is not a thing "
                  "vendor B can load; you re-run the job, and you re-validate it.",
    "proprietary features": "Grounding with a vendor's search index, built-in tools, "
                            "vendor-specific caching semantics. Each one is a feature you "
                            "would have to rebuild rather than re-point.",
    "prompts and evals": "The subtle one. Prompts drift toward one model's quirks, and an "
                         "eval suite tuned on those prompts quietly certifies the incumbent. "
                         "Keep prompts model-neutral and test on two models from the start.",
    "data egress": "Moving the corpus and the logs out has a bill attached. Small next to "
                   "the others until the corpus is large.",
}


def switching_cost(corpus_tokens=0, embed_price_per_m=0.15, eval_cases=200,
                   eval_tokens_per_case=3000, eval_price_per_m=1.5, engineer_days=10,
                   engineer_day_rate=1200, fine_tune_reruns=0, fine_tune_cost_each=0,
                   egress_gb=0, egress_per_gb=0.12) -> dict:
    """Price the exit, term by term.

    Run it before arguing from intuition. The usual shape: fine-tune reruns and
    engineering dominate, embedding API spend is a rounding error, and the term
    you can actually shrink in advance is the engineering one — by keeping raw
    text and chunk boundaries so re-embedding is a batch job rather than a
    re-ingestion project, and by never letting a prompt suite become
    single-model.
    """
    embed = corpus_tokens / 1e6 * embed_price_per_m
    evals = eval_cases * eval_tokens_per_case / 1e6 * eval_price_per_m
    people = engineer_days * engineer_day_rate
    tunes = fine_tune_reruns * fine_tune_cost_each
    egress = egress_gb * egress_per_gb
    total = embed + evals + people + tunes + egress
    terms = {"re-embed corpus": embed, "re-run evals": evals,
             "engineering": people, "re-run fine-tunes": tunes, "data egress": egress}
    dominant = max(terms.items(), key=lambda kv: kv[1])
    return {"terms": terms, "total_usd": total, "dominant_term": dominant[0],
            "dominant_usd": dominant[1]}


def format_switching_cost(result) -> str:
    lines = ["cost to move this workload to another model/vendor:"]
    for name, usd in sorted(result["terms"].items(), key=lambda kv: -kv[1]):
        if usd:
            lines.append(f"  {name:<22}${usd:>12,.0f}")
    lines.append(f"  {'TOTAL':<22}${result['total_usd']:>12,.0f}")
    lines.append("")
    lines.append(f"Dominated by: {result['dominant_term']} "
                 f"(${result['dominant_usd']:,.0f}, "
                 f"{result['dominant_usd'] / (result['total_usd'] or 1):.0%} of the exit).")
    lines.append("")
    lines.append("The goal is not zero dependency — it is a dependency you chose knowingly, "
                 "with an exit cost you have measured rather than feared.")
    return "\n".join(lines)
