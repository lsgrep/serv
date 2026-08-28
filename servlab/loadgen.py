"""Load generator for OpenAI-compatible endpoints (vLLM, SGLang, TGI, llama.cpp).

Two modes, and the difference between them is the single most important thing
to understand about load testing:

* **Closed loop** (`concurrency=N`): N workers, each sends the next request when
  its previous one finishes. Offered load *falls* when the server slows down, so
  the server can never be overloaded. Good for measuring peak throughput.
* **Open loop** (`rps=R`): requests are launched on a Poisson schedule regardless
  of whether the server is keeping up. Queues grow without bound past capacity.
  This is what real traffic does, and the only way to reproduce a death spiral.

If a benchmark shows a server degrading gracefully forever, it was closed loop.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time

from .stats import RequestResult, summarize

# Deliberately banal filler: we want a predictable token count, not a prompt
# that trips a prefix cache or wanders into a refusal.
_FILLER = (
    "the quick brown fox jumps over the lazy dog while the server quietly "
    "fills its key value cache one token at a time and the queue grows "
)


def make_prompt(target_tokens=256, seed=None, unique=True):
    """A prompt of roughly `target_tokens` tokens (~0.75 words/token).

    `unique=True` prefixes a random nonce so runs do not silently benefit from
    prefix caching — turn it off when you *want* to measure prefix-cache hits.
    """
    rng = random.Random(seed)
    words = _FILLER.split()
    n_words = max(4, int(target_tokens * 0.75))
    body = " ".join(words[i % len(words)] for i in range(n_words))
    if unique:
        body = f"[{rng.randrange(10**9)}] " + body
    return body


async def _one_request(client, base_url, model, prompt, max_tokens, scheduled, temperature, extra):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Ask for exactly max_tokens so output length is a control variable,
        # not something the model decides for us.
        "ignore_eos": True,
        **(extra or {}),
    }
    r = RequestResult(start=time.perf_counter(), scheduled=scheduled)
    try:
        async with client.stream("POST", url, json=payload) as resp:
            r.status = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                r.error = body[:200].decode("utf-8", "replace")
                r.end = time.perf_counter()
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    if not r.first_token:
                        r.first_token = time.perf_counter()
                    r.output_tokens += 1
                    r.text += piece
                usage = chunk.get("usage")
                if usage:
                    r.prompt_tokens = usage.get("prompt_tokens", r.prompt_tokens)
    except Exception as exc:  # noqa: BLE001 - a failed request is data, not a crash
        r.error = f"{type(exc).__name__}: {exc}"
    r.end = time.perf_counter()
    return r


async def _drive(base_url, model, *, rps=None, concurrency=None, duration=30.0, n_requests=None,
                 prompt_tokens=256, max_tokens=128, temperature=0.0, seed=0, timeout=600.0,
                 on_result=None, extra=None, unique_prompts=True):
    import httpx

    rng = random.Random(seed)
    results = []
    limits = httpx.Limits(max_connections=(concurrency or 1024) + 8, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0), limits=limits) as client:
        t0 = time.perf_counter()

        def _record(res):
            results.append(res)
            if on_result:
                on_result(res)

        if concurrency:
            async def worker(wid):
                i = 0
                while True:
                    if n_requests is not None and len(results) >= n_requests:
                        return
                    if duration is not None and time.perf_counter() - t0 >= duration:
                        return
                    prompt = make_prompt(prompt_tokens, seed=rng.randrange(10**9), unique=unique_prompts)
                    _record(await _one_request(client, base_url, model, prompt, max_tokens,
                                               time.perf_counter(), temperature, extra))
                    i += 1

            await asyncio.gather(*(worker(w) for w in range(concurrency)))
        else:
            if not rps:
                raise ValueError("pass rps= (open loop) or concurrency= (closed loop)")
            tasks = []
            next_at = t0
            sent = 0
            while True:
                if n_requests is not None and sent >= n_requests:
                    break
                if duration is not None and next_at - t0 >= duration:
                    break
                now = time.perf_counter()
                if next_at > now:
                    await asyncio.sleep(next_at - now)
                prompt = make_prompt(prompt_tokens, seed=rng.randrange(10**9), unique=unique_prompts)
                scheduled = next_at

                async def go(p=prompt, s=scheduled):
                    _record(await _one_request(client, base_url, model, p, max_tokens, s, temperature, extra))

                tasks.append(asyncio.create_task(go()))
                sent += 1
                # Poisson arrivals: exponential gaps. Uniform gaps understate
                # queueing badly — bursts are where servers actually fall over.
                next_at += rng.expovariate(rps)
            if tasks:
                await asyncio.gather(*tasks)
    return results


def run_load(base_url="http://localhost:8000", model="", *, rps=None, concurrency=None,
             duration=30.0, n_requests=None, prompt_tokens=256, max_tokens=128,
             temperature=0.0, seed=0, timeout=600.0, on_result=None, extra=None,
             unique_prompts=True):
    """Blocking entry point, safe to call from a notebook cell.

    Runs the event loop on its own thread so it works whether or not the caller
    already has one (Jupyter always does).
    """
    box = {}

    def target():
        try:
            box["results"] = asyncio.run(_drive(
                base_url, model, rps=rps, concurrency=concurrency, duration=duration,
                n_requests=n_requests, prompt_tokens=prompt_tokens, max_tokens=max_tokens,
                temperature=temperature, seed=seed, timeout=timeout, on_result=on_result,
                extra=extra, unique_prompts=unique_prompts,
            ))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    th = threading.Thread(target=target, name="servlab-loadgen")
    th.start()
    th.join()
    if "error" in box:
        raise box["error"]
    return box["results"]


def sweep(base_url, model, levels, *, mode="concurrency", duration=20.0, warmup=3.0,
          slo_ttft=1.0, slo_tpot=0.05, **kw):
    """Run the same workload at several load levels and summarise each.

    Returns a list of dicts ready for `pandas.DataFrame(...)`. This is lab 2's
    workhorse: the same call on three runtime types is the hardware comparison.
    """
    rows = []
    for level in levels:
        if warmup:
            run_load(base_url, model, concurrency=2, duration=warmup, **kw)
        kwargs = {mode: level}
        t0 = time.perf_counter()
        results = run_load(base_url, model, duration=duration, **kwargs, **kw)
        s = summarize(results, duration=time.perf_counter() - t0,
                      slo_ttft=slo_ttft, slo_tpot=slo_tpot)
        rows.append({
            "level": level,
            "mode": mode,
            "n": s.n,
            "failed": s.failed,
            "req_per_s": s.request_throughput,
            "out_tok_per_s": s.output_throughput,
            "ttft_p50": s.ttft.get("p50"),
            "ttft_p99": s.ttft.get("p99"),
            "tpot_p50": s.tpot.get("p50"),
            "e2e_p99": s.e2e.get("p99"),
            "goodput": s.goodput,
        })
        print(f"[{mode}={level}] {s.request_throughput:.2f} req/s  "
              f"ttft p50={_ms(s.ttft.get('p50'))} p99={_ms(s.ttft.get('p99'))}  "
              f"goodput={s.goodput:.2f} req/s  failed={s.failed}")
    return rows


def _ms(v):
    return "-" if v is None else f"{v*1000:,.0f}ms"
