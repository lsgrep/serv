"""GPT-2 scale engine: measure what the napkin math predicted.

Two things live here.

1. `decode_curve` — per-token latency with and without a KV cache, as context
   grows. Without a cache, step N re-attends over N tokens, so the curve bends
   upward; with one, it is flat until memory traffic catches up. The gap between
   the two curves *is* the reason the KV cache exists, and plotting it is the
   whole point of the exercise.

2. `BatchedEngine` — the `Scheduler` from this package driving real GPT-2
   forward passes, so continuous batching, admission control and preemption are
   observed on a live model rather than simulated.

   Honest about the shortcut: attention here uses HuggingFace's contiguous
   per-sequence cache, not a paged kernel. The block allocator still sizes and
   accounts memory in blocks — so the *policy* is real and the *kernel* is not.
   Writing a paged attention kernel is a different lab.

Requires torch + transformers; import it only in cells that have a GPU.
"""

from __future__ import annotations

import time

from ..napkin import ModelSpec, kv_bytes_per_token
from .allocator import BlockAllocator
from .scheduler import Request, Scheduler, SchedulerConfig


def load(name="gpt2", device=None, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # fp16 on CPU is slow and partly unimplemented — keep CPU runs in fp32.
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return model, tok


def spec_from_hf(model, name=None) -> ModelSpec:
    """Build a `ModelSpec` from a live HF config, so the napkin math in lab 3
    is fed by the same numbers the measurement uses."""
    c = model.config
    n_heads = getattr(c, "num_attention_heads", None) or c.n_head
    n_layers = getattr(c, "num_hidden_layers", None) or c.n_layer
    hidden = getattr(c, "hidden_size", None) or c.n_embd
    n_kv = getattr(c, "num_key_value_heads", n_heads)
    head_dim = getattr(c, "head_dim", None) or hidden // n_heads
    ffn = getattr(c, "intermediate_size", None) or 4 * hidden
    return ModelSpec(
        name=name or getattr(c, "_name_or_path", "model"),
        n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
        hidden=hidden, ffn=ffn, vocab=c.vocab_size,
        params=sum(p.numel() for p in model.parameters()),
    )


def _sync(device):
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def long_prompt(tok, n_tokens=512, seed=0) -> str:
    """A prompt of roughly `n_tokens` tokens.

    The measurement below needs real context to say anything. With a five-token
    prompt both paths are dominated by kernel-launch overhead and the curves
    tell you about Python, not about attention.
    """
    import random

    rng = random.Random(seed)
    words = ("the cache stores keys and values so that each new token attends over "
             "everything before it without recomputing any of the prefix again ").split()
    text = " ".join(words[rng.randrange(len(words))] for _ in range(int(n_tokens * 1.2)))
    ids = tok(text, return_tensors="pt").input_ids[0][:n_tokens]
    return tok.decode(ids)


def decode_curve(model, tok, prompt=None, prompt_tokens=512, n_new=48, use_cache=True,
                 warmup=4, max_context=None):
    """Time every decode step. Returns `[{"step", "context_len", "ms", "cache"}]`.

    Two things here are the difference between a measurement and a plausible
    looking artifact, and both are mistakes worth having made once:

    * **The prompt has to be long.** Without a cache, step *n* re-reads the whole
      prefix; with one, it reads a single token. From a five-token prompt those
      are both "one small forward pass" and the curves cross over noise. The
      default builds a few hundred tokens of context so the comparison is real.
    * **Warm-up must run on the path being measured**, inside this call. Warming
      up once and then timing two configurations lets the second one inherit the
      first one's warm allocator and autotuned kernels — which is how a naive
      benchmark concludes that *no* cache is faster.
    """
    import torch

    device = next(model.parameters()).device
    text = prompt if prompt is not None else long_prompt(tok, prompt_tokens)
    base = tok(text, return_tensors="pt").input_ids.to(device)

    limit = max_context or getattr(model.config, "n_positions", None) or 1 << 30
    if base.shape[-1] + n_new + warmup > limit:
        raise ValueError(
            f"prompt ({base.shape[-1]}) + {n_new} new tokens exceeds this model's "
            f"{limit}-token limit — shorten prompt_tokens or n_new")

    def run(steps, collect):
        ids = base.clone()
        past, cur = None, ids
        rows = []
        for step in range(steps):
            t0 = time.perf_counter()
            with torch.inference_mode():
                if use_cache:
                    out = model(cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                else:
                    # No cache: re-read the entire prefix every step. This is
                    # what a naive loop does, and it is O(n^2) over the sequence.
                    out = model(ids, use_cache=False)
            _sync(device)
            ms = (time.perf_counter() - t0) * 1000
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=-1)
            cur = next_id if use_cache else ids
            if collect:
                rows.append({"step": step, "context_len": ids.shape[-1], "ms": ms,
                             "cache": "with KV cache" if use_cache else "no KV cache"})
        return rows

    run(warmup, collect=False)      # same code path, results discarded
    return run(n_new, collect=True)


def compare_decode_curves(model, tok, prompt_tokens=512, n_new=48, warmup=4) -> dict:
    """Run both settings, each with its own warm-up, and summarise honestly.

    Reports the median rather than the first and last samples: a single first
    sample is the one most contaminated by whatever the allocator was doing, and
    quoting `last / first` as "growth" turns one noisy reading into a headline.
    """
    import statistics

    rows = (decode_curve(model, tok, prompt_tokens=prompt_tokens, n_new=n_new,
                         use_cache=True, warmup=warmup)
            + decode_curve(model, tok, prompt_tokens=prompt_tokens, n_new=n_new,
                           use_cache=False, warmup=warmup))
    by = {}
    for r in rows:
        by.setdefault(r["cache"], []).append(r)

    summary = {"rows": rows, "prompt_tokens": prompt_tokens, "n_new": n_new}
    for label, rs in by.items():
        times = [r["ms"] for r in rs]
        summary[label] = {
            "median_ms": statistics.median(times),
            "first_ms": times[0],
            "last_ms": times[-1],
            "context_first": rs[0]["context_len"],
            "context_last": rs[-1]["context_len"],
        }
    cached = summary.get("with KV cache")
    naive = summary.get("no KV cache")
    if cached and naive:
        summary["speedup_median"] = naive["median_ms"] / cached["median_ms"]
        summary["speedup_last"] = naive["last_ms"] / cached["last_ms"]
    return summary


def format_decode_comparison(summary) -> str:
    cached, naive = summary["with KV cache"], summary["no KV cache"]
    lines = [
        f"prompt {summary['prompt_tokens']} tokens, {summary['n_new']} decode steps, "
        f"context {cached['context_first']} -> {cached['context_last']}",
        "",
        f"{'':<16}{'median':>11}{'first':>11}{'last':>11}",
        f"{'with KV cache':<16}{cached['median_ms']:>10.1f}{'ms':<1}"
        f"{cached['first_ms']:>10.1f}{'ms':<1}{cached['last_ms']:>10.1f}ms",
        f"{'no KV cache':<16}{naive['median_ms']:>10.1f}{'ms':<1}"
        f"{naive['first_ms']:>10.1f}{'ms':<1}{naive['last_ms']:>10.1f}ms",
        "",
        f"speedup from caching: {summary['speedup_median']:.1f}x at the median, "
        f"{summary['speedup_last']:.1f}x at the last token",
    ]
    if summary["speedup_median"] < 1.5:
        lines += ["", "!! Under 1.5x means the measurement is not showing the effect.",
                  "   Almost always: the prompt is too short, so both paths are dominated",
                  "   by launch overhead rather than by how much prefix is re-read."]
    return "\n".join(lines)


def iter_kv_layers(past_key_values):
    """Yield `(keys, values)` per layer, whatever cache layout you were handed.

    HuggingFace has changed this representation three times, and every change
    silently breaks code that assumed the previous one:

    * transformers 5.x — a `Cache` whose `.layers[i]` have `.keys` / `.values`
    * transformers 4.36-4.5x — a `Cache` with parallel `.key_cache` / `.value_cache`
    * before that — a plain tuple of per-layer tuples

    Iterating a modern `Cache` directly yields 3-tuples, so the old
    `for k, v in past_key_values` unpacks with "too many values". Dispatch on
    the shape rather than assuming one.
    """
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:                                  # transformers 5.x
        for layer in layers:
            k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
            if k is not None and v is not None and not callable(k):
                yield k, v
        return

    keys = getattr(past_key_values, "key_cache", None)
    values = getattr(past_key_values, "value_cache", None)
    if keys is not None and values is not None:             # transformers 4.36-4.5x
        yield from zip(keys, values)
        return

    for entry in past_key_values:                           # legacy tuples
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            yield entry[0], entry[1]


def measured_kv_bytes(past_key_values) -> int:
    """Actual bytes held by a cache — compare against `kv_bytes_per_token`
    times context length. They should agree to within a few percent; if they do
    not, your head_dim or layer count is wrong."""
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in iter_kv_layers(past_key_values))


class BatchedEngine:
    """Continuous batching over GPT-2, scheduled by `servlab.toy.Scheduler`."""

    def __init__(self, model, tok, kv_budget_bytes=256 * 1024 * 1024, block_size=16, config=None):
        self.model = model
        self.tok = tok
        self.device = next(model.parameters()).device
        self.spec = spec_from_hf(model)
        # Size the pool from the same formula the napkin sheet uses.
        per_token = kv_bytes_per_token(self.spec, next(model.parameters()).element_size())
        num_blocks = max(4, int(kv_budget_bytes / (per_token * block_size)))
        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        self.scheduler = Scheduler(self.allocator, config or SchedulerConfig(max_num_seqs=16))
        self.caches = {}
        self.tokens = {}
        self.rows = []
        self.kv_bytes_per_token = per_token
        print(f"KV pool: {num_blocks} blocks x {block_size} tokens = "
              f"{num_blocks * block_size:,} tokens ({kv_budget_bytes / 1024**2:.0f} MiB, "
              f"{per_token / 1024:.1f} KiB/token)")

    def submit(self, prompt, max_tokens=64, req_id=None, arrival=0.0):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        rid = req_id or f"r{len(self.tokens)}"
        self.tokens[rid] = ids.to(self.device)
        return self.scheduler.add(Request(id=rid, prompt_len=ids.shape[-1],
                                          max_tokens=max_tokens, arrival=arrival))

    def run(self, max_steps=10_000, on_step=None):
        """Drive to completion, recording a dashboard row per step."""
        import torch

        t0 = time.perf_counter()
        for _ in range(max_steps):
            if self.scheduler.empty:
                break
            now = time.perf_counter() - t0
            out = self.scheduler.step(now=now)

            with torch.inference_mode():
                for req in out.prefilled:
                    res = self.model(self.tokens[req.id], use_cache=True)
                    self.caches[req.id] = res.past_key_values
                    self._append(req, res)
                for req in out.decoded:
                    if req in out.prefilled:
                        continue
                    last = self.tokens[req.id][:, -1:]
                    res = self.model(last, past_key_values=self.caches[req.id], use_cache=True)
                    self.caches[req.id] = res.past_key_values
                    self._append(req, res)
            # A preempted request loses its cache — that is what "recompute"
            # costs, and why preemption storms destroy throughput.
            for req in out.preempted:
                self.caches.pop(req.id, None)
                self.tokens[req.id] = self.tokens[req.id][:, :req.prompt_len]
            for req in out.finished:
                self.caches.pop(req.id, None)

            row = self.scheduler.stats(t=now)
            self.rows.append(row)
            if on_step:
                on_step(row, out)
        return self.rows

    def _append(self, req, res):
        import torch

        next_id = res.logits[:, -1, :].argmax(-1, keepdim=True)
        self.tokens[req.id] = torch.cat([self.tokens[req.id], next_id], dim=-1)

    def text(self, req_id) -> str:
        return self.tok.decode(self.tokens[req_id][0], skip_special_tokens=True)
