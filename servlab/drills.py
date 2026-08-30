"""Randomised whiteboard drills, with the worked answer.

Memorising one worked example teaches you that example. These generate fresh
numbers each time, so what you rehearse is the *method* — and the worked answer
shows the arithmetic the way you would say it out loud, not just the result.

Usage in a notebook:

    d = drills.kv_cache()      # prints the question
    ...  work it on paper, out loud, under three minutes ...
    d.reveal()

The target is not speed for its own sake. It is being able to narrate while
computing, because that is the actual test: an interviewer is listening to your
reasoning, and silence while you do arithmetic reads as uncertainty.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import napkin as nk
from . import pricing as pr

GIB = nk.GIB


@dataclass
class Drill:
    title: str
    question: str
    answer: str = ""
    steps: list = field(default_factory=list)
    watch_for: str = ""
    target_seconds: int = 180
    derivation: object = None      # a Derivation or Worksheet: the worked answer

    def __post_init__(self):
        print(f"[{self.title}]  target: {self.target_seconds // 60} min, narrated\n")
        print(self.question)
        print("\n(work it on paper, out loud, then call .reveal())")

    def reveal(self):
        if self.derivation is not None:
            print(self.derivation)
            if self.watch_for:
                print(f"\n  Watch for: {self.watch_for}\n")
            return self
        print("=" * 66)
        for i, step in enumerate(self.steps, 1):
            head, *rest = str(step).split("\n")
            print(f"  {i}. {head}")
            for line in rest:                    # keep multi-line steps aligned
                print(f"     {line}")
        print("-" * 66)
        print(f"  ANSWER: {self.answer}")
        if self.watch_for:
            print(f"\n  Watch for: {self.watch_for}")
        print("=" * 66)
        return self


def _rng(seed):
    return random.Random(seed if seed is not None else random.randrange(10**9))


# --------------------------------------------------------------------------


def kv_cache(seed=None) -> Drill:
    """Drill 1: KV cache per token, and what it means for one long request."""
    r = _rng(seed)
    layers = r.choice([32, 48, 60, 80, 96])
    kv_heads = r.choice([4, 8, 8, 16])
    head_dim = r.choice([64, 128, 128])
    q_heads = kv_heads * r.choice([2, 4, 4, 8])
    bits = r.choice(["fp16", "fp16", "fp8"])
    ctx = r.choice([8192, 16384, 32768, 131072])

    spec = nk.ModelSpec("drill", layers, q_heads, kv_heads, head_dim,
                        q_heads * head_dim, 4 * q_heads * head_dim, 128256, 0)
    per_tok = nk.kv_bytes_per_token(spec, bits)
    total = per_tok * ctx
    b = nk.dtype_bytes(bits)

    drill = Drill(
        "Drill 1 — KV cache sizing",
        f"A model has {layers} layers, {q_heads} attention heads over {kv_heads} KV heads, "
        f"head_dim {head_dim}, KV cache in {bits}.\n\n"
        f"  a) KV bytes per token?\n"
        f"  b) How much KV does one {ctx:,}-token request hold?\n"
        f"  c) How many such requests fit in 40 GiB of KV budget?",
        answer=f"{nk.human_bytes(per_tok)}/token · {nk.human_bytes(total)} per request · "
               f"{40 * GIB / total:.1f} requests in 40 GiB",
        steps=[
            f"2 (K and V) x {layers} layers x {kv_heads} KV heads x {head_dim} dim x {b:g} bytes",
            f"= {per_tok:,.0f} B = {nk.human_bytes(per_tok)} per token",
            f"x {ctx:,} tokens = {nk.human_bytes(total)} for one request",
            f"40 GiB / {nk.human_bytes(total)} = {40 * GIB / total:.1f} concurrent requests",
        ],
        watch_for="KV heads, not attention heads — that is the GQA discount, and using "
                  f"{q_heads} instead of {kv_heads} would overstate this by {q_heads // kv_heads}x.",
    )
    drill.derivation = nk.derive_kv_per_token(layers, kv_heads, head_dim, bits, ctx_check=ctx)
    return drill


def decode_ceiling(seed=None) -> Drill:
    """Drill 2: decode throughput from bandwidth — given the numbers, not the card."""
    r = _rng(seed)
    params = r.choice([7e9, 13e9, 34e9, 70e9, 141e9])
    bits = r.choice([16, 8, 8, 4])
    bw = r.choice([320, 900, 1555, 3350, 4800])
    vram = r.choice([24, 40, 80, 141])
    batch = r.choice([1, 8, 32])
    ctx = r.choice([2048, 8192])
    layers, kv_heads, head_dim = r.choice([(32, 8, 128), (60, 8, 128), (80, 8, 128), (48, 4, 128)])
    kv_per_tok = 2 * layers * kv_heads * head_dim * 2

    drill = Drill(
        "Drill 2 — decode throughput",
        f"You are handed these numbers and nothing else.\n\n"
        f"  model:  {params/1e9:.0f}B parameters, {layers} layers, {kv_heads} KV heads, "
        f"head_dim {head_dim}\n"
        f"  serving: {bits}-bit weights, batch {batch}, {ctx:,}-token contexts\n"
        f"  card:   {vram} GB, {bw:,} GB/s memory bandwidth\n\n"
        f"  a) Aggregate output tokens/s?\n"
        f"  b) Per sequence — would a user find that fast enough?\n"
        f"  c) Does more batch still help here, or has KV taken over?",
        watch_for="Weights are read once for the whole batch; KV is read per sequence. "
                  "Which term dominates is the entire batching argument, and it flips as "
                  "context grows.",
    )
    drill.derivation = nk.derive_decode_speed(params, bw, batch, ctx, kv_per_tok, bits)
    return drill


def ttft(seed=None) -> Drill:
    """Drill 3: prefill time from FLOPs — the one place FLOPs bind."""
    r = _rng(seed)
    prompt = r.choice([512, 1024, 2048, 4096, 8192])
    params = r.choice([3e9, 8e9, 34e9, 70e9])
    tflops = r.choice([65, 121, 312, 989])
    tp = r.choice([1, 1, 2, 4])
    mfu = r.choice([0.3, 0.4, 0.5])
    sla_ms = r.choice([300, 400, 1000])

    drill = Drill(
        "Drill 3 — time to first token",
        f"  model:  {params/1e9:.0f}B parameters\n"
        f"  prompt: {prompt:,} tokens\n"
        f"  card:   {tflops:,} TFLOPS at your serving dtype, {tp}-way tensor parallel\n"
        f"  assume: {mfu:.0%} model FLOPs utilisation\n\n"
        f"  a) Estimate TTFT.\n"
        f"  b) The SLA is {sla_ms} ms. Do you meet it, and if not what do you change first?",
        watch_for="Prefill is compute bound — the only place peak FLOPs matter. Decode is "
                  "bandwidth bound. Say which regime you are in before you reach for a number.",
    )
    drill.derivation = nk.derive_prefill_time(params, prompt, tflops, mfu, n_gpus=tp)
    return drill


def capacity(seed=None) -> Drill:
    """Drill 4: an architecture and a card. Derive the deployment."""
    r = _rng(seed)
    layers = r.choice([28, 32, 48, 60, 80])
    kv_heads = r.choice([2, 4, 8, 8])
    q_heads = kv_heads * r.choice([2, 4, 4, 8])
    head_dim = r.choice([64, 128, 128])
    hidden = q_heads * head_dim
    params = r.choice([3e9, 8e9, 34e9, 70e9])
    vram = r.choice([16, 24, 40, 80])
    bw = r.choice([320, 600, 1555, 3350])
    bits = r.choice([16, 16, 8, 4])
    ctx = r.choice([2048, 4096, 8192, 32768])

    drill = Drill(
        "Drill 4 — size a deployment from a config",
        f"A config.json and a spec sheet. Nothing is looked up.\n\n"
        f"  num_hidden_layers      {layers}\n"
        f"  num_attention_heads    {q_heads}\n"
        f"  num_key_value_heads    {kv_heads}\n"
        f"  hidden_size            {hidden}\n"
        f"  head_dim               {head_dim}\n"
        f"  parameters             {params/1e9:.0f}B, served at {bits}-bit\n\n"
        f"  card: {vram} GB, {bw:,} GB/s\n"
        f"  you must serve {ctx:,}-token contexts\n\n"
        f"  a) KV bytes per token?\n"
        f"  b) How many concurrent sequences fit?\n"
        f"  c) What is the single cheapest change that doubles (b)?",
        watch_for="num_attention_heads is in the config to catch you. The cache scales with "
                  "num_key_value_heads.",
        target_seconds=300,
    )
    drill.derivation = nk.worksheet(
        name=f"{params/1e9:.0f}B on a {vram} GB card",
        n_layers=layers, n_kv_heads=kv_heads, head_dim=head_dim, params=params,
        vram_gb=vram, mem_bw_gb_s=bw, ctx=ctx, batch=16, weight_bits=bits)
    return drill


def cost(seed=None) -> Drill:
    """Drill 5: per-chat economics and the routing decision."""
    r = _rng(seed)
    chats = r.choice([1e6, 5e6, 10e6, 50e6])
    tin = r.choice([500, 1000, 2000, 4000])
    tout = r.choice([150, 300, 600])
    strong, cheap = "gemini-3.1-pro", "gemini-3.5-flash-lite"
    rows = pr.compare([strong, "gemini-3.6-flash", cheap], tin, tout, requests_per_month=chats)
    plan = pr.RoutingPlan(cheap, strong, escalation_rate=0.15)
    blended = plan.cost_per_call(tin, tout) * chats
    all_strong = pr.call_cost(strong, tin, tout) * chats

    return Drill(
        "Drill 5 — serving cost, live",
        f"{chats / 1e6:.0f}M chats/month, {tin:,} input tokens and {tout} output tokens each.\n\n"
        f"  a) Monthly cost on a frontier model vs the cheap tier?\n"
        f"  b) What does routing 85% to the cheap tier cost?\n"
        f"  c) When would you not bother?",
        answer=f"${all_strong:,.0f}/mo all-frontier vs ${blended:,.0f}/mo routed "
               f"({blended / all_strong:.0%})",
        steps=[
            pr.format_compare(rows),
            f"routed: cheap for all + strong for 15% = ${blended:,.0f}/mo "
            f"({blended / all_strong:.0%} of all-frontier)",
            f"break-even escalation rate is "
            f"{pr.breakeven_escalation(cheap, strong, tin, tout):.0%} — past that, routing "
            "costs more than just using the strong model",
            "skip routing if the bill is small enough that engineering time dominates, or "
            "if quality variance is unacceptable for the use case",
        ],
        watch_for="Count the cheap call you paid for before escalating. Ignoring it is the "
                  "usual reason a routing estimate comes in optimistic.",
        target_seconds=240,
    )


def oom(seed=None) -> Drill:
    """Drill 6: budget a fine-tune and find the term that will kill it."""
    r = _rng(seed)
    from .memory import training_budget

    params = r.choice([7.6e9, 8.03e9, 70.6e9])
    seq = r.choice([1024, 2048, 4096, 8192])
    batch = r.choice([1, 2, 4])
    vocab = 128_256
    layers, hidden = (80, 8192) if params > 50e9 else (32, 4096)
    b = training_budget(params, weight_bits=4, trainable_params=1.6e7, optimizer="adamw8bit",
                        batch=batch, seq_len=seq, n_layers=layers, hidden=hidden,
                        activation_checkpointing=False, vocab=vocab)
    card = 80.0

    return Drill(
        "Drill 6 — will this QLoRA run OOM?",
        f"QLoRA on a {params / 1e9:.0f}B model: 4-bit base, LoRA r=16, paged 8-bit AdamW, "
        f"batch {batch} x {seq:,} tokens, {vocab:,} vocab, no gradient checkpointing, "
        f"on an {card:.0f} GB card.\n\n"
        f"  a) Budget it term by term.\n"
        f"  b) Which term kills it, and what is the fix order?",
        answer=f"~{b.total / GIB:.1f} GiB predicted vs {card:.0f} GB card — "
               f"{'OOM' if b.total / GIB > card else 'fits, with little room'}",
        steps=[
            str(b),
            "QLoRA collapses weights, gradients and optimizer — it does nothing for "
            "activations, which scale with batch x seq and are the usual killer",
            "logits are the term nobody budgets: batch x seq x vocab in fp32, plus the "
            "copy the loss makes",
            "fix order: gradient checkpointing -> shorter seq -> micro-batch 1 with "
            "accumulation -> fused cross-entropy -> paged optimizer -> lower rank",
        ],
        watch_for="If it dies at step 40 rather than step 0, suspect the eval loop's batch "
                  "size, the longest-sample bucket arriving, or allocator fragmentation.",
    )


ALL = [kv_cache, decode_ceiling, ttft, capacity, cost, oom]


def random_drill(seed=None) -> Drill:
    return _rng(seed).choice(ALL)(seed=seed)


def drill_set(n=3, seed=None) -> list:
    """A mixed set, for a timed session."""
    r = _rng(seed)
    picks = r.sample(ALL, min(n, len(ALL)))
    return [p(seed=r.randrange(10**9)) for p in picks]


# --------------------------------------------------------------------------
# Numbers you should not have to derive
# --------------------------------------------------------------------------

FLASHCARDS = [
    ("KV cache bytes per token", "2 x layers x kv_heads x head_dim x bytes"),
    ("Llama-3.3-70B KV per token (fp16)", "0.32 MB — so 32K context is ~10 GB"),
    ("Prefill FLOPs", "~2 x params x prompt_tokens"),
    ("Realistic MFU for prefill", "0.3-0.5"),
    ("Decode ceiling, single stream", "HBM bandwidth / weight bytes (then x0.6-0.7)"),
    ("H100 SXM", "80 GB, 3.35 TB/s, ~1 PFLOP BF16 dense"),
    ("H200", "141 GB, 4.8 TB/s, same compute as H100"),
    ("B200", "~192 GB, ~8 TB/s, ~2x H100-class compute"),
    ("H100 ridge point", "~295 FLOP/byte — decode is O(1), so always bandwidth bound"),
    ("Full fine-tune memory", "~16 bytes/param (fp16 w+g, fp32 master + Adam m,v) + activations"),
    ("Tokens per word", "~1.33 (1 token ~ 0.75 words ~ 4 chars)"),
    ("Batch API discount", "~50%"),
    ("Cache read price", "~10% of input rate"),
    ("Thinking tokens", "billed as output — 5-6x the input rate"),
    ("Seconds in a day", "86,400"),
    ("TTFT that feels instant", "<300 ms; <1 s is fine for chat"),
    ("Human reading speed", "20-40 tok/s"),
    ("Voice mouth-to-ear budget", "<800 ms"),
    ("Reserved GPU rental", "~$2-3/GPU-hr; $4-11 on-demand at hyperscalers"),
    ("When self-hosting starts to pay", "sustained spend past ~3x the fully-loaded self-host cost"),
    ("Speculative decoding win", "1.5-2.5x TPOT at high acceptance; collapses below ~0.6"),
    ("FP8 KV cache", "halves KV bytes -> doubles concurrency; needs Ada/Hopper"),
]


def flashcards(n=8, seed=None, reveal=False):
    """Quiz yourself on the constants. Run it until nothing is a surprise."""
    r = _rng(seed)
    picks = r.sample(FLASHCARDS, min(n, len(FLASHCARDS)))
    for i, (q, a) in enumerate(picks, 1):
        print(f"{i:>2}. {q}")
        if reveal:
            print(f"    -> {a}\n")
    if not reveal:
        print("\n(rerun with reveal=True)")
    return picks
