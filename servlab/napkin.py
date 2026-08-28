"""Napkin math for inference serving.

Everything here is a closed-form estimate you can do on paper. The point of the
labs is to *predict with these first*, then measure, then explain the gap. A
prediction that lands within 2x is a working mental model; one that is off by
10x means you are missing a term.

No torch, no GPU, no network — this module is pure arithmetic so it can be
tested in CI and used to plan a lab before you rent anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 1024**3
MIB = 1024**2


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """The handful of numbers that determine memory and bandwidth behaviour.

    Read them off `config.json`: `num_hidden_layers`, `num_attention_heads`,
    `num_key_value_heads`, `hidden_size`, `intermediate_size`, `vocab_size`.
    `head_dim` is `hidden_size / num_attention_heads` unless the config says
    otherwise (some models, e.g. Llama-3.2-1B, set it explicitly).
    """

    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden: int
    ffn: int
    vocab: int
    params: float

    @property
    def gqa_ratio(self) -> float:
        """Query heads per KV head. This is the KV-cache discount factor."""
        return self.n_heads / self.n_kv_heads

    @property
    def kv_dim(self) -> int:
        """Width of one K (or V) vector per layer."""
        return self.n_kv_heads * self.head_dim


@dataclass(frozen=True)
class GPUSpec:
    """Accelerator envelope.

    `usd_per_hour` is a rough on-demand figure for lab-2 cost math — it is a
    placeholder, not a quote. Edit it to whatever you are actually paying;
    every cost number downstream is only as good as this field.
    """

    name: str
    vram_gb: float
    mem_bw_gb_s: float
    fp16_tflops: float
    capability: tuple  # CUDA compute capability, e.g. (7, 5) for T4
    usd_per_hour: float = 0.0

    @property
    def supports_bf16(self) -> bool:
        # bf16 arrives with Ampere (sm_80). Turing (T4, sm_75) does not have it,
        # which is why vLLM needs `--dtype half` there.
        return self.capability >= (8, 0)

    @property
    def supports_fp8(self) -> bool:
        # FP8 tensor cores land on Hopper (sm_90); Ada (sm_89) has FP8 storage.
        return self.capability >= (8, 9)


# Configs below are read from each model's published config.json.
MODELS = {
    "gpt2": ModelSpec("gpt2", 12, 12, 12, 64, 768, 3072, 50257, 124e6),
    "gpt2-medium": ModelSpec("gpt2-medium", 24, 16, 16, 64, 1024, 4096, 50257, 355e6),
    "tinyllama-1.1b": ModelSpec("tinyllama-1.1b", 22, 32, 4, 64, 2048, 5632, 32000, 1.1e9),
    "llama-3.2-1b": ModelSpec("llama-3.2-1b", 16, 32, 8, 64, 2048, 8192, 128256, 1.24e9),
    "llama-3.2-3b": ModelSpec("llama-3.2-3b", 28, 24, 8, 128, 3072, 8192, 128256, 3.21e9),
    "llama-3.1-8b": ModelSpec("llama-3.1-8b", 32, 32, 8, 128, 4096, 14336, 128256, 8.03e9),
    "qwen2.5-3b": ModelSpec("qwen2.5-3b", 36, 16, 2, 128, 2048, 11008, 151936, 3.09e9),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", 28, 28, 4, 128, 3584, 18944, 152064, 7.62e9),
    "mistral-7b": ModelSpec("mistral-7b", 32, 32, 8, 128, 4096, 14336, 32768, 7.25e9),
    # 80 layers, 64 query heads over 8 KV heads -> 0.32 MiB/token in fp16.
    # Worth knowing cold: it is the model most napkin questions are posed about.
    "llama-3.3-70b": ModelSpec("llama-3.3-70b", 80, 64, 8, 128, 8192, 28672, 128256, 70.6e9),
    "llama-3.1-405b": ModelSpec("llama-3.1-405b", 126, 128, 8, 128, 16384, 53248, 128256, 405.9e9),
}

GPUS = {
    "T4": GPUSpec("T4", 16, 320, 65, (7, 5), 0.35),
    "L4": GPUSpec("L4", 24, 300, 121, (8, 9), 0.80),
    "A10G": GPUSpec("A10G", 24, 600, 125, (8, 6), 1.00),
    "A100-40GB": GPUSpec("A100-40GB", 40, 1555, 312, (8, 0), 1.80),
    "A100-80GB": GPUSpec("A100-80GB", 80, 2039, 312, (8, 0), 2.50),
    "H100-80GB": GPUSpec("H100-80GB", 80, 3350, 989, (9, 0), 3.50),
    "H200": GPUSpec("H200", 141, 4800, 989, (9, 0), 4.00),
    "B200": GPUSpec("B200", 192, 8000, 2250, (10, 0), 6.00),
    # TPUs have no CUDA compute capability; (9, 9) is a stand-in that keeps the
    # bf16/fp8 predicates true. Read the FLOPs as the announced FP8 figure.
    "TPU-v7": GPUSpec("TPU-v7", 192, 7400, 4614, (9, 9), 0.0),
}

# Bytes per element, by the name you would pass to a serving stack.
DTYPE_BYTES = {"fp32": 4, "float": 4, "fp16": 2, "half": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5, "awq": 0.5, "gptq": 0.5}


def dtype_bytes(dtype) -> float:
    if isinstance(dtype, (int, float)):
        return float(dtype)
    try:
        return float(DTYPE_BYTES[str(dtype).lower()])
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}") from exc


def model(name) -> ModelSpec:
    return name if isinstance(name, ModelSpec) else MODELS[str(name).lower()]


def gpu(name) -> GPUSpec:
    return name if isinstance(name, GPUSpec) else GPUS[str(name)]


def spec_from_config(cfg, name=None, gated_mlp=True, tied_embeddings=None) -> ModelSpec:
    """Build a `ModelSpec` from a HuggingFace config (or a plain dict).

    Lets a notebook do the napkin math for whatever model it just pulled,
    without downloading the weights first. The parameter count is estimated
    from the shapes — embeddings, attention projections, MLP — which lands
    within a couple of percent of the real count for standard decoder stacks.
    """
    get = (lambda k, d=None: getattr(cfg, k, d)) if not isinstance(cfg, dict) else (lambda k, d=None: cfg.get(k, d))
    n_heads = get("num_attention_heads") or get("n_head")
    n_layers = get("num_hidden_layers") or get("n_layer")
    hidden = get("hidden_size") or get("n_embd")
    n_kv = get("num_key_value_heads", n_heads) or n_heads
    head_dim = get("head_dim") or hidden // n_heads
    ffn = get("intermediate_size") or 4 * hidden
    vocab = get("vocab_size")
    if tied_embeddings is None:
        tied_embeddings = bool(get("tie_word_embeddings", False))

    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    attn = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    mlp = (3 if gated_mlp else 2) * hidden * ffn
    params = vocab * hidden * (1 if tied_embeddings else 2) + n_layers * (attn + mlp)
    return ModelSpec(
        name=name or str(get("_name_or_path", "model")),
        n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
        hidden=hidden, ffn=ffn, vocab=vocab, params=float(params),
    )


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def weight_bytes(spec, dtype="fp16") -> float:
    """Bytes the parameters occupy. Quantised weights still carry scales and
    zero-points, so a 4-bit checkpoint on disk is nearer 4.5 bits/param — see
    `awq_weight_bytes` for the honest version."""
    return model(spec).params * dtype_bytes(dtype)


def awq_weight_bytes(spec, w_bits=4, group_size=128) -> float:
    """4-bit weights + fp16 scale and 4-bit zero-point per group of `group_size`."""
    p = model(spec).params
    per_param_bits = w_bits + (16 + w_bits) / group_size
    return p * per_param_bits / 8


def kv_bytes_per_token(spec, dtype="fp16") -> float:
    """The single most useful number in serving.

        2 (K and V) x layers x kv_heads x head_dim x bytes

    Multiply by sequence length for one request; by total tokens in flight for
    the whole server. Everything about capacity follows from this.
    """
    s = model(spec)
    return 2 * s.n_layers * s.kv_dim * dtype_bytes(dtype)


def kv_bytes(spec, seq_len, batch=1, dtype="fp16") -> float:
    return kv_bytes_per_token(spec, dtype) * seq_len * batch


def kv_budget_bytes(gpu_spec, spec, weight_dtype="fp16", util=0.90, activation_gb=1.0) -> float:
    """VRAM left for KV cache after weights, activations, and the slack a
    serving stack keeps back.

    `util` mirrors vLLM's `--gpu-memory-utilization`: the fraction of the card
    the engine is allowed to claim at all. `activation_gb` covers CUDA context,
    activations, and the graph pool — about 1 GiB for a small model, more if you
    raise `--max-num-batched-tokens`.
    """
    g = gpu(gpu_spec)
    total = g.vram_gb * GIB * util
    free = total - weight_bytes(spec, weight_dtype) - activation_gb * GIB
    return max(free, 0.0)


def kv_capacity_tokens(gpu_spec, spec, weight_dtype="fp16", kv_dtype="fp16", util=0.90, activation_gb=1.0) -> float:
    """How many tokens of KV the card can hold at once, across all sequences."""
    return kv_budget_bytes(gpu_spec, spec, weight_dtype, util, activation_gb) / kv_bytes_per_token(spec, kv_dtype)


def max_concurrent_sequences(gpu_spec, spec, seq_len, **kw) -> float:
    """Concurrency ceiling at a fixed context length.

    Below this the server queues; above it, it preempts. This is the number the
    death-spiral in lab 1 is crossing.
    """
    return kv_capacity_tokens(gpu_spec, spec, **kw) / seq_len


def fits(gpu_spec, spec, weight_dtype="fp16", seq_len=2048, concurrency=1, **kw) -> bool:
    return kv_capacity_tokens(gpu_spec, spec, weight_dtype=weight_dtype, **kw) >= seq_len * concurrency


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


def prefill_flops(spec, n_tokens) -> float:
    """~2 FLOPs per parameter per token for the matmuls (attention's quadratic
    term is small until context gets long, and is ignored here on purpose —
    it is one of the gaps worth explaining when the measurement disagrees)."""
    return 2 * model(spec).params * n_tokens


def prefill_time_s(gpu_spec, spec, n_tokens, mfu=0.4) -> float:
    """Prefill is compute bound: it is a big matmul, so it lands somewhere near
    a real fraction (`mfu`) of peak FLOPs."""
    g = gpu(gpu_spec)
    return prefill_flops(spec, n_tokens) / (g.fp16_tflops * 1e12 * mfu)


def decode_bytes_per_step(spec, batch, ctx_len, weight_dtype="fp16", kv_dtype="fp16") -> float:
    """Bytes that must cross the memory bus for one decode step.

    Weights are read once for the whole batch — that is why batching is nearly
    free until the KV term catches up.
    """
    return weight_bytes(spec, weight_dtype) + kv_bytes(spec, ctx_len, batch, kv_dtype)


def decode_step_time_s(gpu_spec, spec, batch=1, ctx_len=1024, weight_dtype="fp16", kv_dtype="fp16", bw_efficiency=0.7) -> float:
    """Decode is memory bound: one token per sequence, so time ~= bytes / bandwidth.

    `bw_efficiency` is the fraction of spec-sheet bandwidth you actually get
    (0.6-0.8 is typical). If your measurement needs an efficiency above 1.0 to
    fit, your model of what is being read is wrong.
    """
    g = gpu(gpu_spec)
    b = decode_bytes_per_step(spec, batch, ctx_len, weight_dtype, kv_dtype)
    return b / (g.mem_bw_gb_s * 1e9 * bw_efficiency)


def decode_tokens_per_s(gpu_spec, spec, batch=1, ctx_len=1024, **kw) -> float:
    """Aggregate output tokens/s across the batch."""
    return batch / decode_step_time_s(gpu_spec, spec, batch, ctx_len, **kw)


def decode_step_time_tp_s(gpu_spec, spec, batch=1, ctx_len=1024, tp=1, weight_dtype="fp16",
                          kv_dtype="fp16", bw_efficiency=0.7, allreduce_us_per_layer=6.0):
    """Decode step time under tensor parallelism.

    Each GPU holds 1/tp of the weights and 1/tp of the KV heads, so the memory
    term divides cleanly. What does not divide is the all-reduce after every
    attention and MLP block — two per layer — and that term is why TP buys
    latency at the cost of efficiency, and why it stops scaling once you leave
    NVLink.

    `allreduce_us_per_layer` is the part to sanity-check against a measurement:
    on NVLink it is single-digit microseconds; across PCIe or a network it can
    be an order of magnitude worse, at which point TP is a trap.
    """
    if tp < 1:
        raise ValueError("tp must be >= 1")
    g, m = gpu(gpu_spec), model(spec)
    per_gpu_bytes = decode_bytes_per_step(m, batch, ctx_len, weight_dtype, kv_dtype) / tp
    mem_s = per_gpu_bytes / (g.mem_bw_gb_s * 1e9 * bw_efficiency)
    comm_s = 0.0 if tp == 1 else m.n_layers * 2 * allreduce_us_per_layer * 1e-6
    return mem_s + comm_s


def tp_scaling(gpu_spec, spec, batch=1, ctx_len=1024, tps=(1, 2, 4, 8), **kw):
    """Latency and efficiency against TP degree — the table to reason from.

    `efficiency` is speedup / tp: 1.0 would be perfect scaling. When it falls
    below ~0.6 you are paying for two GPUs and getting one and a bit, which is
    a fine trade for a latency SLA and a bad one for a throughput fleet.
    """
    base = decode_step_time_tp_s(gpu_spec, spec, batch, ctx_len, tp=1, **kw)
    rows = []
    for tp in tps:
        t = decode_step_time_tp_s(gpu_spec, spec, batch, ctx_len, tp=tp, **kw)
        rows.append({"tp": tp, "step_s": t, "speedup": base / t,
                     "efficiency": (base / t) / tp, "tok_s": batch / t})
    return rows


def fits_with_tp(gpu_spec, spec, weight_dtype="fp16", tp=1, seq_len=2048, concurrency=1, **kw):
    """Does it fit across `tp` cards? Weights and KV both shard."""
    g = gpu(gpu_spec)
    sharded = GPUSpec(g.name, g.vram_gb * tp, g.mem_bw_gb_s, g.fp16_tflops, g.capability, g.usd_per_hour)
    return fits(sharded, spec, weight_dtype, seq_len, concurrency, **kw)


def spec_decode_speedup(acceptance_rate, gamma=4, draft_cost_ratio=0.15, verify_overhead=0.1):
    """Expected TPOT speedup from speculative decoding.

    The draft proposes `gamma` tokens; the target verifies them in one forward
    pass. Expected accepted tokens per cycle for acceptance rate `a` is the
    truncated geometric mean `(1 - a^(gamma+1)) / (1 - a)`, and one cycle costs
    one target pass plus `gamma` draft passes.

    The two things to say out loud: it is a *latency* optimisation that
    consumes spare compute, so it degrades at high batch where there is none;
    and the win collapses fast below ~0.6 acceptance, which is why the draft
    model must match the target's distribution, not merely be small.
    """
    a = float(acceptance_rate)
    if not 0 <= a <= 1:
        raise ValueError("acceptance_rate must be in [0, 1]")
    accepted = gamma + 1 if a == 1 else (1 - a ** (gamma + 1)) / (1 - a)
    cycle_cost = 1 + verify_overhead + gamma * draft_cost_ratio
    return accepted / cycle_cost


def arithmetic_intensity(spec, batch, ctx_len=0, weight_dtype="fp16") -> float:
    """FLOPs per byte for a decode step. Compare against the GPU's ridge point
    (`ridge_point`): below it you are bandwidth bound, above it compute bound."""
    flops = 2 * model(spec).params * batch
    return flops / decode_bytes_per_step(spec, batch, ctx_len, weight_dtype)


def ridge_point(gpu_spec) -> float:
    """FLOPs/byte where a GPU stops being memory bound. T4 is ~203; that is why
    a batch of 1 leaves ~99% of the card idle."""
    g = gpu(gpu_spec)
    return g.fp16_tflops * 1e12 / (g.mem_bw_gb_s * 1e9)


# --------------------------------------------------------------------------
# Queueing
# --------------------------------------------------------------------------


def little_law_concurrency(rps, latency_s) -> float:
    """L = lambda W. Requests in flight, given arrival rate and residency time."""
    return rps * latency_s


def utilization(rps, service_time_s, servers=1) -> float:
    return rps * service_time_s / servers


def mm1_wait_s(rps, service_time_s):
    """Expected queue wait for M/M/1. Not a model of vLLM — a model of *why*
    the last 10% of utilisation costs you everything.

    Returns inf at or past saturation, which is the honest answer.
    """
    rho = utilization(rps, service_time_s)
    if rho >= 1:
        return math.inf
    return rho * service_time_s / (1 - rho)


def saturation_rps(service_time_s) -> float:
    return 1.0 / service_time_s


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def cost_per_million_tokens(gpu_spec, tokens_per_s) -> float:
    g = gpu(gpu_spec)
    if tokens_per_s <= 0:
        return math.inf
    return g.usd_per_hour / 3600 / tokens_per_s * 1e6


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def human_bytes(n) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TiB"  # pragma: no cover


def memory_report(gpu_spec, spec, weight_dtype="fp16", kv_dtype="fp16", seq_len=2048, util=0.90, activation_gb=1.0) -> str:
    """The block you paste into an interview answer."""
    g, s = gpu(gpu_spec), model(spec)
    w = weight_bytes(s, weight_dtype)
    per_tok = kv_bytes_per_token(s, kv_dtype)
    budget = kv_budget_bytes(g, s, weight_dtype, util, activation_gb)
    seqs = max_concurrent_sequences(g, s, seq_len, weight_dtype=weight_dtype, kv_dtype=kv_dtype, util=util, activation_gb=activation_gb)
    lines = [
        f"{s.name} @ {weight_dtype} on {g.name} ({g.vram_gb:g} GB, util={util:.0%})",
        f"  weights            {human_bytes(w)}",
        f"  activations/slack  {human_bytes(activation_gb * GIB)}",
        f"  KV budget          {human_bytes(budget)}",
        f"  KV per token       {human_bytes(per_tok)}  (GQA {s.gqa_ratio:g}:1, kv={kv_dtype})",
        f"  KV total capacity  {kv_capacity_tokens(g, s, weight_dtype, kv_dtype, util, activation_gb):,.0f} tokens",
        f"  concurrency @ {seq_len:,} ctx   {seqs:,.1f} sequences",
    ]
    if budget <= 0:
        lines.append("  !! weights alone do not fit — quantise, shard, or pick a smaller model")
    return "\n".join(lines)
