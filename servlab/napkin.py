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

from .derive import Derivation, Given, Worksheet

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
# Derivations — the same arithmetic, with the working shown
#
# Every function below takes *raw scalars*, not preset keys. That is deliberate.
# In a real sizing conversation nobody hands you `MODELS["llama-3.3-70b"]`; they
# hand you a config.json and a spec sheet, and the skill being tested is whether
# you can put those numbers in the right places. The presets in this module are
# for checking your answer afterwards, not for producing it.
# --------------------------------------------------------------------------

CONFIG_FIELDS = {
    "num_hidden_layers": "layers — the multiplier on everything in the KV cache",
    "num_attention_heads": "query heads — used for head_dim, NOT for the cache size",
    "num_key_value_heads": "KV heads — THIS is what the cache scales with (GQA). "
                           "Absent means it equals num_attention_heads (no GQA)",
    "head_dim": "head dimension, if stated. Otherwise hidden_size / num_attention_heads "
                "— several modern configs set it explicitly and it is not always the ratio",
    "hidden_size": "model width; feeds head_dim and the parameter estimate",
    "intermediate_size": "FFN width; parameter estimate only",
    "vocab_size": "embedding and output layer size; matters for training memory (logits)",
    "torch_dtype": "what the weights ship in — and the thing that breaks vLLM on a T4 "
                   "when it says bfloat16 and the card is Turing",
    "max_position_embeddings": "the context ceiling the weights support, not what you must serve",
}


def config_guide() -> str:
    """Which fields in a config.json you actually need, and the trap in each.

    Read this before deriving anything from a model you have not seen. The whole
    napkin exercise is four numbers out of a JSON file; knowing which four, and
    which one people take from the wrong field, is most of the skill.
    """
    lines = ["READING A config.json FOR SIZING", "=" * 74, ""]
    for k, v in CONFIG_FIELDS.items():
        lines.append(f"  {k}")
        lines.append(f"      {v}")
    lines += ["", "  The single most common error: using num_attention_heads where the",
              "  formula wants num_key_value_heads. On a Llama-3 that overstates the KV",
              "  cache by 4x, and every capacity number downstream inherits the error."]
    return "\n".join(lines)


def _bytes_per_param(bits):
    """Accept either a bit count (16, 8, 4) or a dtype name ("fp16", "awq")."""
    if isinstance(bits, str):
        return dtype_bytes(bits)
    return bits / 8


def from_config(cfg, weight_bits=16, kv_dtype="fp16", ctx=8192, params=None) -> Derivation:
    """Read a config.json and show which field fed which term.

    The realistic version of the exercise: you are handed a model you have never
    sized, you open its config, and you have to know which four numbers matter.
    Pass the parsed JSON (or a HuggingFace config object) and this prints the
    mapping *and* the derivation, so a wrong field is visible rather than baked
    into an answer.
    """
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    n_heads = get("num_attention_heads") or get("n_head")
    n_layers = get("num_hidden_layers") or get("n_layer")
    hidden = get("hidden_size") or get("n_embd")
    n_kv = get("num_key_value_heads") or n_heads
    stated_head_dim = get("head_dim")
    head_dim = stated_head_dim or (hidden // n_heads)

    d = derive_kv_per_token(n_layers, n_kv, head_dim, kv_dtype, ctx_check=ctx)
    d.title = f"KV cache per token — {get('_name_or_path', 'this config')}"
    d.givens.insert(0, Given("num_attention_heads", n_heads, "",
                             "read, but NOT used in this formula — it is the GQA trap"))
    if stated_head_dim:
        d.check(f"head_dim was stated explicitly as {stated_head_dim}; hidden_size/heads would "
                f"have given {hidden // n_heads} — always prefer the stated value")
    else:
        d.check(f"head_dim not stated, so hidden_size / num_attention_heads = "
                f"{hidden} / {n_heads} = {head_dim}")
    if n_kv == n_heads:
        d.check("num_key_value_heads equals num_attention_heads: no GQA, so the cache is as "
                "large as it gets for this shape")
    else:
        d.check(f"GQA {n_heads // n_kv}:1 — using num_attention_heads here would have "
                f"overstated the cache by {n_heads // n_kv}x")
    dtype = get("torch_dtype")
    if dtype and "bfloat16" in str(dtype):
        d.check("torch_dtype is bfloat16 — this is the field that makes vLLM refuse to "
                "start on a pre-Ampere card unless you pass --dtype half")
    return d


def derive_kv_per_token(n_layers, n_kv_heads, head_dim, dtype="fp16", ctx_check=32768):
    """KV bytes per token, shown.

    The one formula to be able to write from memory. Everything about serving
    capacity is downstream of it.
    """
    d = Derivation("KV cache per token",
                   formula="2 (K and V) x layers x kv_heads x head_dim x bytes_per_element")
    b = dtype_bytes(dtype)
    d.given("layers", n_layers, source="config.json: num_hidden_layers")
    d.given("kv heads", n_kv_heads,
            source="config.json: num_key_value_heads  (NOT num_attention_heads)")
    d.given("head dim", head_dim,
            source="config.json: head_dim, else hidden_size / num_attention_heads")
    d.given("bytes per element", b, source=str(dtype))

    per_layer = d.step("K and V for one layer",
                       f"2 x {n_kv_heads} x {head_dim} x {b:g}", 2 * n_kv_heads * head_dim * b, "B")
    total = d.step("across all layers", f"{per_layer:,.0f} x {n_layers}",
                   per_layer * n_layers, "B")
    d.result_label = "KV per token"
    d.check(f"{human_bytes(total)}/token x {ctx_check:,} ctx = "
            f"{human_bytes(total * ctx_check)} for ONE request at full context")
    if n_kv_heads == 0:
        d.warn("kv_heads of 0 is not possible — re-read the config")
    return d


def derive_weight_bytes(params, bits=16, name="weights"):
    """Parameters to bytes. Trivial, and worth writing out because the bit-width
    is where quantisation enters every other calculation."""
    d = Derivation(f"{name} in memory", formula="parameters x bytes_per_parameter")
    b = _bytes_per_param(bits)
    d.given("parameters", params / 1e9, "B params", "the model card, or estimated from shapes")
    d.given("bytes per parameter", b,
            source=f"{bits}-bit  (fp16 = 2 B, fp8/int8 = 1 B, int4 = 0.5 B)")
    total = d.step("weights", f"{params/1e9:,.1f}e9 x {b:g}", params * b, "B")
    d.result_label = "weights"
    d.check(f"{human_bytes(total)} — compare against the card before going further")
    return d


def derive_capacity(vram_gb, params, kv_per_token, ctx, weight_bits=16, util=0.90,
                    activation_gb=1.0):
    """VRAM to concurrent sequences, with every subtraction visible.

    The order matters and is worth narrating: the card is not all yours, weights
    come off the top, activations and the CUDA context take a slice, and what is
    left is divided by the per-token cost times the context length.
    """
    d = Derivation("concurrency from memory",
                   formula="(vram x util - weights - activations) / (kv_per_token x ctx)")
    b = _bytes_per_param(weight_bits)
    d.given("card memory", vram_gb, "GiB",
            "spec sheet. Vendors quote decimal GB; the allocator works in GiB, and the "
            "~7% gap is well inside this estimate's error bars")
    d.given("memory utilisation", util, source="vLLM --gpu-memory-utilization")
    d.given("parameters", params / 1e9, "B params", "model card")
    d.given("bytes per parameter", b,
            source=f"{weight_bits}-bit weights  (fp16 = 2 B, fp8 = 1 B, int4 = 0.5 B)")
    d.given("activations + CUDA context", activation_gb, "GiB",
            "~1 GiB small model; more with a large --max-num-batched-tokens")
    d.given("KV per token", kv_per_token, "B", "derived above")
    d.given("context length", ctx, "tokens", "what you commit to serving")

    usable = d.step("memory the engine may claim", f"{vram_gb:g} GiB x {util:g}",
                    vram_gb * GIB * util, "B")
    weights = d.step("weights", f"{params/1e9:,.1f}e9 x {b:g}", params * b, "B")
    budget = d.step("left for KV cache",
                    f"{human_bytes(usable)} - {human_bytes(weights)} - {activation_gb:g} GiB",
                    max(usable - weights - activation_gb * GIB, 0.0), "B")
    per_seq = d.step("KV for one full-context sequence",
                     f"{kv_per_token:,.0f} B x {ctx:,}", kv_per_token * ctx, "B")
    seqs = d.step("concurrent sequences", f"{human_bytes(budget)} / {human_bytes(per_seq)}",
                  (budget / per_seq) if per_seq else 0.0, "sequences")
    d.result_label = "concurrency"
    d.result_unit = "sequences"

    if budget <= 0:
        d.warn("the weights alone do not fit — quantise, shard across GPUs, or pick a "
               "smaller model. Every number below this line is meaningless.")
    elif seqs < 1:
        d.warn("under one full-context sequence fits. The model runs, but only for short "
               "prompts — cap max_model_len rather than promising this context.")
    else:
        d.check(f"halving context to {ctx//2:,} would give ~{seqs*2:,.0f} sequences — "
                "context and concurrency are the same knob")
    return d


def derive_decode_speed(params, mem_bw_gb_s, batch=1, ctx=1024, kv_per_token=0,
                        weight_bits=16, efficiency=0.7):
    """Decode throughput from bandwidth. The other half of every sizing answer.

    The narration that matters: a decode step re-reads every weight to produce
    one token *per sequence*, so the weight read is amortised across the batch
    and the KV read is not.
    """
    d = Derivation("decode speed from bandwidth",
                   formula="tokens/s = batch / ((weights + batch x ctx x kv_per_token) "
                           "/ (bandwidth x efficiency))")
    b = _bytes_per_param(weight_bits)
    d.given("parameters", params / 1e9, "B params", "model card")
    d.given("bytes per parameter", b, source=f"{weight_bits}-bit weights")
    d.given("memory bandwidth", mem_bw_gb_s, "GB/s", "spec sheet")
    d.given("achieved fraction of it", efficiency, source="0.6-0.8 in practice; measure it")
    d.given("batch size", batch, "sequences", "what the scheduler is actually running")
    d.given("context length", ctx, "tokens", "average, not maximum")
    d.given("KV per token", kv_per_token, "B", "derived above")

    weights = d.step("weight bytes read every step", f"{params/1e9:,.1f}e9 x {b:g}",
                     params * b, "B", "read once for the whole batch — this is why batching works")
    kv = d.step("KV bytes read every step",
                f"{batch} x {ctx:,} x {kv_per_token:,.0f}", batch * ctx * kv_per_token, "B",
                "grows with batch AND context — this is what eventually stops batching helping")
    total = d.step("bytes per step", f"{human_bytes(weights)} + {human_bytes(kv)}",
                   weights + kv, "B")
    step_s = d.step("time per step", f"{human_bytes(total)} / ({mem_bw_gb_s:,.0f} GB/s x {efficiency:g})",
                    total / (mem_bw_gb_s * 1e9 * efficiency), "s")
    tps = d.step("tokens per second", f"{batch} / {step_s*1000:,.2f} ms", batch / step_s if step_s else 0.0,
                 "tok/s")
    d.result_label = "aggregate throughput"
    d.result_unit = "tok/s"
    d.check(f"per sequence that is {tps/batch if batch else 0:,.1f} tok/s "
            f"({'above' if tps/max(batch,1) > 20 else 'below'} the ~20 tok/s reading speed "
            "a user notices)")
    share = kv / total if total else 0
    d.check(f"KV is {share:.0%} of the bytes moved — "
            + ("weights still dominate, so more batch is nearly free"
               if share < 0.4 else "KV now dominates, so more batch buys little"))
    return d


def derive_prefill_time(params, prompt_tokens, tflops, mfu=0.4, n_gpus=1):
    """Time to first token, from FLOPs. The one place FLOPs are the binding term."""
    d = Derivation("time to first token",
                   formula="TTFT = (2 x params x prompt_tokens) / (FLOPS x MFU x gpus)")
    d.given("parameters", params / 1e9, "B params", "model card")
    d.given("prompt length", prompt_tokens, "tokens", "your traffic, not the maximum")
    d.given("peak compute", tflops, "TFLOPS", "spec sheet, at the dtype you are serving")
    d.given("GPUs", n_gpus, source="tensor parallel degree")
    d.given("model FLOPs utilisation", mfu, source="0.3-0.5 realistically")

    flops = d.step("prefill FLOPs", f"2 x {params/1e9:,.1f}e9 x {prompt_tokens:,}",
                   2 * params * prompt_tokens, "FLOP",
                   "~2 FLOPs per parameter per token; ignores attention's quadratic term")
    rate = d.step("usable compute", f"{n_gpus} x {tflops:,.0f}e12 x {mfu:g}",
                  n_gpus * tflops * 1e12 * mfu, "FLOP/s")
    t = d.step("TTFT", f"{flops:.3g} / {rate:.3g}", flops / rate if rate else 0.0, "s")
    d.result_label = "TTFT"
    d.result_unit = "s"
    d.check(f"{t*1000:,.0f} ms — "
            + ("feels instant" if t < 0.3 else
               "acceptable for chat" if t < 1.0 else
               "a user will notice this; prefix-cache the shared prompt or raise TP"))
    d.check("if the prompt shares a long system prefix, most of this disappears with "
            "prefix caching — measure the shared share before optimising anything else")
    return d


def derive_cost_per_million(usd_per_hour, tokens_per_s, label="output tokens"):
    """Throughput to unit economics."""
    d = Derivation("cost per million tokens",
                   formula="$/hr / 3600 / tokens_per_s x 1e6")
    d.given("instance price", usd_per_hour, "USD/hour", "YOUR contract, not a list price")
    d.given("throughput", tokens_per_s, "tok/s", "measured, at your operating point")
    per_s = d.step("cost per second", f"{usd_per_hour:g} / 3600", usd_per_hour / 3600, "USD/s")
    per_tok = d.step("cost per token", f"{per_s:.3g} / {tokens_per_s:,.0f}",
                     per_s / tokens_per_s if tokens_per_s else float("inf"), "USD")
    d.step(f"cost per 1M {label}", f"{per_tok:.3g} x 1e6", per_tok * 1e6, "USD")
    d.result_label = "cost"
    d.result_unit = "USD per 1M tokens"
    d.check("compare against a managed API's per-million price before concluding anything "
            "— and remember this number excludes the people who run the fleet")
    return d


def worksheet(*, name="model", n_layers, n_kv_heads, head_dim, params, vram_gb, mem_bw_gb_s,
              tflops=0, ctx=4096, batch=16, prompt_tokens=1024, weight_bits=16, kv_dtype="fp16",
              util=0.90, activation_gb=1.0, efficiency=0.7, mfu=0.4, usd_per_hour=0.0,
              n_gpus=1) -> Worksheet:
    """The whole sizing answer from raw numbers, with every step shown.

    This is the deliverable: hand it an architecture and a spec sheet — no preset
    keys, no lookups — and it produces the chain a good answer walks, in order.
    Read the output aloud and you have given the answer.

        worksheet(name="mystery 34B",
                  n_layers=60, n_kv_heads=8, head_dim=128, params=34e9,
                  vram_gb=80, mem_bw_gb_s=3350, tflops=989,
                  ctx=8192, batch=32, weight_bits=8, usd_per_hour=3.50)
    """
    ws = Worksheet(f"sizing {name}")
    kv = ws.add(derive_kv_per_token(n_layers, n_kv_heads, head_dim, kv_dtype, ctx_check=ctx))
    ws.add(derive_weight_bytes(params, weight_bits))
    ws.add(derive_capacity(vram_gb, params, kv.value, ctx, weight_bits, util, activation_gb))
    speed = ws.add(derive_decode_speed(params, mem_bw_gb_s, batch, ctx, kv.value,
                                       weight_bits, efficiency))
    if tflops:
        ws.add(derive_prefill_time(params, prompt_tokens, tflops, mfu, n_gpus))
    if usd_per_hour:
        ws.add(derive_cost_per_million(usd_per_hour, speed.value))
    return ws


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def human_bytes(n) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TiB"  # pragma: no cover


def memory_report(gpu_spec, spec, weight_dtype="fp16", kv_dtype="fp16", seq_len=2048,
                  util=0.90, activation_gb=1.0, show_work=True) -> str:
    """The sizing block, with the arithmetic that produced each line beside it.

    Every row carries its own substitution, because a column of numbers teaches
    you to trust a function and a column of substitutions teaches you the method.
    Pass `show_work=False` for the bare figures once you no longer need them.

    For the full step-by-step version — givens, working, sanity checks — use
    `worksheet()`, which takes raw scalars instead of preset keys.
    """
    g, m = gpu(gpu_spec), model(spec)
    wb = dtype_bytes(weight_dtype)
    kvb = dtype_bytes(kv_dtype)

    w = weight_bytes(m, weight_dtype)
    act = activation_gb * GIB
    usable = g.vram_gb * GIB * util
    budget = kv_budget_bytes(g, m, weight_dtype, util, activation_gb)
    per_tok = kv_bytes_per_token(m, kv_dtype)
    capacity = kv_capacity_tokens(g, m, weight_dtype, kv_dtype, util, activation_gb)
    seqs = max_concurrent_sequences(g, m, seq_len, weight_dtype=weight_dtype,
                                    kv_dtype=kv_dtype, util=util, activation_gb=activation_gb)

    rows = [
        ("weights", human_bytes(w),
         f"{m.params / 1e9:,.2f}e9 params x {wb:g} B/param"),
        ("activations + ctx", human_bytes(act),
         "~assumed: CUDA context, activations, graph pool — not derived"),
        ("card, usable", human_bytes(usable),
         f"{g.vram_gb:g} GiB x {util:.0%} utilisation"),
        ("KV budget", human_bytes(budget),
         f"{human_bytes(usable)} - {human_bytes(w)} - {human_bytes(act)}"
         + ("   (negative, clamped to zero)" if usable - w - act < 0 else "")),
        ("KV per token", human_bytes(per_tok),
         f"2 x {m.n_layers} layers x {m.n_kv_heads} kv_heads x {m.head_dim} head_dim "
         f"x {kvb:g} B" + (f"   [GQA {m.gqa_ratio:g}:1]" if m.gqa_ratio > 1 else "   [no GQA]")),
        ("KV capacity", f"{capacity:,.0f} tokens",
         f"{human_bytes(budget)} / {human_bytes(per_tok)}"),
        (f"concurrency @ {seq_len:,}", f"{seqs:,.1f} sequences",
         f"{capacity:,.0f} tokens / {seq_len:,} per sequence"),
    ]

    label_w = max(len(r[0]) for r in rows) + 2
    value_w = max(len(r[1]) for r in rows) + 2
    head = f"{m.name} @ {weight_dtype} on {g.name} ({g.vram_gb:g} GB, util={util:.0%})"
    lines = [head]
    for label, value, work in rows:
        line = f"  {label:<{label_w}}{value:>{value_w}}"
        if show_work:
            # A tilde marks a number that was assumed rather than derived.
            line += f"   {work[1:]}" if work.startswith("~") else f"   = {work}"
        lines.append(line)

    if budget <= 0:
        lines.append("  !! weights alone do not fit — quantise, shard, or pick a smaller model")
        lines.append("     (every number below the KV budget line is meaningless here)")
    elif seqs < 1:
        lines.append("  !! under one full-context sequence fits — cap max_model_len rather "
                     "than promising this context")
    if show_work:
        lines.append("")
        lines.append("  Four numbers off the config (layers, kv_heads, head_dim, params) and")
        lines.append("  two off the spec sheet (memory, and the utilisation you choose).")
        lines.append("  nk.worksheet(...) shows the same chain step by step from raw scalars.")
    return "\n".join(lines)
