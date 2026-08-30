# Derive it, don't recall it

Nobody should walk into a conversation having memorised that an H100 has 3.35
TB/s of bandwidth or that Llama-3.3-70B has 80 layers. Those numbers are
**given** to you — in a config.json, on a spec sheet, by the person asking. What
is being tested is whether you can put them in the right places, in the right
order, and say what the answer means.

So this page is not a numbers sheet. It is the set of formulas, what each term
is, **where you read it off**, and the sanity check that catches a unit error
before it reaches a slide.

Every formula here has a `derive_*` function in `servlab.napkin` that takes raw
scalars and prints the substitution. Use those to check yourself, never to
produce an answer you did not work out first.

```python
from servlab import napkin as nk

nk.derive_kv_per_token(n_layers=80, n_kv_heads=8, head_dim=128, dtype="fp16")
nk.from_config(json.load(open("config.json")))          # shows which field fed which term
nk.worksheet(name="mystery 34B", n_layers=60, n_kv_heads=8, head_dim=128,
             params=34e9, vram_gb=80, mem_bw_gb_s=3350, tflops=989,
             ctx=8192, batch=32, weight_bits=8, usd_per_hour=3.50)
```

---

## 1. KV cache per token

```
kv_bytes_per_token = 2 × layers × kv_heads × head_dim × bytes_per_element
```

| Term | Read it from | Trap |
|---|---|---|
| `2` | K and V, both stored | — |
| `layers` | `num_hidden_layers` | — |
| `kv_heads` | `num_key_value_heads` | **This is the one people get wrong.** Using `num_attention_heads` overstates a Llama-3 cache by 4× |
| `head_dim` | `head_dim` if stated, else `hidden_size / num_attention_heads` | Several modern configs state it, and it is not always the ratio |
| `bytes_per_element` | your KV dtype — fp16 = 2, fp8 = 1 | Independent of the *weight* dtype |

**Sanity check:** multiply by your context length. If one full-context request
is not a plausible fraction of the card, re-check `kv_heads`.

**Say out loud:** "KV heads, not attention heads — that's the GQA discount."

---

## 2. Weights in memory

```
weight_bytes = parameters × bytes_per_parameter
```

fp16 = 2 B, fp8/int8 = 1 B, int4 = 0.5 B.

**Trap:** a 4-bit checkpoint is not 4 bits per parameter. Group scales and
zero-points put AWQ nearer 4.5 bits — `nk.awq_weight_bytes` has the honest
version, and the difference is a few percent, not a rounding error at 70B.

**Sanity check:** compare against the card *before* going any further. If the
weights alone do not fit, every number after this one is meaningless.

---

## 3. Concurrency

```
kv_budget   = vram × utilisation − weights − activations
concurrency = kv_budget / (kv_bytes_per_token × context_length)
```

| Term | Read it from | Trap |
|---|---|---|
| `vram` | spec sheet | Vendors quote decimal GB; allocators work in GiB. ~7% — inside the error bars here, but say so |
| `utilisation` | `--gpu-memory-utilization`, typically 0.90 | It is a fraction of the *whole* card, not of what's left |
| `activations` | ~1 GiB small models, more with a large `--max-num-batched-tokens` | Rises with batch and with CUDA graphs |
| `context_length` | what you commit to serving | Not `max_position_embeddings` — that is what the weights *support* |

**Sanity check:** halving the context should double the concurrency. If it does
not, you have made an arithmetic error.

**Say out loud:** "Context length and concurrency are the same knob. There is no
third option." And: "Memory is a cliff, not a slope."

---

## 4. Decode speed

```
bytes_per_step = weights + (batch × context × kv_bytes_per_token)
step_time      = bytes_per_step / (bandwidth × efficiency)
tokens_per_s   = batch / step_time
```

`efficiency` is the fraction of spec-sheet bandwidth you actually achieve —
0.6–0.8 in practice. **If your model needs more than 1.0 to fit a measurement,
your model of what is being read is wrong.** That is a finding, not a fudge.

The two terms behave differently, and the difference is the whole batching
argument:

* **weights** are read once for the *entire batch* — so batching is nearly free
  while this term dominates,
* **KV** is read per sequence and grows with context — so batching stops helping
  once this term takes over.

**Sanity check:** divide by batch. Per-sequence tokens/s below ~20 is slower than
a person reads.

**Say out loud:** "Decode re-reads every weight to produce one token per
sequence. That's why it's bandwidth bound and why continuous batching exists."

---

## 5. Time to first token

```
prefill_flops = 2 × parameters × prompt_tokens
ttft          = prefill_flops / (FLOPS × MFU × gpus)
```

MFU (model FLOPs utilisation) is 0.3–0.5 realistically. The `2` is roughly two
FLOPs per parameter per token; this ignores attention's quadratic term, which is
a rounding error until long context — see §7.

**This is the only place peak FLOPs bind.** Prefill is compute bound, decode is
bandwidth bound, and naming which regime you are in before reaching for a number
is most of the answer.

**Levers when you miss the SLA, cheapest first:** prefix-cache the shared system
prompt (often 50–80% of the prompt, so nearly free), chunked prefill so it stops
blocking decode, more tensor parallelism (halves the time, costs efficiency),
then a smaller model.

---

## 6. Cost per million tokens

```
$ per 1M = ($/hour ÷ 3600 ÷ tokens_per_s) × 1e6
```

**Trap:** the `$/hour` must be *your* number. Reserved and on-demand differ by
more than most of the performance gaps you would be arguing about. And this
figure excludes the people who run the fleet, which at small scale is the
largest line item — see [lab 7](../notebooks/07_token_economics.ipynb).

---

## 7. When attention stops being a rounding error

```
attention_flops  ≈ 4 × n² × d_model          (prefill, per layer)
projection_flops ≈ 2 × n × (projection widths)
ffn_flops        ≈ 2 × n × 2 × ffn_mult × d_model²
```

Attention's share grows linearly with context. For an 8B-shaped layer it is
~2% at 512 tokens, ~9% at 2K, and passes half at ~21K. Below the crossover,
context is nearly free; above it, doubling context more than doubles the cost.

`nk.quadratic_crossover(d_model, n_heads, n_kv_heads)` computes it for any shape.

---

## 8. Training memory

```
total ≈ weights + gradients + optimizer + activations + logits + overhead
```

| Term | Full fine-tune (fp16 + AdamW) | QLoRA |
|---|---|---|
| weights | 2 B/param | 0.5 B/param |
| gradients | 2 B/param | 2 B/param **of the adapters only** |
| optimizer | 8 B/param (fp32 moments + master) | ~2 B/param of adapters |
| activations | `batch × seq × hidden × layers × k` | **unchanged** |
| logits | `batch × seq × vocab × 4 B × 2` | **unchanged** |

QLoRA collapses three of five terms and does nothing for the other two. That is
why a QLoRA OOM is almost always a **sequence-length** problem, and why "use a
smaller model" is usually the wrong response.

**The term nobody budgets:** logits. A 128K vocab at 4K sequence length is ~4 GiB
per sample once the fp32 loss copy is counted. A chunked or fused cross-entropy
removes it.

---

## 9. Queueing

```
utilisation = arrival_rate × service_time
wait        = utilisation × service_time / (1 − utilisation)      [M/M/1]
concurrency = arrival_rate × latency                              [Little's law]
```

Not a model of vLLM — a model of *why the last 10% of utilisation costs
everything*. Nothing about the server changes between 80% and 95% load; the wait
quadruples.

**Say out loud:** "The wall isn't where the capacity is."

---

## The order to work them in

Given a config and a card, this is the chain — and `nk.worksheet()` prints
exactly it:

1. **KV per token** — from the config's four numbers.
2. **Weights** — does it fit at all? Stop here if not.
3. **KV budget → concurrency** — at the context you have committed to.
4. **Decode speed** — and check which term dominates the bytes.
5. **TTFT** — the compute-bound half.
6. **Cost** — if a price was given.

Six steps, four of which are one multiplication. Practise them on random
architectures with [lab 0](../notebooks/00_drills.ipynb), which hands you raw
numbers and never a model name.

## The habit

State assumptions, round aggressively, land on a number, then say what would
change it:

> "Call it 80 layers, 8 KV heads, 128 head dim, fp16 — that's 2 × 80 × 8 × 128 ×
> 2, about 320 KiB per token. At 8K context that's 2.6 GiB a request, so on an
> 80 GB card with fp8 weights I've got roughly 40 GiB of KV budget and around
> fifteen concurrent sequences. If that's not enough concurrency the cheapest fix
> is FP8 KV, which doubles it, and I'd want to eval that before shipping it."

That is the whole skill. Nothing in it was recalled.
