# Diagnosis rehearsal

Eight scenarios. Each one you **inject yourself**, then diagnose as though you
had walked in on it. Cover the "root cause" section before you start.

How to practise: set a five-minute timer, inject the fault, and narrate. Not
silently — out loud, in the words you would use to a colleague. The gap between
knowing the answer and being able to say it under mild pressure is the entire
difference this rehearsal is closing.

The shape of a good answer, every time:

1. **What I observe** — the specific metric, with its value.
2. **What that rules in and out** — one sentence, before touching anything.
3. **The one command I would run next**, and what each outcome would mean.
4. **The fix, and what it costs.** Every fix has a cost. Naming it is the part
   that separates a good answer from a memorised one.

---

## 1. TTFT climbing, throughput flat

**Inject**
```bash
./load.py --rps 8 --duration 120
```

**Symptom** — `waiting` grows without bound. `kv cache` pins near 100%. Mean
TTFT rises steadily; no request fails. Output tokens/s is roughly unchanged from
the healthy run.

**Say** — "The server is saturated and failing by queueing rather than by
erroring. Throughput is flat because the engine is as busy as it ever was; the
queue is absorbing the excess, so latency is unbounded. Nothing is broken. There
is simply more work arriving than leaving."

**Root cause** — Offered load exceeds capacity. Open loop, so the client never
backs off.

**Fix, and the cost** — Admission control or a rate limit: shed load at the edge
so some requests fail fast rather than all of them timing out. Cap
`--max-num-seqs` to bound in-flight work and therefore latency, at the cost of
peak throughput. Then scale out, which is the honest answer once the cheap
levers are spent.

**The follow-up you will get** — *"Why is throughput not dropping?"* Because
throughput measures work completed, and the engine is completing work at its
maximum rate. What collapsed is **goodput** — work completed inside the SLO.
That gap is the whole failure.

---

## 2. The server will not start at all

**Inject**
```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --dtype bfloat16     # on a T4
```

**Symptom** — Exits during startup: *"Bfloat16 is only supported on GPUs with
compute capability of at least 8.0"*.

**Say** — "Turing is sm_75 and has no bf16. The config's `torch_dtype` is
bfloat16 and vLLM honours it, so it refuses rather than silently downcasting."

**Root cause** — Model dtype does not match hardware capability.

**Fix, and the cost** — `--dtype half`. Costs nothing here; fp16 and bf16 have
the same footprint. bf16's wider exponent matters for training stability, not
for inference of an already-trained model.

**Worth knowing** — This is why `serve.sh` reads `compute_cap` from
`nvidia-smi` and picks the dtype for you. The same class of error covers FP8
(needs sm_89+) and any kernel that assumes Ampere. If a wheel fails oddly on a
T4, suspect the hardware floor before your setup.

---

## 3. Starts, then dies with "no available memory for the cache blocks"

**Inject**
```bash
MAX_LEN=32768 ./serve.sh
```

**Symptom** — Weights load, then startup fails on KV cache allocation.

**Say** — "Weights fit; the KV cache does not. The engine needs at least one
sequence's worth of blocks at `max_model_len`, and after weights there is not
enough left."

**The arithmetic to do out loud** — "3B in fp16 is about 6 GiB. On a 16 GB card
at 0.9 utilisation that leaves roughly 7-8 GiB for KV. This model is 36 KiB per
token, so 32k of context is about 1.1 GiB for a single sequence — which fits,
but barely, and leaves no room to batch. Push `max_model_len` higher, or raise
utilisation past what activations need, and it stops fitting at all."

**Root cause** — `max_model_len` × KV-per-token exceeds the post-weights budget.

**Fix, and the cost** — Lower `--max-model-len` (long prompts get rejected);
raise `--gpu-memory-utilization` (less headroom for activation spikes — this is
the one that OOMs mid-run instead of at startup, which is worse); quantise the
KV cache to FP8 on Ada+ (halves bytes per token, small quality risk); or use a
smaller or quantised model.

---

## 4. Throughput is fine, p99 is terrible

**Inject** — two generators at once, one with long prompts:
```bash
./load.py --rps 3 --prompt-tokens 128  --duration 120 &
./load.py --rps 0.3 --prompt-tokens 3500 --duration 120
```

**Symptom** — Mean TTFT looks acceptable. p99 is many times p50. `waiting` is
low, `kv cache` is not pinned, no preemptions.

**Say** — "The queue is not the problem — `waiting` is low. This is
head-of-line blocking: a long prefill occupies the engine for a whole step, and
every decoding sequence stalls behind it. The tail is a *mix* problem, not a
capacity problem."

**Root cause** — Long prefills stalling decode. Bimodal prompt lengths.

**Fix, and the cost** — Chunked prefill, so a long prompt is split across steps
and interleaves with decode: the long request's own TTFT gets slightly worse,
everyone else's tail gets much better. Alternatively, route long prompts to a
separate pool, at the cost of running two deployments.

**How to prove it before fixing** — Correlate TTFT with prompt length. If the
tail is entirely long prompts it is admission ordering; if the tail is *short*
prompts, they are the ones queueing behind the long ones, which is head-of-line
blocking confirmed.

---

## 5. Preemption storm

**Inject**
```bash
UTIL=0.42 MAX_LEN=4096 ./serve.sh          # deliberately starve the KV pool
./load.py --concurrency 32 --duration 120
```

**Symptom** — `preempt` climbs steadily. `kv cache` sits at 100%. Throughput is
*lower* than at smaller concurrency. Nothing errors.

**Say** — "The engine is admitting more sequences than it has blocks for, so it
evicts running sequences to make room. In recompute mode the evicted sequence
loses its KV cache and has to prefill again — so the same work is being done
twice and throughput drops. More concurrency is making it worse, not better."

**Root cause** — Admitted concurrency exceeds what the KV pool can sustain.

**Fix, and the cost** — Cap `--max-num-seqs` so admitted work fits (lower peak
throughput, but higher *useful* throughput); raise the KV budget via
utilisation or a smaller `max_model_len`; quantise the KV cache.

**The insight worth stating** — Preemption is invisible in a latency dashboard
until it has already cost you throughput. `vllm:num_preemptions_total` climbing
is the earliest reliable signal that a server is past its sustainable
concurrency, and most dashboards do not plot it.

---

## 6. Latency is beautiful and the GPU is idle

**Inject**
```bash
./load.py --concurrency 1 --duration 60
```

**Symptom** — TTFT excellent, `running` is 1, `nvidia-smi` shows low
utilisation, throughput is a fraction of what the sweep showed.

**Say** — "The client is the bottleneck, not the server. One request in flight
means batch size one, and decode at batch one uses about 1% of the card's
compute — it is entirely memory-bandwidth bound reading the weights. The server
has nothing to batch."

**Root cause** — Closed-loop generator with too little concurrency. Also what a
misconfigured benchmark looks like.

**Fix, and the cost** — More concurrent clients, or open loop. Free — you were
measuring the wrong thing.

**The interview version** — This is the most common benchmarking mistake and a
good thing to volunteer. "Before I trust any latency number I check whether the
generator kept up: if the client was late sending, the measurement is a floor,
not a measurement." `load.py` prints exactly that warning when the generator
falls behind.

---

## 7. The first request after a restart is very slow

**Inject** — restart the server and immediately send one request.

**Symptom** — First request takes seconds; subsequent identical requests are
fast.

**Say** — "Cold start: weights paging in, CUDA context and kernel autotuning,
CUDA-graph capture if it is enabled, and an empty prefix cache. It is a
warm-up cost, not a serving problem — but it *is* a user-visible problem if you
put a cold replica straight into rotation."

**Root cause** — Cold start.

**Fix, and the cost** — Warm every replica before it takes traffic, and keep it
out of the load balancer until a synthetic request succeeds. Costs a few seconds
of deploy time and removes an entire class of pager alert.

**Adjacent** — On a TPU this is far worse and structural: a new shape is a
recompile (lab 6), which is why TPU stacks warm every padding bucket at startup.

---

## 8. Fine in staging, bad in production

No injection — this one is a conversation, and it is the one most likely to be
asked as a scenario rather than a live debug.

**Say** — "First I would check whether the *input distribution* matches. Staging
load generators send uniform synthetic prompts; real traffic is bimodal, has
shared system prompts, and much longer contexts. Three things follow from that,
and I would check them in this order:

1. **Prompt length distribution.** KV cache is linear in context, so a p99 of
   4000 tokens against a synthetic 256 is an 8x memory difference per request —
   your capacity model was measuring a different workload.
2. **Prefix sharing.** Real traffic often shares a long system prompt. With
   prefix caching on, real traffic can be *cheaper* than synthetic. Without it,
   you are re-prefilling the same tokens for every request.
3. **Output length distribution.** Time in the engine is dominated by output
   tokens, so a shift in output length moves concurrency directly through
   Little's law."

**The generalisable point** — Reproduce the *distribution*, not the average. A
benchmark at the mean of a bimodal workload measures a workload that does not
exist.

---

## Warm-up questions

Answer these before touching the terminal. If any takes more than about thirty
seconds, that is the lab to go back to.

1. KV cache bytes per token, from the formula. Why KV heads and not attention heads?
2. Why is decode memory bound and prefill compute bound? What follows for batching?
3. What is a preemption, what does it cost, and which metric shows it first?
4. Throughput versus goodput. Which one holds up during an outage, and why?
5. Given a 16 GB card and a 3B fp16 model, how many concurrent 2k-token sequences?
6. Open loop versus closed loop. Which one can reproduce an outage, and why not the other?
7. Name three fixes for a saturated server and the cost of each.
8. What does `--enforce-eager` turn off, and what does it buy on a small card?
