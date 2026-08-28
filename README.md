# serv — an LLM inference serving lab ladder

Six labs that teach inference serving by making you measure it. Each one is a
Colab notebook that runs on a free T4, backed by a small tested Python package
so the notebooks stay narrative and the logic stays honest.

Labs 0, 7, 8 and 11 run on a laptop. The rest want a GPU, and the free tier is
genuinely enough for all of them.

**Where to start:** the numbering is build order, not learning order. Read
**11 → 03 → 01** first — lab 11 derives the attention mechanics that lab 3's
napkin math assumes and lab 1's failures are made of. After those three, the
rest can be taken in any order.

The organising idea: **predict on paper, then measure, then explain the gap.**
A prediction within 2x means your mental model works. Off by 10x means a term is
missing, and finding which one is the lesson.

Each notebook opens with one sentence — the claim you should be able to make
when you finish it. That sentence is the deliverable; the code is how you earn
the right to say it.

## The ladder

| | Lab | Open | Runs on | What you walk away with |
|---|---|---|---|---|
| 0 | [Drills](notebooks/00_drills.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/00_drills.ipynb) | CPU | The whiteboard problems on random numbers, timed, with worked answers — fluency, not understanding |
| 1 | [Serving under load](notebooks/01_serving_under_load.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/01_serving_under_load.ipynb) | T4 | vLLM under overload, live: queue depth, KV cache, TTFT. Throughput holds while goodput collapses |
| 2 | [Hardware economics](notebooks/02_hardware_economics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/02_hardware_economics.ipynb) | T4 → L4 → A100 | The same sweep on three cards. Cost per million tokens at your latency SLO |
| 3 | [Napkin math + toy engine](notebooks/03_kv_math_and_toy_engine.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/03_kv_math_and_toy_engine.ipynb) | T4 (or CPU) | The KV formula, and a paged-block engine with continuous batching and preemption that proves it |
| 4 | [QLoRA and the OOM postmortem](notebooks/04_qlora_oom_postmortem.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/04_qlora_oom_postmortem.ipynb) | T4 | A training memory budget, a deliberate OOM, and an allocator snapshot that names the culprit |
| 5 | [Quantisation: quality and cost](notebooks/05_quantization_quality_cost.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/05_quantization_quality_cost.ipynb) | T4 (FP8 needs L4) | FP16 vs INT4-AWQ on accuracy, latency, VRAM, cost — plus the distributional check people skip |
| 6 | [TPU serving in JAX](notebooks/06_tpu_jax_serving.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/06_tpu_jax_serving.ipynb) | Colab TPU | Why static shapes make serving a padding-bucket problem, and what that says about CUDA |
| 7 | [Token economics](notebooks/07_token_economics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/07_token_economics.ipynb) | CPU | Price a workload live: routing curves, break-even escalation, managed vs self-host with the FTE line visible |
| 8 | [RAG and the eval harness](notebooks/08_rag_and_evals.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/08_rag_and_evals.ipynb) | CPU | Retrieval measured on its own, the retrieval-vs-synthesis triage, a calibrated judge and a regression gate |
| 9 | [Serving levers, measured](notebooks/09_serving_levers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/09_serving_levers.ipynb) | T4 / L4 | Prefix caching, chunked prefill, FP8 KV, speculative decoding, TP — each with the workload where it does nothing |
| 10 | [A fine-tune, and what it taught](notebooks/10_finetune_what_it_teaches.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/10_finetune_what_it_teaches.ipynb) | T4 | One LoRA, measured on two axes: format compliance jumps, held-out factual recall barely moves, retrieval beats it |
| 11 | [Attention from scratch](notebooks/11_attention_from_scratch.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsgrep/serv/blob/main/notebooks/11_attention_from_scratch.ipynb) | CPU | Derive the KV cache instead of quoting it — causal masking, GQA as a repeat, online softmax, and why prefix caching is a *prefix* cache |
Each notebook also carries that badge in its own first cell, so however you
arrive at one, it is a click away from running.

## Then leave the notebook

Labs 1 and 2 adapt well to a notebook — the server runs detached, cells poll
`/metrics`, one cell live-updates a matplotlib chart, and watching the death
spiral develop as a chart is better intuition than watching it scroll past.

What a notebook cannot rehearse is diagnosing it live, which is what an
interview actually tests. [`terminal/`](terminal/) is the same two labs as shell
scripts — tmux with four panes, `curl`, the log — plus
[`terminal/DIAGNOSIS.md`](terminal/DIAGNOSIS.md): eight faults you inject
yourself, with the reasoning to say out loud.

**Notebooks to build the intuition; terminal to rehearse the performance.**

## Quickstart

**Colab.** Open a notebook, run cell 1. It clones this repo, installs what that
lab needs, and prints the GPU it found along with the warnings that apply to it.
Every notebook is idempotent, so a disconnect costs three minutes.

**Locally, or on a rented box.**

```bash
git clone https://github.com/lsgrep/serv.git && cd serv
pip install -e ".[plot,load,dev]"
pytest -q                                    # 149 tests, CPU only, ~7s

# no GPU? the scheduler still reproduces the dynamics
python -c "
from servlab.toy.scheduler import simulate
rows, reqs = simulate(rps=14, duration=30)
print('peak queue', max(r['waiting'] for r in rows), 'preemptions', rows[-1]['preemptions'])"
```

## T4 gotchas, in the order they will bite you

The free tier is genuinely enough for every phenomenon in these labs. It has
five sharp edges, and `servlab` handles the first two for you:

1. **No bf16.** Turing is sm_75. vLLM reads `bfloat16` from most modern configs
   and refuses to start. Pass `--dtype half` — `servlab.serve` detects the card
   and does it automatically.
2. **16 GB is not enough for an 8B model in fp16.** Weights alone are ~15 GiB,
   leaving nothing for KV. Use a 3B in fp16 or an 8B AWQ quant. Every phenomenon
   in these labs reproduces fine at 3B.
3. **No FP8 anything.** Needs Ada (sm_89) or newer. Lab 5's FP8 row fills itself
   in when you rerun on an L4.
4. **Recent vLLM increasingly assumes Ampere+.** If you hit strange install or
   kernel errors on a T4, that is the hardware, not you. Pin an older release,
   or move to an L4, which removes this entire class of problem.
5. **Sessions are ephemeral.** Cell 1 of every notebook holds all installs and
   downloads; long runs checkpoint to Drive.

And one that is secretly a feature: **switching runtime type is free hardware
comparison.** The same sweep notebook on T4 → L4 → A100 is lab 2, with no infra
work at all.

## What is in the package

`servlab` is split by what each module needs, so the arithmetic and the
scheduling policy stay testable on a CPU runner:

| module | needs | what it is |
|---|---|---|
| `napkin` | nothing | KV-cache and roofline math, model and GPU specs, queueing |
| `prometheus` | nothing | `/metrics` parser with vLLM's naming (both spellings) |
| `stats` | nothing | TTFT / TPOT / goodput, defined once so every lab agrees |
| `toy.allocator`, `toy.scheduler` | nothing | Paged blocks, continuous batching, preemption — plus a GPU-free simulation of the death spiral |
| `monitor`, `loadgen`, `serve` | network | Metrics poller, open/closed-loop generators, background `vllm serve` |
| `pricing` | nothing | Token economics: a price table you maintain, routing curves, managed-vs-self-host |
| `attention` | numpy | Attention, KV cache, GQA, flash tiling and RoPE, built from scratch |
| `rag` | nothing | Chunking, BM25, hybrid fusion, recall@k — and the retrieval-vs-synthesis triage |
| `finetune` | nothing | Paired train/held-out datasets and scorers for the fine-tuning experiment |
| `drills` | nothing | Randomised whiteboard problems with worked answers |
| `evalkit` | network | Quantisation eval, judge calibration, sample-size maths, the regression gate |
| `gateway` | network | A thin multi-provider client, and a computed exit cost |
| `memory` | torch | Training budget math, OOM snapshot recorder |
| `toy.engine` | torch | The scheduler driving real GPT-2 forward passes |
| `plots` | matplotlib | Chart defaults: fixed hue order, one axis per plot, red reserved for status |

The toy engine emits the same metric names vLLM does, so lab 1's dashboard
function plots lab 3's toy engine without changes. That is deliberate: if you
can read one chart you can read the other.

[`docs/INTERVIEW_MAP.md`](docs/INTERVIEW_MAP.md) maps each claim worth making to
the lab that produces the receipt for it — and lists, plainly, what these labs
do not cover.

## Notes on the numbers

* **Prices in `servlab.napkin.GPUS` are placeholders.** Edit `usd_per_hour` to
  what you actually pay — reserved and on-demand differ by more than the
  performance gaps these labs measure, so every cost conclusion sits downstream
  of that one field.
* **`servlab.pricing.MODELS` is a snapshot you maintain**, stamped with
  `VERIFIED_ON`. API pricing moves monthly and intro rates expire; `staleness()`
  will warn you, but it cannot re-verify for you. Re-check before quoting
  anything to anyone.
* **Colab GPUs are shared and throttled.** Ratios measured the same day are
  useful; absolute numbers are not publication-grade. Re-measure on a dedicated
  box if a number matters commercially, and say so when you present it.

## Development

```bash
pip install -e ".[dev]"
ruff check servlab tests
pytest -q
```

CI runs both on every push. The tests cover the parts worth trusting: the KV and
roofline arithmetic, the Prometheus parse (including the `+Inf` bucket case), the
latency and goodput definitions, and the allocator and scheduler — including
that the simulation actually reproduces the death spiral. The network paths are
covered too, against fake `/metrics` and SSE servers on real sockets, because a
stream-format change turns TTFT into `None` rather than into an error.
