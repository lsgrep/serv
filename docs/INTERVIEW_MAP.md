# From claim to receipt

Interview prep usually produces a document of things you can *say*. The risk is
that ground-level specificity is exactly what a good interviewer probes for, and
a claim you have only read collapses on the second follow-up.

This maps each thing you want to be able to claim to the lab that gives you the
receipt — the measurement you took yourself, on hardware you can name.

Use it two ways: to find the lab behind a claim you feel shaky on, and — after
running a lab — to find the sentence it entitles you to say.

## Architecture and stack decisions

| The claim | The receipt | Lab |
|---|---|---|
| "Decode is memory-bandwidth bound; prefill is compute bound, and everything follows from that" | You derived both, then measured GPT-2 with and without a KV cache | [11](../notebooks/11_attention_from_scratch.ipynb), [03](../notebooks/03_kv_math_and_toy_engine.ipynb) |
| "A KV cache is *correct* because masking is causal — an encoder couldn't have one" | Cached decoding proven equal to full recomputation, in a test | [11](../notebooks/11_attention_from_scratch.ipynb) |
| "GQA is a repeat at inference — it buys concurrency, not speed" | The broadcast, and 80 GiB vs 10 GiB at 32K context | [11](../notebooks/11_attention_from_scratch.ipynb) |
| "Flash attention saves memory traffic, not FLOPs" | Online softmax implemented, tiled output matched to naive, both costs printed | [11](../notebooks/11_attention_from_scratch.ipynb) |
| "Prefix caching only works on prefixes because positions are rotated into the keys" | RoPE scores at the right and wrong offsets | [11](../notebooks/11_attention_from_scratch.ipynb) |
| "Give me a config and a GPU and I'll tell you the concurrency ceiling" | `napkin.memory_report` against what the engine actually allocated | [03](../notebooks/03_kv_math_and_toy_engine.ipynb), [00](../notebooks/00_drills.ipynb) |
| "GQA is a KV-cache optimisation — it's KV heads, not attention heads" | The 4x error, drawn, on four real model configs | [03](../notebooks/03_kv_math_and_toy_engine.ipynb) |
| "Batching is nearly free until the KV term catches up" | Arithmetic intensity against the ridge point, then the measured sweep | [02](../notebooks/02_hardware_economics.ipynb), [03](../notebooks/03_kv_math_and_toy_engine.ipynb) |
| "Retrieval fixes knowledge; fine-tuning fixes style" | The 34-of-40 triage, run on a corpus you can show | [08](../notebooks/08_rag_and_evals.ipynb) |
| ...and the measured version, with numbers from your own run | One LoRA scored on format, seen phrasings, held-out phrasings, and retrieval | [10](../notebooks/10_finetune_what_it_teaches.ipynb) |
| "The legitimate reasons to fine-tune are format, distillation, tone and refusals" | The format task going from a third of responses to nearly all | [10](../notebooks/10_finetune_what_it_teaches.ipynb) |
| "Catastrophic forgetting is real — here's what it looks like" | The capability probe, before and after | [10](../notebooks/10_finetune_what_it_teaches.ipynb) |
| "Testing on phrasings you trained on measures memorisation" | Paired train/held-out splits over identical facts | [10](../notebooks/10_finetune_what_it_teaches.ipynb) |
| "The eval harness is the portability layer" | A regression gate, plus a computed exit cost | [08](../notebooks/08_rag_and_evals.ipynb), [07](../notebooks/07_token_economics.ipynb) |

## Cost and the self-host question

| The claim | The receipt | Lab |
|---|---|---|
| "I can price a workload live and land on a call" | Three workloads, assumptions stated, verdict and flip-trigger named | [07](../notebooks/07_token_economics.ipynb) |
| "Thinking tokens bill as output — that's why the bill doubled" | The four billing regimes on identical traffic | [07](../notebooks/07_token_economics.ipynb) |
| "Route 85% to the cheap tier and the blended cost collapses" | The routing curve, including the break-even rate past which it stops paying | [07](../notebooks/07_token_economics.ipynb) |
| "Below ~$20-30K/month, self-hosting costs more in people than tokens" | Fully-loaded comparison with the FTE line visible | [07](../notebooks/07_token_economics.ipynb) |
| "The cheapest card and the cheapest card *at your SLO* are different cards" | The same sweep on T4 / L4 / A100, cost per million at a latency gate | [02](../notebooks/02_hardware_economics.ipynb) |
| "Some lock-in is the price of value — ours is measured" | `switching_cost()` broken out term by term | [07](../notebooks/07_token_economics.ipynb) |

## Debugging, live

| The claim | The receipt | Lab |
|---|---|---|
| "Overloaded servers fail by queueing, not erroring — I've watched it" | The death spiral, live, plus goodput collapsing while throughput holds | [01](../notebooks/01_serving_under_load.ipynb) |
| "95% GPU utilisation doesn't mean healthy" | Queue-vs-execution split, batch composition, preemption count | [01](../notebooks/01_serving_under_load.ipynb), [terminal](../terminal/DIAGNOSIS.md) |
| "Preemption is invisible in a latency dashboard until it has cost you throughput" | The preemption storm, injected and measured | [01](../notebooks/01_serving_under_load.ipynb), [03](../notebooks/03_kv_math_and_toy_engine.ipynb) |
| "That p99 is head-of-line blocking, not capacity" | Bimodal prompt workload, before and after chunked prefill | [09](../notebooks/09_serving_levers.ipynb) |
| "I'd measure the shared-prefix share before enabling prefix caching" | The control run where the flag buys nothing | [09](../notebooks/09_serving_levers.ipynb) |
| "Speculative decoding fades at high batch" | The acceptance curve, and the batch-32 comparison | [09](../notebooks/09_serving_levers.ipynb) |
| "TP is a trap when the memory term is small or the fabric is slow" | `tp_scaling` on NVLink vs off it, and a 1B model going *slower* | [09](../notebooks/09_serving_levers.ipynb) |

## Training and memory

| The claim | The receipt | Lab |
|---|---|---|
| "QLoRA removes three of the four memory terms and does nothing for the fourth" | The budget, term by term, with activations unchanged | [04](../notebooks/04_qlora_oom_postmortem.ipynb) |
| "A QLoRA OOM is usually a sequence-length problem" | The crossing point, drawn against your card's capacity | [04](../notebooks/04_qlora_oom_postmortem.ipynb) |
| "The logits tensor is the term nobody budgets" | Gigabytes per sample at 128K vocab, and the fused-CE fix | [04](../notebooks/04_qlora_oom_postmortem.ipynb), [00](../notebooks/00_drills.ipynb) |
| "It OOMed at step 40, not step 0 — here's why" | The allocator snapshot, read in memory_viz | [04](../notebooks/04_qlora_oom_postmortem.ipynb) |
| "Fragmentation and genuine demand need different fixes" | Reserved-minus-allocated, and when `expandable_segments` helps | [04](../notebooks/04_qlora_oom_postmortem.ipynb) |

## Measurement discipline

| The claim | The receipt | Lab |
|---|---|---|
| "Sixty prompts can't detect a three-point difference" | The confidence intervals, and the sample size that could | [08](../notebooks/08_rag_and_evals.ipynb) |
| "An uncalibrated LLM judge is a vibe check with a spreadsheet" | Kappa exposing a judge that says yes to everything | [08](../notebooks/08_rag_and_evals.ipynb) |
| "Measure recall@k separately from answer quality" | Retrieval evaluated alone, before any generation | [08](../notebooks/08_rag_and_evals.ipynb) |
| "A quantised model needs your eval, not the paper's" | FP16 vs INT4 on accuracy *and* distributional drift | [05](../notebooks/05_quantization_quality_cost.ipynb) |
| "Top-1 agreement is the sensitive test people skip" | Divergence points located token by token | [05](../notebooks/05_quantization_quality_cost.ipynb) |
| "Open loop vs closed loop decides whether a benchmark means anything" | The same server, both modes, different conclusions | [01](../notebooks/01_serving_under_load.ipynb) |

## Breadth beyond CUDA

| The claim | The receipt | Lab |
|---|---|---|
| "On a TPU, shapes are part of the program — serving is a padding-bucket problem" | Compile time per bucket, measured | [06](../notebooks/06_tpu_jax_serving.ipynb) |
| "Buffer donation isn't an optimisation, it's a design requirement for a KV cache" | The cost of the copy, measured both ways | [06](../notebooks/06_tpu_jax_serving.ipynb) |
| "Sharding is declarative — you annotate, the compiler implements" | A mesh and two `PartitionSpec`s | [06](../notebooks/06_tpu_jax_serving.ipynb) |
| "Memory arithmetic transfers across vendors; kernels don't" | The same KV formula validated on TPU | [06](../notebooks/06_tpu_jax_serving.ipynb) |

## Fluency, not just understanding

Understanding is what the labs build. Fluency is what fails under pressure, and
it needs separate practice: [00 — drills](../notebooks/00_drills.ipynb)
generates each whiteboard problem on fresh numbers, with a target time and the
worked answer. Do them until the *first move* is automatic — knowing where to
start is most of what a timer measures.

The self-check at the end of that notebook is ten questions with no arithmetic
in them at all. Those are the ones to be fluent on, because explanations are
what most of a conversation actually consists of.

## What these labs do not cover

Say this plainly if asked, so the gaps are deliberate rather than discovered:

* **Multi-node serving at real scale** — collective bandwidth, what breaks at
  70B across hosts, disaggregated prefill/decode fleets. Lab 9 models TP
  arithmetic; nothing here runs a multi-host deployment.
* **Writing kernels** — flash attention, paged attention, Triton, Pallas. Lab 3
  implements the scheduling *policy* and explicitly not the kernel.
* **Training at scale** — FSDP, pipeline parallelism, real data pipelines,
  preference tuning (DPO/RLHF). Labs 4 and 10 are single-card supervised
  fine-tuning on small models.
* **Production operations** — autoscaling on queue depth, canaries, request
  hedging, multi-tenant fairness.
* **Agent architectures** — planning loops, tool selection, multi-hop latency
  budgets. Lab 7 prices a hop; nothing here builds the chain.

Naming a gap and the counter-move raises credibility. Claiming coverage you do
not have loses it on the first follow-up.
