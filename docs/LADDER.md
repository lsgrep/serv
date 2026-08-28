# The ladder, and why it is in this order

Each lab exists because there is a question it is the only way to answer. The
order is not arbitrary: each one uses the previous one's vocabulary.

## 1 — Serving under load

**Question:** what does an overloaded LLM server actually look like?

**Why first:** every other lab is in service of this one. Before optimising
anything you need to have seen the failure mode — and it is not the one people
expect. LLM servers do not fail by erroring. They fail by queueing, and the
metric that tells you first (`num_requests_waiting`) is not the one on most
dashboards.

**The transferable insight:** throughput and goodput diverge under saturation. A
throughput dashboard shows a healthy system during an outage.

## 2 — Hardware economics

**Question:** which GPU should we buy?

**Why second:** now that you can measure a server, measure three. This is where
the roofline model earns its place: prefill is compute bound, decode is memory
bound, and that single distinction explains why a card with twice the FLOPs can
be no faster at decode.

**The transferable insight:** the cheapest card per token and the cheapest card
that meets your SLO are usually different cards. And memory is a cliff, not a
slope — the small card does not get gradually worse, it stops being an option.

## 3 — Napkin math and the toy engine

**Question:** why does any of that happen?

**Why third:** labs 1 and 2 produce phenomena; this one produces the model that
predicts them. It is also the lab that pays off most in interviews, because
"I have used vLLM" and "I can derive what vLLM is doing" are very different
answers.

Half A is arithmetic: KV bytes per token, concurrency, the ridge point. Half B
is ~300 lines of engine — paged allocator, continuous batching, preemption —
that demonstrates the arithmetic was right.

**The transferable insight:** the KV formula, and the fact that context length
and concurrency are the same knob.

## 4 — QLoRA and the OOM postmortem

**Question:** where does training memory go, and what do I do when it runs out?

**Why here:** it is the same predict-then-measure loop pointed at training, and
it introduces a tool — the allocator snapshot — that turns "it OOMed" into a
diagnosis. It also corrects a common wrong model: QLoRA removes three of the
four memory terms and does nothing about the fourth, which is why QLoRA OOMs are
sequence-length problems.

**The transferable insight:** get evidence before you change flags. Guess-and-
rerun is the slowest possible debugging loop and it teaches you nothing.

## 5 — Quantisation: quality and cost

**Question:** what does 4-bit cost me?

**Why here:** it needs lab 2's cost framing and lab 3's memory math to be a real
question rather than a benchmark score. It also introduces the measurement most
people skip — distributional drift — and the statistical humility to go with it:
sixty prompts cannot detect a three-point difference.

**The transferable insight:** quantisation's headline benefit is usually "runs
at all on the hardware you have", not "a bit faster".

## 6 — TPU serving in JAX

**Question:** how much of what I just learned is about LLMs, and how much is
about CUDA?

**Why last:** it is the only lab that recontextualises the others. Static shapes
turn serving into a padding-bucket problem; sharding becomes an annotation
rather than a codepath. Seeing a second execution model is what makes the first
one visible as a choice.

**The transferable insight:** memory arithmetic transfers across vendors;
kernels and scheduling assumptions do not.

## 7 — Token economics

**Question:** what does this cost, and should we host it ourselves?

**Why here:** it needs lab 2's cost framing, but it is a different question —
lab 2 prices *GPUs*, this prices *tokens*, and most AI workloads are bought by
the token. It is also the lab that most rewards being done out loud: the
failure mode is not bad arithmetic, it is hedging.

**The transferable insight:** four multipliers (batch, caching, prompt hygiene,
thinking tokens) move a bill more than any model choice, and the fully-loaded
self-host comparison has a line for people that is usually the largest one.

## 8 — RAG and the eval harness

**Question:** the answers are wrong. What is actually broken?

**Why here:** it is the highest-leverage diagnostic in applied GenAI work, and
it inverts the usual instinct. Splitting failures into *the passage was never
retrieved* versus *it was retrieved and the answer is still wrong* relocates the
problem in an hour, and only the second bucket is a model question at all.

It also carries the measurement discipline the whole repo depends on: recall@k
measured separately, a judge calibrated against humans before it is trusted,
confidence intervals stated before anyone celebrates three points, and a
regression gate that explains its own failures so people keep it switched on.

**The transferable insight:** fine-tuning teaches style, not facts. Diagnose
before prescribing, and make the diagnosis cheap enough that nobody has to be
wrong out loud.

## 9 — Serving levers, measured

**Question:** those optimisations everyone lists — what does each one actually buy?

**Why last:** lab 1's diagnosis playbook names them; this turns each into a
number, including a control run showing the workload where it does nothing.
That control is the whole point. "Enable prefix caching" is documentation.
"Measure what share of your input is shared, because on unique prompts it is
just memory you stopped using for KV" is experience.

**The transferable insight:** every lever is a trade, and the workload decides.
Being able to name the case where your own recommendation is wrong is the
strongest form of the recommendation.

---

## What the ladder does not cover

Worth knowing, and worth saying if asked, so the gaps are deliberate rather than
invisible:

* **Multi-node serving** — collective bandwidth over a real network, what breaks
  at 70B across hosts, disaggregated prefill/decode fleets. Lab 9 models tensor
  parallelism arithmetically and lab 6 touches sharding declaratively; neither
  runs a multi-host deployment.
* **Structured output and constrained decoding** — grammar-constrained sampling
  and its throughput cost.
* **Real production concerns** — autoscaling on queue depth, canaries, request
  hedging, multi-tenant fairness.
* **Writing kernels** — flash attention, paged attention, Triton, Pallas. Lab 3
  implements the *policy* and explicitly not the kernel.

Any of these is a reasonable lab 7.
