# The terminal rig — labs 1 and 2, rehearsed

The notebooks build the intuition. This builds the performance.

A live-debugging interview does not feel like a notebook. It feels like a
terminal, a log scrolling past, someone watching you think, and no autocomplete
for the metric name you half-remember. Do the notebook labs first; then do this
once on a rented box, out loud, with the timer running.

## Setup

Any single-GPU box works. A T4 or L4 by the hour is plenty — nothing here needs
an A100, and the constrained card makes the failures arrive sooner.

```bash
git clone https://github.com/lsgrep/serv.git && cd serv/terminal
./setup.sh                      # venv, vllm, model download (~10 min, once)
./lab.sh                        # tmux: 4 panes, server already starting
```

`lab.sh` lays out:

```
+------------------------+------------------------+
| ./serve.sh             | ./watch_metrics.sh     |
| (vLLM log, live)       | (the 7 numbers, 1s)    |
+------------------------+------------------------+
| ./load.py ...          | watch nvidia-smi       |
| (you drive from here)  |                        |
+------------------------+------------------------+
```

Left column is what you control. Right column is what you read. In an interview
you will be talking while looking at the right column, so practise reading it
while explaining something else — that split is most of the difficulty.

## The basic run

```bash
# healthy
./load.py --rps 1 --duration 90

# past capacity — open loop, so the client does not back off
./load.py --rps 6 --duration 90

# capacity curve
./load.py --sweep 1,2,4,8,16,32,64 --duration 25 --json runs/sweep.json
```

Watch the order things move in the metrics pane. It is always the same:
`waiting` climbs → `kv cache` pins at 100% → `preempt` starts rising → mean TTFT
detaches. Being able to narrate that sequence *while it happens* is the thing
being tested.

## Commands worth having in muscle memory

```bash
# the three numbers, without the dashboard
curl -s localhost:8000/metrics | grep -E 'num_requests_(running|waiting)|cache_usage'

# is it alive, and what is it serving
curl -s localhost:8000/health -o /dev/null -w '%{http_code}\n'
curl -s localhost:8000/v1/models | python3 -m json.tool

# one request, timed, from a cold cache
time curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":16}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'

# what did the engine decide at startup — KV blocks, max concurrency, dtype
grep -E 'KV cache|Maximum concurrency|dtype' runs/vllm.log

# preemption, in the log rather than the counter
grep -ci preempt runs/vllm.log

# is the GPU actually busy, or is the client the bottleneck
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

## Then: `DIAGNOSIS.md`

Eight scenarios, each with a fault you inject yourself, the symptom you will
see, and the reasoning to say out loud. Work through them until you can name the
root cause from the metrics pane alone, before running anything else.

That is the actual rehearsal. Everything before this point is setup.
