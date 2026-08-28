#!/usr/bin/env python3
"""Load generator CLI — the same engine the notebooks use.

    ./load.py --rps 4 --duration 90              # open loop: can overload
    ./load.py --concurrency 16 --duration 60     # closed loop: cannot overload
    ./load.py --sweep 1,2,4,8,16,32              # capacity curve

Open loop is the default because it is the only mode that reproduces an outage:
requests launch on a Poisson schedule whether or not the server keeps up.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from servlab.loadgen import run_load, sweep  # noqa: E402
from servlab.stats import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--rps", type=float, help="open loop: arrivals per second")
    ap.add_argument("--concurrency", type=int, help="closed loop: requests in flight")
    ap.add_argument("--sweep", help="comma-separated concurrency levels")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--slo-ttft", type=float, default=1.0)
    ap.add_argument("--slo-tpot", type=float, default=0.05)
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args()

    if args.sweep:
        levels = [int(x) for x in args.sweep.split(",")]
        rows = sweep(args.url, args.model, levels, duration=args.duration, warmup=5,
                     prompt_tokens=args.prompt_tokens, max_tokens=args.max_tokens,
                     slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=2)
            print("saved ->", args.json)
        return

    if not args.rps and not args.concurrency:
        args.rps = 4.0
        print("no mode given; defaulting to open loop at 4 req/s")

    results = run_load(args.url, args.model, rps=args.rps, concurrency=args.concurrency,
                       duration=args.duration, prompt_tokens=args.prompt_tokens,
                       max_tokens=args.max_tokens)
    s = summarize(results, slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot)
    print(s)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ttft": s.ttft, "tpot": s.tpot, "e2e": s.e2e,
                       "req_per_s": s.request_throughput,
                       "out_tok_per_s": s.output_throughput,
                       "goodput": s.goodput, "failed": s.failed}, f, indent=2)
        print("saved ->", args.json)


if __name__ == "__main__":
    main()
