"""Start a vLLM server as a background process from a notebook cell.

The notebook adaptation of a serving lab: the server runs detached with its log
on disk, cells poll `/health` and `/metrics`, and the log is a cell away when
startup fails (which, the first time, it will).
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request

from .env import detect


def health_ok(base_url="http://localhost:8000", timeout=2.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=timeout) as r:  # noqa: S310
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def build_command(model, port=8000, *, dtype=None, max_model_len=2048, gpu_memory_utilization=0.90,
                  max_num_seqs=None, enforce_eager=True, quantization=None, extra=()):
    """Assemble a `vllm serve` command with the flags a small card needs.

    `enforce_eager=True` by default: CUDA-graph capture costs a minute of
    startup and a chunk of VRAM, and on a 16 GB card you would rather have the
    memory and the faster iteration. Turn it off when you are measuring
    best-case latency.
    """
    env = detect()
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--dtype", dtype or env.vllm_dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        # Keep per-request log lines but truncate the prompt echo: the log is
        # where you watch the death spiral in text, and full prompts drown it.
        "--max-log-len", "80",
    ]
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if quantization:
        cmd += ["--quantization", quantization]
    cmd += list(extra)
    return cmd


class VLLMServer:
    """A background `vllm serve`, with its log where you can read it."""

    def __init__(self, model, port=8000, log_path=None, **kw):
        self.model = model
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.log_path = log_path or f"runs/vllm-{port}.log"
        self.cmd = build_command(model, port, **kw)
        self.proc = None

    def start(self, wait=True, timeout=900):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        if health_ok(self.base_url):
            print(f"a server is already serving on :{self.port} — reusing it "
                  "(restart the runtime if you changed flags)")
            return self
        print("$ " + " ".join(shlex.quote(c) for c in self.cmd))
        log = open(self.log_path, "ab", buffering=0)
        # start_new_session detaches it from the notebook's process group, so a
        # KeyboardInterrupt in a cell does not take the server down with it.
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if wait:
            self.wait_ready(timeout=timeout)
        return self

    def wait_ready(self, timeout=900, poll=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if health_ok(self.base_url):
                print(f"\nready in {time.time() - t0:.0f}s  ->  {self.base_url}")
                return True
            if self.proc and self.proc.poll() is not None:
                print(f"\nserver exited with code {self.proc.returncode}. Last log lines:\n")
                print(self.tail(40))
                raise RuntimeError("vLLM failed to start — read the log above")
            print(f"\rloading… {time.time() - t0:.0f}s", end="")
            time.sleep(poll)
        raise TimeoutError(f"server not ready after {timeout}s; see {self.log_path}")

    def tail(self, n=40) -> str:
        try:
            with open(self.log_path, errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except FileNotFoundError:
            return "(no log yet)"

    def stop(self, grace=10):
        if not self.proc:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self.proc.wait(timeout=grace)
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.proc = None
        print("server stopped (VRAM released — check with nvidia-smi)")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
