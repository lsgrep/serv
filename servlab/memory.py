"""Training-memory napkin math, and an OOM postmortem you can actually read.

Lab 4's shape: predict the budget, run the fine-tune, catch the OOM, dump a
snapshot, then explain the gap between prediction and reality. `torch.cuda`'s
memory history is the tool that turns "it OOMed" into "the optimizer state and
the activation checkpoints collided at step 3".
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

GIB = 1024**3


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------


@dataclass
class TrainingBudget:
    weights: float = 0.0
    gradients: float = 0.0
    optimizer: float = 0.0
    activations: float = 0.0
    overhead: float = 0.0

    @property
    def total(self) -> float:
        return self.weights + self.gradients + self.optimizer + self.activations + self.overhead

    def __str__(self):
        rows = [("weights", self.weights), ("gradients", self.gradients),
                ("optimizer", self.optimizer), ("activations", self.activations),
                ("cuda ctx + fragmentation", self.overhead)]
        body = "\n".join(f"  {k:<26} {v / GIB:7.2f} GiB" for k, v in rows)
        return f"{body}\n  {'TOTAL':<26} {self.total / GIB:7.2f} GiB"


def training_budget(params, *, weight_bits=16, trainable_params=None, optimizer="adamw",
                    grad_bits=16, batch=1, seq_len=512, n_layers=32, hidden=4096,
                    activation_checkpointing=True, overhead_gb=1.5) -> TrainingBudget:
    """Where training memory goes.

    Full fine-tuning of a 7B model in fp16 needs ~14 GiB of weights, ~14 GiB of
    gradients and ~84 GiB of Adam state (fp32 moments + master weights) — which
    is why nobody does it on one card. QLoRA changes two of those three terms:
    weights drop to 4-bit, and gradients/optimizer state only exist for the
    adapters, which are well under 1% of the parameters. The activation term is
    the one QLoRA does *not* fix, and it is the one that OOMs you at long
    sequence lengths.
    """
    trainable = params if trainable_params is None else trainable_params
    opt_bytes_per_param = {"adamw": 8, "adamw8bit": 2, "sgd": 4, "none": 0}[optimizer]
    # Activations scale with batch x seq x hidden x layers; the constant depends
    # on how many tensors the layer keeps. ~2 bytes x ~10 tensors is a decent
    # first cut, and checkpointing trades most of it for a recompute pass.
    act_per_token = hidden * n_layers * 2 * (1.5 if activation_checkpointing else 10)
    return TrainingBudget(
        weights=params * weight_bits / 8,
        gradients=trainable * grad_bits / 8,
        optimizer=trainable * opt_bytes_per_param,
        activations=batch * seq_len * act_per_token,
        overhead=overhead_gb * GIB,
    )


def lora_trainable_params(params, *, hidden=4096, n_layers=32, rank=16, targets=4) -> float:
    """Parameter count for LoRA adapters: `2 x r x hidden` per target matrix."""
    return n_layers * targets * 2 * rank * hidden


# --------------------------------------------------------------------------
# Measure
# --------------------------------------------------------------------------


def gpu_memory_report() -> str:
    """Allocated / reserved / peak, in the words the allocator uses.

    `allocated` is live tensors. `reserved` is what the caching allocator holds
    from the driver — the gap between them is fragmentation, and the reason an
    OOM can happen with gigabytes apparently "free".
    """
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"
    a = torch.cuda.memory_allocated() / GIB
    r = torch.cuda.memory_reserved() / GIB
    pa = torch.cuda.max_memory_allocated() / GIB
    pr = torch.cuda.max_memory_reserved() / GIB
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    return (f"allocated {a:5.2f} GiB (peak {pa:5.2f})   "
            f"reserved {r:5.2f} GiB (peak {pr:5.2f})   of {total:.1f} GiB\n"
            f"fragmentation gap {r - a:5.2f} GiB")


def reset_peak():
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def record_memory(path="runs/oom_snapshot.pickle", max_entries=200_000, dump_always=False):
    """Record allocator history and dump it when the block raises.

    Wrap the training loop in this. When it OOMs you get a pickle to load into
    https://pytorch.org/memory_viz — that visualiser is the difference between
    guessing and seeing which allocation crossed the line.

        with record_memory("runs/oom.pickle"):
            trainer.train()

    Download the pickle (`files.download` in Colab) and drag it onto the page.
    """
    import torch

    enabled = torch.cuda.is_available()
    if enabled:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield path
        if enabled and dump_always:
            torch.cuda.memory._dump_snapshot(path)
            print(f"snapshot -> {path}")
    except Exception:
        if enabled:
            torch.cuda.memory._dump_snapshot(path)
            print(f"\nOOM (or other failure). snapshot -> {path}")
            print(gpu_memory_report())
            print("\nLoad it at https://pytorch.org/memory_viz — look for the tallest "
                  "block at the moment of failure and ask what allocated it.")
        raise
    finally:
        if enabled:
            torch.cuda.memory._record_memory_history(enabled=None)


def oom_hints(exc) -> str:
    """The checklist to walk in order, cheapest fix first."""
    return (
        f"{type(exc).__name__}: {str(exc)[:200]}\n\n"
        "Work down this list; each step costs more than the one above it:\n"
        "  1. per-device batch size 1, raise gradient_accumulation_steps to keep the effective batch\n"
        "  2. gradient_checkpointing=True (trades ~30% step time for most of the activation memory)\n"
        "  3. shorter max_seq_length — activations are linear in it, attention worse\n"
        "  4. paged_adamw_8bit optimizer (moments to host memory on spike)\n"
        "  5. lower LoRA rank / fewer target modules\n"
        "  6. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True if the snapshot shows fragmentation\n"
        "     (a large reserved-minus-allocated gap) rather than genuine demand\n"
    )
