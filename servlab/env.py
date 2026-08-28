"""Environment detection, and the T4 warnings that otherwise cost you an evening."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Env:
    in_colab: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: float = 0.0
    capability: tuple = ()
    torch_version: str = ""
    cuda_version: str = ""
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0

    @property
    def supports_bf16(self) -> bool:
        return bool(self.capability) and self.capability >= (8, 0)

    @property
    def supports_fp8(self) -> bool:
        return bool(self.capability) and self.capability >= (8, 9)

    @property
    def vllm_dtype(self) -> str:
        """What to pass to `--dtype`.

        Turing (T4, sm_75) has no bf16, and vLLM reads `torch_dtype: bfloat16`
        straight out of most modern configs — so it errors out at startup unless
        you force half. This one line is the difference between "vLLM is broken
        on Colab" and a working lab.
        """
        return "auto" if self.supports_bf16 else "half"


def _nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                rows.append((parts[0], float(parts[1]) / 1024))
            except ValueError:
                continue
    return rows


def detect() -> Env:
    env = Env(in_colab="google.colab" in sys.modules or os.path.exists("/content"))

    try:
        import torch

        env.torch_version = torch.__version__
        env.cuda_version = getattr(torch.version, "cuda", "") or ""
        if torch.cuda.is_available():
            env.gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            env.gpu_name = props.name
            env.vram_gb = props.total_memory / 1024**3
            env.capability = (props.major, props.minor)
    except ImportError:
        smi = _nvidia_smi()
        if smi:
            env.gpu_count = len(smi)
            env.gpu_name, env.vram_gb = smi[0]
            env.notes.append("torch not installed — GPU details read from nvidia-smi")

    if not env.has_gpu:
        env.warnings.append(
            "No GPU visible. In Colab: Runtime > Change runtime type > T4 GPU, then rerun this cell."
        )
        return env

    if env.capability and env.capability < (8, 0):
        env.warnings.append(
            f"{env.gpu_name} is compute capability {env.capability[0]}.{env.capability[1]} (pre-Ampere): "
            "no bf16 and no FP8. Pass --dtype half to vLLM; skip the FP8 columns."
        )
        env.notes.append("Recent vLLM/kernel releases increasingly assume Ampere+. Odd kernel "
                         "errors here are the hardware, not your code.")
    if not env.supports_fp8:
        env.notes.append("No FP8 on this card (needs Ada sm_89+). The FP8 row in lab 5 waits for an L4.")
    if env.vram_gb and env.vram_gb < 20:
        env.notes.append(
            f"{env.vram_gb:.0f} GB: an 8B model in fp16 (~15 GiB of weights) leaves no room for KV. "
            "Use a 3B model in fp16, or an 8B AWQ quant. Every phenomenon in these labs reproduces at 3B."
        )
    return env


def banner(env=None) -> Env:
    """Print the environment block. Cell 1 of every notebook ends with this."""
    env = env or detect()
    print("=" * 68)
    if env.has_gpu:
        cc = f"sm_{env.capability[0]}{env.capability[1]}" if env.capability else "sm_?"
        print(f"GPU      {env.gpu_name}  x{env.gpu_count}   {env.vram_gb:.1f} GB   {cc}")
        print(f"dtypes   fp16 yes | bf16 {'yes' if env.supports_bf16 else 'NO'} | "
              f"fp8 {'yes' if env.supports_fp8 else 'NO'}   ->  vLLM --dtype {env.vllm_dtype}")
    else:
        print("GPU      none visible")
    if env.torch_version:
        print(f"torch    {env.torch_version}  (cuda {env.cuda_version or 'n/a'})")
    print(f"colab    {'yes' if env.in_colab else 'no'}")
    for w in env.warnings:
        print(f"\n!!  {w}")
    for n in env.notes:
        print(f"\n--  {n}")
    print("=" * 68)
    return env


def mount_drive(path="/content/drive"):
    """Mount Drive for checkpointing. No-op outside Colab so notebooks stay
    runnable on a rented box."""
    try:
        from google.colab import drive
    except ImportError:
        print("not in Colab — checkpoints stay on local disk")
        return None
    drive.mount(path, force_remount=False)
    return path
