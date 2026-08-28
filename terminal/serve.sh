#!/usr/bin/env bash
# Start vLLM with flags that suit the card in front of you.
#
# The dtype line is the one that matters: Turing (T4, sm_75) has no bf16, and
# vLLM reads bfloat16 out of most modern configs, so it refuses to start unless
# you force half.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-2048}"
UTIL="${UTIL:-0.90}"
MAX_SEQS="${MAX_SEQS:-}"
LOG="${LOG:-runs/vllm.log}"

mkdir -p "$(dirname "$LOG")"

CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo 0.0)"
MAJOR="${CAP%%.*}"
if [ "${MAJOR:-0}" -lt 8 ] 2>/dev/null; then
  DTYPE="half"
  echo "-- compute capability $CAP (pre-Ampere): forcing --dtype half, no FP8 available"
else
  DTYPE="auto"
  echo "-- compute capability $CAP: bf16 available"
fi

ARGS=(serve "$MODEL"
      --port "$PORT"
      --dtype "$DTYPE"
      --max-model-len "$MAX_LEN"
      --gpu-memory-utilization "$UTIL"
      --max-log-len 80
      --enforce-eager)
[ -n "$MAX_SEQS" ] && ARGS+=(--max-num-seqs "$MAX_SEQS")

echo "== vllm ${ARGS[*]}"
echo "== log: $LOG"
exec vllm "${ARGS[@]}" 2>&1 | tee -a "$LOG"
