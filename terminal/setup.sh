#!/usr/bin/env bash
# One-shot provisioning for a rented GPU box. Idempotent.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
VENV="${VENV:-$HOME/.venv/servlab}"

echo "== python env: $VENV"
python3 -m venv "$VENV" 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -q --upgrade pip
pip install -q vllm httpx matplotlib pandas huggingface_hub
pip install -q -e "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== pre-downloading $MODEL (so the lab does not start with a 5-minute wait)"
python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("cached")
PY

command -v tmux >/dev/null || echo "!! tmux not installed: apt-get install -y tmux"
command -v jq   >/dev/null || echo "-- jq not installed (optional, nicer curl output)"

echo
echo "ready. next:  ./lab.sh"
