#!/usr/bin/env bash
# Four panes, which is the whole rig:
#
#   +---------------------+---------------------+
#   | server log          | /metrics, 1s        |
#   +---------------------+---------------------+
#   | load generator      | scratch: curl, htop |
#   +---------------------+---------------------+
#
# Left column is what you control, right column is what you read. In an
# interview you will be talking while looking at the right column.
set -euo pipefail

SESSION="${SESSION:-servlab}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' exists — attaching (tmux kill-session -t $SESSION to reset)"
  exec tmux attach -t "$SESSION"
fi

tmux new-session  -d -s "$SESSION" -c "$HERE" -n lab
tmux split-window -h -t "$SESSION:lab" -c "$HERE"
tmux split-window -v -t "$SESSION:lab.0" -c "$HERE"
tmux split-window -v -t "$SESSION:lab.1" -c "$HERE"

tmux send-keys -t "$SESSION:lab.0" './serve.sh' C-m
tmux send-keys -t "$SESSION:lab.1" 'sleep 45; ./watch_metrics.sh' C-m
tmux send-keys -t "$SESSION:lab.2" '# ./load.py --rps 1 --duration 90    then    --rps 6' ''
tmux send-keys -t "$SESSION:lab.3" 'watch -n2 nvidia-smi' C-m

tmux select-pane -t "$SESSION:lab.2"
exec tmux attach -t "$SESSION"
