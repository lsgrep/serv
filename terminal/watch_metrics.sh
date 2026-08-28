#!/usr/bin/env bash
# The pane you stare at. One screenful, refreshed every second.
#
# Everything here is also available in Grafana, and in an interview you will not
# have Grafana. Being fluent with curl and grep against /metrics is the skill.
set -euo pipefail

URL="${URL:-http://localhost:8000/metrics}"
INTERVAL="${INTERVAL:-1}"

read -r -d '' AWKP <<'AWK' || true
/^vllm:num_requests_running/          { r = $2 }
/^vllm:num_requests_waiting/          { w = $2 }
/^vllm:(gpu_cache|kv_cache)_usage_perc/ { kv = $2 * 100 }
/^vllm:num_preemption[s]?_total/      { p = $2 }
/^vllm:generation_tokens_total/       { g = $2 }
/^vllm:prompt_tokens_total/           { pt = $2 }
/^vllm:request_success_total/         { fin += $2 }
/^vllm:time_to_first_token_seconds_sum/   { ts = $2 }
/^vllm:time_to_first_token_seconds_count/ { tc = $2 }
END {
  printf "running   %6d      waiting   %6d\n", r, w
  printf "kv cache  %5.1f%%      preempt   %6d\n", kv, p
  printf "prompt tok%9d   gen tok %9d\n", pt, g
  printf "finished  %6d      mean TTFT %6.2fs\n", fin, (tc > 0 ? ts / tc : 0)
}
AWK

while true; do
  clear
  date "+%H:%M:%S   $URL"
  echo "-------------------------------------------------"
  curl -s --max-time 2 "$URL" | awk -F' ' "$AWKP" || echo "server not responding"
  echo "-------------------------------------------------"
  echo "waiting climbing + kv at 100% + preempt rising = the death spiral"
  sleep "$INTERVAL"
done
