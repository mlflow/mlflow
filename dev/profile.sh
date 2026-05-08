#!/usr/bin/env bash
# Sample CPU and memory usage while running a command, then post mermaid
# xychart-beta line charts to GITHUB_STEP_SUMMARY (or stdout if running
# locally outside GitHub Actions).
#
# Usage: dev/profile.sh <command> [args...]
#
# Environment overrides:
#   PROFILE_INTERVAL   sampling interval in seconds (default: 2)
#   PROFILE_DATA_FILE  path to write raw CSV samples (default: /tmp/profile-metrics.csv)
#
# Linux-only: reads /proc/stat and /proc/meminfo.

set -uo pipefail

SAMPLE_FILE=${PROFILE_DATA_FILE:-/tmp/profile-metrics.csv}
SAMPLE_INTERVAL=${PROFILE_INTERVAL:-2}
SUMMARY_FILE=${GITHUB_STEP_SUMMARY:-/dev/stdout}

if [ ! -r /proc/stat ] || [ ! -r /proc/meminfo ]; then
  echo "profile.sh: /proc/stat or /proc/meminfo unavailable; running command without sampling" >&2
  exec "$@"
fi

(
  prev_total=0
  prev_idle=0
  while true; do
    read _ u n s i io irq si st _ < /proc/stat
    t=$((u + n + s + i + io + irq + si + st))
    if [ "$prev_total" -ne 0 ]; then
      dt=$((t - prev_total))
      di=$(((i + io) - prev_idle))
      cpu=$((dt > 0 ? 100 * (dt - di) / dt : 0))
    else
      cpu=0
    fi
    prev_total=$t
    prev_idle=$((i + io))
    mem=$(awk '/MemTotal/{T=$2} /MemAvailable/{A=$2} END{print int((T-A)/1024)}' /proc/meminfo)
    printf '%s,%s,%s\n' "$(date +%s)" "$cpu" "$mem"
    sleep "$SAMPLE_INTERVAL"
  done
) >"$SAMPLE_FILE" 2>&1 &
sampler_pid=$!

cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true

  if [ ! -s "$SAMPLE_FILE" ]; then
    return 0
  fi

  local sampled=/tmp/profile-sampled.csv
  local n step total cpu_list mem_list
  n=$(wc -l <"$SAMPLE_FILE")
  step=$((n / 30 + 1))
  awk -v step="$step" 'NR % step == 0' "$SAMPLE_FILE" >"$sampled"
  total=$(awk -F, 'NR==1{first=$1} END{print $1-first}' "$SAMPLE_FILE")
  cpu_list=$(awk -F, '{printf "%s,",$2}' "$sampled" | sed 's/,$//')
  mem_list=$(awk -F, '{printf "%s,",$3}' "$sampled" | sed 's/,$//')

  {
    echo '## CPU usage (%)'
    echo '```mermaid'
    echo 'xychart-beta'
    echo "  x-axis \"Elapsed (s)\" 0 --> $total"
    echo '  y-axis "CPU %" 0 --> 100'
    echo "  line [$cpu_list]"
    echo '```'
    echo
    echo '## Memory used (MB)'
    echo '```mermaid'
    echo 'xychart-beta'
    echo "  x-axis \"Elapsed (s)\" 0 --> $total"
    echo '  y-axis "Memory MB"'
    echo "  line [$mem_list]"
    echo '```'
  } >>"$SUMMARY_FILE"
}
trap cleanup EXIT

"$@"
