#!/usr/bin/env bash
# Sample CPU and memory usage while running a command, then post mermaid
# xychart-beta line charts to GITHUB_STEP_SUMMARY (or stdout if running
# locally outside GitHub Actions).
#
# Usage: dev/profile.sh <command> [args...]
#
# Environment overrides:
#   PROFILE_INTERVAL_SECONDS  sampling interval in seconds (default: 2)
#   PROFILE_DATA_FILE         path to write raw CSV samples (default: /tmp/profile-metrics.csv)
#
# Linux-only: reads /proc/stat and /proc/meminfo.

set -uo pipefail

SAMPLE_FILE=${PROFILE_DATA_FILE:-/tmp/profile-metrics.csv}
SAMPLE_INTERVAL_SECONDS=${PROFILE_INTERVAL_SECONDS:-2}
SUMMARY_FILE=${GITHUB_STEP_SUMMARY:-/dev/stdout}

if [ ! -r /proc/stat ] || [ ! -r /proc/meminfo ]; then
  echo "profile.sh: /proc/stat or /proc/meminfo unavailable; running command without sampling" >&2
  exec "$@"
fi

(
  printf 'timestamp,cpu_pct,mem_gb\n'
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
    mem=$(awk '/MemTotal/{T=$2} /MemAvailable/{A=$2} END{printf "%.2f", (T-A)/1024/1024}' /proc/meminfo)
    printf '%s,%s,%s\n' "$(date +%s)" "$cpu" "$mem"
    sleep "$SAMPLE_INTERVAL_SECONDS"
  done
) >"$SAMPLE_FILE" 2>&1 &
sampler_pid=$!

cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true

  if [ ! -s "$SAMPLE_FILE" ]; then
    return 0
  fi

  local data_csv=/tmp/profile-data.csv
  local sampled=/tmp/profile-sampled.csv
  local n step total cpu_list mem_list
  tail -n +2 "$SAMPLE_FILE" >"$data_csv"
  n=$(wc -l <"$data_csv")
  step=$((n / 30 + 1))
  awk -v step="$step" 'NR % step == 0' "$data_csv" >"$sampled"
  total=$(awk -F, 'NR==1{first=$1} END{print $1-first}' "$data_csv")
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
    echo '## Memory used (GB)'
    echo '```mermaid'
    echo 'xychart-beta'
    echo "  x-axis \"Elapsed (s)\" 0 --> $total"
    echo '  y-axis "Memory GB"'
    echo "  line [$mem_list]"
    echo '```'
  } >>"$SUMMARY_FILE"
}
trap cleanup EXIT

"$@"
