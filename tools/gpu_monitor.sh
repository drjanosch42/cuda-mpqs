#!/usr/bin/env bash
# gpu_monitor.sh — GPU telemetry recorder for CUDA-MPQS benchmarking.
# Records power, memory, utilization, and temperature at 3-second intervals.
# Works on discrete GPUs (nvidia-smi) and Jetson Orin (tegrastats fallback for power).
#
# Usage:
#   ./tools/gpu_monitor.sh -o telemetry.csv -- ./build/tests/cuda-mpqs --verbose
#   ./tools/gpu_monitor.sh -o telemetry.csv -p <PID>

set -euo pipefail

INTERVAL=3
OUTPUT=""
TARGET_PID=""
CMD_ARGS=()
IS_JETSON=false
TEGRA_TMPFILE=""

usage() {
    cat <<'EOF'
Usage:
  gpu_monitor.sh [OPTIONS] -- COMMAND [ARGS...]
  gpu_monitor.sh [OPTIONS] -p PID

Options:
  -o FILE   Write CSV to FILE (default: stdout)
  -i SECS   Polling interval in seconds (default: 3)
  -p PID    Monitor an existing process
  -h        Show this help
EOF
    exit "${1:-0}"
}

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o) OUTPUT="$2"; shift 2 ;;
        -i) INTERVAL="$2"; shift 2 ;;
        -p) TARGET_PID="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        --) shift; CMD_ARGS=("$@"); break ;;
        *)  echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$TARGET_PID" && ${#CMD_ARGS[@]} -eq 0 ]]; then
    echo "Error: provide a command after -- or a PID with -p" >&2
    usage 1
fi

# --- Jetson detection ---
if [[ -f /etc/nv_tegra_release ]] || command -v tegrastats &>/dev/null; then
    IS_JETSON=true
fi

# --- Output setup ---
if [[ -n "$OUTPUT" ]]; then
    exec 3>"$OUTPUT"
else
    exec 3>&1   # CSV to stdout
fi

# --- CSV header ---
echo "timestamp,power_w,mem_used_mb,mem_total_mb,gpu_util_pct,gpu_temp_c" >&3

# --- Tegrastats helper (Jetson power fallback) ---
# Parses VDD_GPU_SOC or VDD_CPU_GPU_CV power rail from tegrastats output.
# Returns milliwatts, or empty string on failure.
parse_tegra_power_mw() {
    local line="$1"
    local mw=""
    # Try VDD_GPU_SOC first (Orin Nano / NX), then VDD_CPU_GPU_CV (AGX Orin)
    for rail in VDD_GPU_SOC VDD_CPU_GPU_CV; do
        mw=$(echo "$line" | grep -oP "${rail}\s+\K[0-9]+" | head -1) && break
    done
    echo "${mw:-}"
}

start_tegrastats() {
    TEGRA_TMPFILE=$(mktemp /tmp/gpu_monitor_tegra.XXXXXX)
    tegrastats --interval $((INTERVAL * 1000)) --logfile "$TEGRA_TMPFILE" &
    TEGRA_PID=$!
}

stop_tegrastats() {
    if [[ -n "${TEGRA_PID:-}" ]]; then
        kill "$TEGRA_PID" 2>/dev/null || true
        wait "$TEGRA_PID" 2>/dev/null || true
    fi
    [[ -n "${TEGRA_TMPFILE:-}" ]] && rm -f "$TEGRA_TMPFILE"
}

# Read latest tegrastats power in watts (float).
read_tegra_power_w() {
    [[ ! -f "$TEGRA_TMPFILE" ]] && echo "" && return
    local last_line
    last_line=$(tail -1 "$TEGRA_TMPFILE" 2>/dev/null) || true
    local mw
    mw=$(parse_tegra_power_mw "$last_line")
    if [[ -n "$mw" && "$mw" != "0" ]]; then
        awk "BEGIN { printf \"%.2f\", $mw / 1000.0 }"
    else
        echo ""
    fi
}

# --- nvidia-smi query ---
query_gpu() {
    nvidia-smi --query-gpu=timestamp,power.draw,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo ""
}

# --- Accumulator variables for summary ---
declare -a ALL_POWER=()
declare -a ALL_MEM=()
declare -a ALL_UTIL=()
SAMPLE_COUNT=0

# --- Cleanup ---
cleanup() {
    $IS_JETSON && stop_tegrastats
    exec 3>&-
}
trap cleanup EXIT

# --- Start Jetson tegrastats if needed ---
if $IS_JETSON; then
    start_tegrastats
fi

# --- Launch wrapped command if provided ---
if [[ ${#CMD_ARGS[@]} -gt 0 ]]; then
    "${CMD_ARGS[@]}" &
    TARGET_PID=$!
fi

START_TIME=$(date +%s)

# --- Polling loop ---
while kill -0 "$TARGET_PID" 2>/dev/null; do
    RAW=$(query_gpu)
    if [[ -z "$RAW" ]]; then
        sleep "$INTERVAL"
        continue
    fi

    # Parse nvidia-smi CSV: "2026/03/25 12:00:00.000, 150.00, 4096, 16384, 85, 62"
    IFS=',' read -r ts power mem_used mem_total gpu_util gpu_temp <<< "$RAW"

    # Trim whitespace
    ts=$(echo "$ts" | xargs)
    power=$(echo "$power" | xargs)
    mem_used=$(echo "$mem_used" | xargs)
    mem_total=$(echo "$mem_total" | xargs)
    gpu_util=$(echo "$gpu_util" | xargs)
    gpu_temp=$(echo "$gpu_temp" | xargs)

    # Handle [N/A] or [Not Supported] fields
    [[ "$power" == *"N/A"* || "$power" == *"Not Supported"* ]] && power=""
    [[ "$mem_used" == *"N/A"* || "$mem_used" == *"Not Supported"* ]] && mem_used=""
    [[ "$mem_total" == *"N/A"* || "$mem_total" == *"Not Supported"* ]] && mem_total=""
    [[ "$gpu_util" == *"N/A"* || "$gpu_util" == *"Not Supported"* ]] && gpu_util=""
    [[ "$gpu_temp" == *"N/A"* || "$gpu_temp" == *"Not Supported"* ]] && gpu_temp=""

    # Jetson power fallback via tegrastats
    if [[ -z "$power" ]] && $IS_JETSON; then
        power=$(read_tegra_power_w)
    fi

    # Emit CSV row
    echo "${ts},${power},${mem_used},${mem_total},${gpu_util},${gpu_temp}" >&3

    # Accumulate for summary
    [[ -n "$power" ]]    && ALL_POWER+=("$power")
    [[ -n "$mem_used" ]] && ALL_MEM+=("$mem_used")
    [[ -n "$gpu_util" ]] && ALL_UTIL+=("$gpu_util")
    ((SAMPLE_COUNT++)) || true

    sleep "$INTERVAL"
done

# --- Wait for wrapped command exit code ---
EXIT_CODE=0
if [[ ${#CMD_ARGS[@]} -gt 0 ]]; then
    wait "$TARGET_PID" || EXIT_CODE=$?
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# --- Summary ---
summary() {
    local label="$1"; shift
    local -n arr=$1
    if [[ ${#arr[@]} -eq 0 ]]; then
        echo "  $label: N/A"
        return
    fi
    local sum=0 peak=0 avg
    for v in "${arr[@]}"; do
        sum=$(awk "BEGIN { printf \"%.2f\", $sum + $v }")
        peak=$(awk "BEGIN { print ($v > $peak) ? $v : $peak }")
    done
    avg=$(awk "BEGIN { printf \"%.2f\", $sum / ${#arr[@]} }")
    echo "  $label: avg=$avg, peak=$peak"
    # Return avg via global for energy calc
    if [[ "$label" == "Power (W)" ]]; then
        _AVG_POWER="$avg"
    fi
}

echo "" >&2
echo "=== GPU Telemetry Summary ===" >&2
echo "  Duration: ${DURATION}s  (${SAMPLE_COUNT} samples @ ${INTERVAL}s interval)" >&2

_AVG_POWER=""
summary "Power (W)" ALL_POWER >&2
summary "Memory (MB)" ALL_MEM >&2
summary "GPU Util (%)" ALL_UTIL >&2

if [[ -n "$_AVG_POWER" && "$_AVG_POWER" != "0" ]]; then
    energy=$(awk "BEGIN { printf \"%.4f\", $_AVG_POWER * $DURATION / 3600.0 }")
    echo "  Energy estimate: ${energy} Wh" >&2
else
    echo "  Energy estimate: N/A" >&2
fi

if [[ ${#CMD_ARGS[@]} -gt 0 ]]; then
    echo "  Command exit code: $EXIT_CODE" >&2
fi
echo "=============================" >&2

exit "$EXIT_CODE"
