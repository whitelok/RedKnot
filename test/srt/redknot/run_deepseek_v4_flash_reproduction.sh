#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$script_dir/../../.." && pwd -P)
venv=${REDKNOT_VENV:-$repo/.venv_tf5}
lock_file=${REDKNOT_RUN_LOCK:-/tmp/redknot_deepseek_v4_flash_release.lock}

exec 9>"$lock_file"
if ! flock -n 9; then
  printf 'ERROR: another RedKnot release wrapper already owns %s\n' \
    "$lock_file" >&2
  printf '%s\n' \
    'Wait for it to finish; the eight GPUs cannot run two TP8 instances.' >&2
  exit 75
fi

active_release_pids=$(pgrep -f '[b]enchmark_RedKnot_DeepSeekV4Flash.py' || true)
active_worker_pids=$(pgrep -f '[b]enchmark_dsv4_redknot_http.py' || true)
if [[ -n "$active_release_pids" || -n "$active_worker_pids" ]]; then
  printf '%s\n' \
    'ERROR: another RedKnot benchmark is already using this TP8 server.' \
    "release_pids=${active_release_pids:-none}" \
    "worker_pids=${active_worker_pids:-none}" \
    'Wait for its result and automatic holder restoration before retrying.' >&2
  exit 75
fi

cd "$script_dir"
if [[ -x "$venv/bin/python" ]]; then
  printf '[setup] validating existing environment: %s\n' "$venv"
  ./setup_deepseek_v4_flash_env.sh --check-only
else
  printf '[setup] creating pinned environment: %s\n' "$venv"
  ./setup_deepseek_v4_flash_env.sh
fi

source ./environment-deepseek-v4-flash.env
printf '[run] python=%s\n' "$(command -v python)"
if [[ $# -eq 0 ]]; then
  printf '%s\n' \
    '[run] default suites=256K + 440K, 15 cases each (10 short + 5 long)'
fi
exec python benchmark_RedKnot_DeepSeekV4Flash.py "$@"
