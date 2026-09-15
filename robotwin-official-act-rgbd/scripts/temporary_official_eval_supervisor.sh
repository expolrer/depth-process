#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'Usage: %s GPU PREFERRED_VARIANT DEADLINE_EPOCH\n' "$0" >&2
  exit 2
fi

gpu="$1"
preferred="$2"
deadline="$3"
root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
eval_queue="$root/scheduler/official_act_official6000_eval"
state="$root/scheduler/official_act_eval_until_0700"
[[ "$gpu" =~ ^[1-3]$ ]] || { printf 'Temporary evaluation is restricted to GPU1-3\n' >&2; exit 2; }
[[ "$deadline" =~ ^[0-9]+$ ]] || { printf 'Deadline must be a Unix epoch\n' >&2; exit 2; }
mkdir -p "$state" "$eval_queue/claims" "$eval_queue/done"
exec 9>"$state/gpu${gpu}.lock"
flock -n 9 || { printf 'GPU%s temporary evaluation supervisor already running\n' "$gpu"; exit 0; }
printf '%s\n' "$$" > "$state/gpu${gpu}.pid"

cleanup_stale_claims() {
  local claim owner
  for claim in "$eval_queue"/claims/*; do
    [[ -d "$claim" ]] || continue
    owner="$(cat "$claim/worker_pid" 2>/dev/null || true)"
    if [[ -z "$owner" || ! -r "/proc/$owner/cmdline" ]]; then
      rm -rf "$claim"
    fi
  done
}

while (( $(date +%s) < deadline )); do
  remaining=$((deadline - $(date +%s)))
  printf '%s GPU%s starts/continues official evaluation; deadline=%s remaining_s=%s\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$(date -d "@$deadline" --iso-8601=seconds)" "$remaining"
  set +e
  timeout --signal=TERM --kill-after=120s "${remaining}s" \
    "$repo/scripts/official_eval_worker.sh" "$gpu" "$preferred"
  status="$?"
  set -e
  cleanup_stale_claims
  if (( $(date +%s) >= deadline )); then
    break
  fi
  printf '%s GPU%s evaluation worker exited %s before deadline; restarting\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$status"
  sleep 15
done

printf '{"status":"deadline_reached","gpu":%s,"deadline":"%s","stopped_at":"%s"}\n' \
  "$gpu" "$(date -d "@$deadline" --iso-8601=seconds)" "$(date --iso-8601=seconds)" \
  > "$state/gpu${gpu}_complete.json"
printf '%s GPU%s temporary official evaluation stopped at deadline\n' "$(date --iso-8601=seconds)" "$gpu"
