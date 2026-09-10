#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 DATA WORKSPACE TASK_YAML [extra auto-labeler-v3 arguments...]" >&2
  exit 2
fi

data=$1
workspace=$2
task=$3
shift 3

exec auto-labeler-v3 run \
  --data "$data" \
  --workspace "$workspace" \
  --task "$task" \
  "$@"
