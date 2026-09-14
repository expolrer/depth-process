#!/usr/bin/env bash
set -euo pipefail

repo=/ssd/hhw/depth-model/repos/official-act-rgbd
python=/ssd/hhw/depth-model/envs/aloha/bin/python

bash -n "$repo/scripts/run_train.sh"
bash -n "$repo/scripts/run_eval.sh"
"$python" -c "import json, pathlib; [json.loads(p.read_text()) for p in pathlib.Path('$repo').glob('*.json')]; print('JSON_OK')"
"$python" "$repo/scripts/workflow_guard.py" train ACT0_RGB stack_blocks_two
if "$python" "$repo/scripts/workflow_guard.py" train ACT2_DEPTH_CNN stack_blocks_two; then
  printf 'workflow guard failed to reject an out-of-stage variant\n' >&2
  exit 1
else
  printf 'WORKFLOW_REJECTION_OK\n'
fi

scratch=$(mktemp -d /tmp/official-act-artifact-test.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
for name in policy_best.ckpt policy_last.ckpt dataset_stats.pkl config.json metrics.jsonl; do
  printf 'test-%s\n' "$name" > "$scratch/$name"
done
"$python" "$repo/scripts/artifact_manifest.py" create "$scratch" --repo "$repo"
"$python" "$repo/scripts/artifact_manifest.py" verify "$scratch"
printf 'EXECUTION_PLAN_VALIDATION_PASSED\n'
