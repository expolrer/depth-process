# RoboTwin Benchmark Operations

These scripts mirror the live benchmark scheduler under `/ssd/depth-model` on the 56 server.

- `continuous_gpu_worker_v2.sh` runs a recoverable shared queue on physical GPUs 4-6 only.
- `continuous_supervisor_v2.sh` reports completion for the three allowed workers.
- `precompute_a3_tokens_v2.sh` creates LingBot depth tokens with three shards on GPUs 4-6.
- `export_robotwin_benchmark_status.py` exports the live JSON and Markdown snapshots published in `docs/robotwin_benchmark/`.

The scheduler preserves completed 30k checkpoints, retries failed evaluations, and does not schedule project work on GPUs 0-3 or GPU7.

```bash
python scripts/robotwin_benchmark/export_robotwin_benchmark_status.py \
  --root /ssd/depth-model \
  --output-json docs/robotwin_benchmark/robotwin_benchmark_status.json \
  --output-md docs/robotwin_benchmark/ROBOTWIN_BENCHMARK_STATUS.md
```
