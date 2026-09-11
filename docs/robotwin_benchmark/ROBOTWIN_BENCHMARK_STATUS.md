# RoboTwin RGB-D Benchmark Status

Snapshot: `2026-09-11T14:24:20+08:00`

This is a live experiment snapshot. Success rates are published only after a full 100-rollout evaluation completes.

## Progress

- Fully evaluated: **10/36**
- Trained, evaluation pending: **13**
- Training now: **3**
- Evaluation now: **0**
- Pending: **10**
- Project scheduler GPUs: **4, 5, 6 only**

## Success Rate

| Variant | pick_dual_bottles | stack_blocks_two | handover_mic | place_a2b_left | place_a2b_right | pick_diverse_bottles |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 28% | 35% | 89% | 0% | 2% | 5% |
| A1 | - | - | - | - | - | - |
| A2 | - | - | - | - | - | - |
| A3 | - | - | - | - | - | - |
| A4 | 61% | - | - | 41% | 47% | - |
| A5 | - | - | - | 0% | - | - |

## Architectures

- **A0**: RGB + joint ACT baseline
- **A1**: RGB-D four-channel early concatenation
- **A2**: RGB ResNet + depth CNN + joint, token-level late fusion
- **A3**: RGB encoder + LingBot-Depth/ViT depth tokens
- **A4**: Fused point cloud + DP3
- **A5**: RGB tokens + depth tokens + joint-conditioned action DiT

## Active Workers

- **GPU4**: `2026-09-11T14:23:25+08:00 WAITING_EXISTING gpu=4 task=A4-train-handover_mic pid=3812073`
- **GPU5**: `2026-09-11T14:23:25+08:00 WAITING_EXISTING gpu=5 task=A4-train-stack_blocks_two pid=1836128`
- **GPU6**: `2026-09-11T14:23:25+08:00 WAITING_EXISTING gpu=6 task=A4-train-pick_diverse_bottles pid=733881`

## Interpretation

A0 is the completed RGB-only control. A4 currently has the strongest completed depth-enabled results, but cross-architecture conclusions must wait until A1-A5 use the same task seeds and all 100-rollout evaluations finish.
