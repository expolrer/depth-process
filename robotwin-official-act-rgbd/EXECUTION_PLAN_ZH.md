# Official ACT RGB-D 唯一执行计划

本文与 `execution_plan.json` 是本项目后续训练、迁移和评测的唯一执行依据。README 负责解释，
JSON 负责约束脚本。任何实验不得跳阶段；若需要改变架构、任务、seed、训练预算或准入门槛，必须先
修改这两个文件并提交 Git，再启动作业。

## 1. 计算资源分工

| 环境 | 允许工作 | GPU 规则 |
| --- | --- | --- |
| 56 H100 服务器 | 数据生成、开发、正式训练、validation、在线评测、导出 checkpoint | GPU0-3 长期训练/评测；GPU5-7 仅用到北京时间 2026-09-15 07:00；GPU4 不使用 |
| AutoDL RTX 4090 D | RoboTwin 单环境、batch 1、固定 seed 在线推理 | 24GB 显存；单卡实例的逻辑 `cuda:0` 合法；不得在评测机继续训练或改变权重 |

56 H100 训练、AutoDL 4090 D 评测是可行的。checkpoint 只保存标准 PyTorch `state_dict`，加载时先映射到
CPU；不得把 CUDA graph、优化器 CUDA tensor 或 H100 专属 FP8 权重作为部署依赖。正式评测固定
FP32 推理；若后续启用 FP16/BF16，必须另建精度一致性实验，不能替换主结果。

## 2. AutoDL 最低配置

服务器现有 RoboTwin + SAPIEN 单进程实测占用约 7.0-7.6 GiB 显存，这只是当前场景的观测值，
不是可租用配置下限。渲染、视频、任务资产和驱动会产生额外峰值，因此采用以下准入标准：

| 评测范围 | 项目准入下限 | 推荐配置 | 说明 |
| --- | --- | --- | --- |
| `ACT0-5/7` | NVIDIA CUDA GPU，16 GiB VRAM；8 vCPU；32 GiB RAM；100 GiB 可用 SSD | 24 GiB VRAM；16 vCPU；64 GiB RAM；200 GiB SSD | 单环境、batch 1；16GB 需先通过 20-rollout 显存预检 |
| `ACT6_LINGBOT_DEPTH` | 24 GiB VRAM；16 vCPU；64 GiB RAM；150 GiB 可用 SSD | 24 GiB 或更高；64 GiB RAM；250 GiB SSD | 同时加载 SAPIEN、ACT、ViT-L/14 与 xFormers，16GB 不进入正式队列 |

8GB 显卡不作为本项目正式评测设备。普通架构可选择 16GB 消费卡；当前指定的 RTX 4090 D 具有
24GB 显存，满足全部 ACT0-7 的项目准入线。租机前还需确认 Linux、NVIDIA 驱动、Vulkan/EGL、CUDA
与容器内 SAPIEN 可用，而不只看显存数字。

## 3. 跨机器产物契约

56 H100 每次正式训练结束后，输出目录必须至少含有：

- `policy_best.ckpt`，这是训练完成后唯一保留的模型权重；
- `dataset_stats.pkl`、`config.json`、`metrics.jsonl`；
- `training_complete.json` 和 `artifact_manifest.json`，记录完成状态、SHA256、字节数、源码 commit 和生成时间；
- 项目根目录对应版本的 `UPSTREAM_LOCK.json`、`experiment_matrix.json` 和环境锁文件。

使用 `python scripts/artifact_manifest.py create CHECKPOINT_DIR` 生成清单。传到 AutoDL 后必须先运行
`python scripts/artifact_manifest.py verify CHECKPOINT_DIR`。任一文件缺失、大小不符或 SHA256 不一致，
`run_eval.sh` 会拒绝评测。评测只读 checkpoint，不得原地覆盖训练产物。

训练过程中只保留包含模型、优化器和随机状态的 `training_last.pt`。达到6000 epochs 后自动删除该文件、
`policy_last.ckpt` 和所有周期权重，只保留最终用于评测的 `policy_best.ckpt`。

若代码目录不是 Git checkout，部署时必须把对应的 40 位 Git commit 写入项目根目录 `.source_commit`；
缺少该标记时 artifact 导出会失败，禁止产生无法追溯源码的正式 checkpoint。

## 4. 当前官方6000轮训练与评测：Q0

当前目标收缩为尽快回答：在官方 ACT 中直接加入深度图后，任务成功率是否出现值得继续投入的变化。

- 任务只用对深度误差敏感的 `stack_blocks_two/depth_master_clean`。
- 同时训练 `ACT0_RGB`、`ACT1_EARLY_RGBD`、`ACT2_DUAL_SHARED`、`ACT3_DUAL_PER_VIEW`、
  `ACT4_XYZMAP`、`ACT5_POINT_TOKENS`、`ACT6_LINGBOT_DEPTH`、`ACT7_DEPTH_TRANSFORMER`。
- 八个模型使用完全相同的数据、划分、相机顺序、6000 epochs、batch 8 和 seed 0。
- `ACT0` 是官方 RGB + joint 基线；其余模型只使用 clean metric depth，不输入 validity mask。
- 每个 epoch 按官方方式在验证 episode 中随机取帧，按验证总 loss 选择最佳权重。
- 每100 epochs 原子保存续训状态；GPU5-7 到时优雅退出，未完成作业由 GPU0-3 续训。
- 八个训练全部完成后，立即以官方 temporal aggregation 和同一100-seed列表开始在线评测。

本轮 validation prior-action L1 只用于检查收敛并选择 checkpoint，不能代替 RoboTwin 在线任务成功率。
八架构训练完成后再冻结候选和评测计划，不能用训练 loss 直接宣称深度有效或无效。

## 5. 后续严格状态机

1. `Q0_OFFICIAL6000_TRAIN_THEN_EVAL`：当前阶段，同场景按官方预算训练并评测八种架构。
2. `Q1_CONFIRM_DEPTH_EFFECT`：同一对照扩展为 2000 epochs、100 episodes，确认方向性结果。
3. `P1_PRIMARY_SCREEN`：确认值得继续后，再按固定预算比较 `ACT2/4/5`。
4. `P2_EXTENDED_SCREEN`：仅在 P1 没有明确赢家或容量诊断需要时比较 `ACT3/6/7`。
5. `P3_MULTI_TASK`：胜出前端扩展到其他任务和三训练 seed。
6. `P4_DEPTH_VALIDITY_ROBUSTNESS`：最后再验证 validity、zero/shuffle、clean/noisy/processed 交叉矩阵。
7. `P5_VLA_TRANSFER`：只有小模型胜出架构才迁移到 PI0.5、InternVLA 或 LingBot-VLA。

阶段晋级必须先提交上一阶段的结果和原始日志索引。当前阶段为 `Q0_OFFICIAL6000_TRAIN_THEN_EVAL`；
`scripts/workflow_guard.py` 只允许八个架构在 `stack_blocks_two` 上训练。

## 6. 每个实验的固定顺序

1. Git 工作区干净，记录 commit；运行 `scripts/preflight.sh` 与 parity/data contract 检查。
2. 从 `experiment_matrix.json` 选择唯一实验 ID，固定 task/config/train seed/eval seed。
3. 56 H100 训练；validation 只用于按完整 frame grid 的 prior-action L1 选 `policy_best.ckpt`。
4. 生成并验证 artifact manifest，再传输到 AutoDL 4090 D；传输后再次验证。
5. Q0 只完成八架构训练和 validation 汇总；后续在线评测先做独立资源检查。
6. Q1 及之后使用冻结的 100-seed 列表；同一对比组必须使用同一环境 commit 和 seed 顺序。
7. 汇总成功率、Wilson 区间、阶段成功率、动作越界/跳变、运行时和峰值显存，提交 Git。
8. 只有阶段门槛通过后，才把 `execution_plan.json` 的下一阶段标为 active 并提交 Git。

## 7. 禁止项

- 禁止把 Q0 的 validation loss 写成最终“深度有效/无效”结论。
- 禁止把旧 FairACT checkpoint 或 smoke 结果混入 OfficialACTRGBD 主表。
- 禁止用不同数据、相机顺序、评测 seed 或 checkpoint 选择规则比较架构。
- 禁止把 attention map、训练 loss 或 20-rollout smoke 当作任务成功率证据。
- 禁止在 Q0 阶段加入 validity、深度反事实、噪声/修复方法，避免扩大变量。
- 禁止直接修改运行中的服务器副本；所有计划或代码变更先提交 Git，再部署对应 commit。
