# Official ACT RGB-D 唯一执行计划

本文与 `execution_plan.json` 是本项目后续训练、迁移和评测的唯一执行依据。README 负责解释，
JSON 负责约束脚本。任何实验不得跳阶段；若需要改变架构、任务、seed、训练预算或准入门槛，必须先
修改这两个文件并提交 Git，再启动作业。

## 1. 计算资源分工

| 环境 | 允许工作 | GPU 规则 |
| --- | --- | --- |
| 56 H100 服务器 | 数据生成、开发、正式训练、validation、导出 checkpoint | 默认只使用物理 GPU4-7；物理 GPU0 永久禁用；GPU2-3 只在用户明确给出的时间窗内使用 |
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

- `policy_best.ckpt`、`policy_last.ckpt`；
- `dataset_stats.pkl`、`config.json`、`metrics.jsonl`；
- `artifact_manifest.json`，记录上述文件 SHA256、字节数、源码 commit 和生成时间；
- 项目根目录对应版本的 `UPSTREAM_LOCK.json`、`experiment_matrix.json` 和环境锁文件。

使用 `python scripts/artifact_manifest.py create CHECKPOINT_DIR` 生成清单。传到 AutoDL 后必须先运行
`python scripts/artifact_manifest.py verify CHECKPOINT_DIR`。任一文件缺失、大小不符或 SHA256 不一致，
`run_eval.sh` 会拒绝评测。评测只读 checkpoint，不得原地覆盖训练产物。

若代码目录不是 Git checkout，部署时必须把对应的 40 位 Git commit 写入项目根目录 `.source_commit`；
缺少该标记时 artifact 导出会失败，禁止产生无法追溯源码的正式 checkpoint。

## 4. 当前快速验证：Q0

当前目标收缩为尽快回答：在官方 ACT 中直接加入深度图后，任务成功率是否出现值得继续投入的变化。

- 任务只用对深度误差敏感的 `stack_blocks_two/depth_master_clean`。
- 对照只用 `ACT0_RGB` 与 `ACT1_EARLY_RGBD`；两者使用同一数据、划分、相机顺序和训练 seed。
- `ACT1` 将归一化 metric depth 作为每个相机的第 4 通道，不输入 validity mask。
- 两个模型均训练 500 epochs、batch 8、seed 0，并行运行以缩短墙钟时间。
- AutoDL 4090 D 使用同一份固定 seed，先各评测 30 episodes。
- 本轮不做 zero/shuffle depth、不做噪声/处理深度、不做三训练 seed，也不比较其他深度编码器。

30 episodes 只提供方向性证据，不作为最终论文结论。若成功率差值绝对值达到 10 个百分点且两者没有
异常动作差异，则进入 2000 epochs + 100 episodes 的同任务确认；若两者成功率都低于 10%，先排查
训练与部署管线；其余情况直接扩大评测，不能宣称深度有效或无效。

## 5. 后续严格状态机

1. `Q0_RAPID_DEPTH_CHECK`：当前阶段，500 epochs、30 episodes 比较 `ACT0` 与无 validity 的 `ACT1`。
2. `Q1_CONFIRM_DEPTH_EFFECT`：同一对照扩展为 2000 epochs、100 episodes，确认方向性结果。
3. `P1_PRIMARY_SCREEN`：确认值得继续后，再按固定预算比较 `ACT2/4/5`。
4. `P2_EXTENDED_SCREEN`：仅在 P1 没有明确赢家或容量诊断需要时比较 `ACT3/6/7`。
5. `P3_MULTI_TASK`：胜出前端扩展到其他任务和三训练 seed。
6. `P4_DEPTH_VALIDITY_ROBUSTNESS`：最后再验证 validity、zero/shuffle、clean/noisy/processed 交叉矩阵。
7. `P5_VLA_TRANSFER`：只有小模型胜出架构才迁移到 PI0.5、InternVLA 或 LingBot-VLA。

阶段晋级必须先提交上一阶段的结果和原始日志索引。当前阶段为 `Q0_RAPID_DEPTH_CHECK`；
`scripts/workflow_guard.py` 会拒绝其他架构或任务，启动脚本会从计划读取 500 epochs 和 30 episodes。

## 6. 每个实验的固定顺序

1. Git 工作区干净，记录 commit；运行 `scripts/preflight.sh` 与 parity/data contract 检查。
2. 从 `experiment_matrix.json` 选择唯一实验 ID，固定 task/config/train seed/eval seed。
3. 56 H100 训练；validation 只用于按完整 frame grid 的 prior-action L1 选 `policy_best.ckpt`。
4. 生成并验证 artifact manifest，再传输到 AutoDL 4090 D；传输后再次验证。
5. Q0 将 30 episodes 同时作为方向性评测与资源检查；峰值显存必须低于 21.6GB，且无控制器异常风暴。
6. Q1 及之后使用冻结的 100-seed 列表；同一对比组必须使用同一环境 commit 和 seed 顺序。
7. 汇总成功率、Wilson 区间、阶段成功率、动作越界/跳变、运行时和峰值显存，提交 Git。
8. 只有阶段门槛通过后，才把 `execution_plan.json` 的下一阶段标为 active 并提交 Git。

## 7. 禁止项

- 禁止把 Q0 的 30-episode 结果写成最终“深度有效/无效”结论。
- 禁止把旧 FairACT checkpoint 或 smoke 结果混入 OfficialACTRGBD 主表。
- 禁止用不同数据、相机顺序、评测 seed 或 checkpoint 选择规则比较架构。
- 禁止把 attention map、训练 loss 或 20-rollout smoke 当作任务成功率证据。
- 禁止在 Q0 阶段加入 validity、深度反事实、噪声/修复方法或其他编码器，避免扩大变量。
- 禁止直接修改运行中的服务器副本；所有计划或代码变更先提交 Git，再部署对应 commit。
