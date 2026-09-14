# Official ACT RGB-D 唯一执行计划

本文与 `execution_plan.json` 是本项目后续训练、迁移和评测的唯一执行依据。README 负责解释，
JSON 负责约束脚本。任何实验不得跳阶段；若需要改变架构、任务、seed、训练预算或准入门槛，必须先
修改这两个文件并提交 Git，再启动作业。

## 1. 计算资源分工

| 环境 | 允许工作 | GPU 规则 |
| --- | --- | --- |
| 56 服务器 | 数据生成、开发、短验证；经授权后也可训练 | 默认只使用物理 GPU4-7；物理 GPU0 永久禁用；GPU2-3 只在用户明确给出的时间窗内使用 |
| H100 训练机 | 正式训练、validation、导出 checkpoint | 允许逻辑 `cuda:0`；单架构单卡训练，多个独立任务才并行多卡 |
| AutoDL 评测机 | RoboTwin 单环境、batch 1、固定 seed 在线推理 | 单卡实例的逻辑 `cuda:0` 合法；不得在评测机继续训练或改变权重 |

H100 训练、消费级 GPU 评测是可行的。checkpoint 只保存标准 PyTorch `state_dict`，加载时先映射到
CPU；不得把 CUDA graph、优化器 CUDA tensor 或 H100 专属 FP8 权重作为部署依赖。正式评测固定
FP32 推理；若后续启用 FP16/BF16，必须另建精度一致性实验，不能替换主结果。

## 2. AutoDL 最低配置

服务器现有 RoboTwin + SAPIEN 单进程实测占用约 7.0-7.6 GiB 显存，这只是当前场景的观测值，
不是可租用配置下限。渲染、视频、任务资产和驱动会产生额外峰值，因此采用以下准入标准：

| 评测范围 | 项目准入下限 | 推荐配置 | 说明 |
| --- | --- | --- | --- |
| `ACT0-5/7` | NVIDIA CUDA GPU，16 GiB VRAM；8 vCPU；32 GiB RAM；100 GiB 可用 SSD | 24 GiB VRAM；16 vCPU；64 GiB RAM；200 GiB SSD | 单环境、batch 1；16GB 需先通过 20-rollout 显存预检 |
| `ACT6_LINGBOT_DEPTH` | 24 GiB VRAM；16 vCPU；64 GiB RAM；150 GiB 可用 SSD | 24 GiB 或更高；64 GiB RAM；250 GiB SSD | 同时加载 SAPIEN、ACT、ViT-L/14 与 xFormers，16GB 不进入正式队列 |

8GB 显卡不作为本项目正式评测设备。普通架构可选择 16GB 消费卡；为了让全部 ACT0-7 使用同一台
机器并减少环境分叉，优先租 24GB 卡。租机前还需确认 Linux、NVIDIA 驱动、Vulkan/EGL、CUDA
与容器内 SAPIEN 可用，而不只看显存数字。

## 3. 跨机器产物契约

H100 每次正式训练结束后，输出目录必须至少含有：

- `policy_best.ckpt`、`policy_last.ckpt`；
- `dataset_stats.pkl`、`config.json`、`metrics.jsonl`；
- `artifact_manifest.json`，记录上述文件 SHA256、字节数、源码 commit 和生成时间；
- 项目根目录对应版本的 `UPSTREAM_LOCK.json`、`experiment_matrix.json` 和环境锁文件。

使用 `python scripts/artifact_manifest.py create CHECKPOINT_DIR` 生成清单。传到 AutoDL 后必须先运行
`python scripts/artifact_manifest.py verify CHECKPOINT_DIR`。任一文件缺失、大小不符或 SHA256 不一致，
`run_eval.sh` 会拒绝评测。评测只读 checkpoint，不得原地覆盖训练产物。

若代码目录不是 Git checkout，部署时必须把对应的 40 位 Git commit 写入项目根目录 `.source_commit`；
缺少该标记时 artifact 导出会失败，禁止产生无法追溯源码的正式 checkpoint。

## 4. 严格状态机

1. `P0_BASELINE`：在三个 sentinel 任务训练和 100-seed 评测 `ACT0_RGB`，建立受控官方基线。
2. `P1_PRIMARY_SCREEN`：按固定预算比较 `ACT1/2/4/5`；20-seed 只用于淘汰，晋级者完成 100-seed。
3. `P2_EXTENDED_SCREEN`：仅在 P1 没有明确赢家或容量诊断需要时比较 `ACT3/6/7`。
4. `P3_CLEAN_GT`：一至两个胜出前端使用 clean GT Depth，三训练 seed、每 seed 100 episodes。
5. `P4_ROBUSTNESS`：同一 GT master 派生 clean/noisy/processed，完成 train/deploy 交叉矩阵。
6. `P5_ALL_TASKS`：扩展到六任务，并做 zero/shuffle depth 与跨视角反事实。
7. `P6_VLA_TRANSFER`：只有小模型胜出架构才迁移到 PI0.5、InternVLA 或 LingBot-VLA。

阶段晋级必须提交上一阶段的汇总 JSON、成功率 Wilson 95% 区间、异常动作统计和原始日志索引。
当前阶段为 `P0_BASELINE`。`scripts/workflow_guard.py` 会拒绝当前阶段以外的正式训练和评测。

## 5. 每个实验的固定顺序

1. Git 工作区干净，记录 commit；运行 `scripts/preflight.sh` 与 parity/data contract 检查。
2. 从 `experiment_matrix.json` 选择唯一实验 ID，固定 task/config/train seed/eval seed。
3. H100 训练；validation 只用于按完整 frame grid 的 prior-action L1 选 `policy_best.ckpt`。
4. 生成并验证 artifact manifest，再传输到 AutoDL；传输后再次验证。
5. AutoDL 运行 20-rollout 资源和管线预检；峰值显存应低于总显存的 90%，且无控制器异常风暴。
6. 使用冻结的 100-seed 列表正式评测；同一对比组必须使用同一环境 commit 和 seed 顺序。
7. 汇总成功率、Wilson 区间、阶段成功率、动作越界/跳变、运行时和峰值显存，提交 Git。
8. 只有阶段门槛通过后，才把 `execution_plan.json` 的下一阶段标为 active 并提交 Git。

## 6. 禁止项

- 禁止在 `ACT0` 完整基线前用深度架构结果下结论。
- 禁止把旧 FairACT checkpoint 或 smoke 结果混入 OfficialACTRGBD 主表。
- 禁止用不同数据、相机顺序、评测 seed 或 checkpoint 选择规则比较架构。
- 禁止把 attention map、训练 loss 或 20-rollout smoke 当作任务成功率证据。
- 禁止在没有清单校验和资源预检时直接启动 100-seed 正式评测。
- 禁止直接修改运行中的服务器副本；所有计划或代码变更先提交 Git，再部署对应 commit。
