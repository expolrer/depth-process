# Robot Interaction Auto Labeler V3

V3 是与 V1、V2 并列的独立自动标注系统。它复用已验证的 GroundingDINO/SAM3 候选、
RGB-D/关节/夹爪交互实例评分、SAM2 双向跟踪和三视角人工修正页面，并在其后增加面向大规模
具身数据治理的事件图、质量校准、三维几何和训练闭环。V1、V2 源码和原有输出格式不被修改。

```text
ROS bag / LeRobot / extracted RGB-D
  -> V2 目标实例识别与双向跟踪
  -> P0 统一事件图、版本、稳定分片和金标集
  -> P1 task/subtask/event 分层边界与三级自然语言
  -> P2 金标可靠度校准、GVL 进度检查、异常质量门控
  -> P3 动态腕部 FK、点云融合、6D 轨迹、跨视角重投影
  -> 人工只复核低分/遮挡/漂移/异常项
  -> P4 纠错蒸馏与 ACT/pi0.5 配对消融
```

## 安装

Grounded-SAM2 路线使用 Python 3.10：

```bash
cd /ssd/hhw/depth-process
uv venv --python 3.10 .venv-auto-labeler-v3
uv pip install --python .venv-auto-labeler-v3/bin/python \
  -e './interaction-labeler-v1[rosbag,lerobot,models]' \
  -e './interaction-auto-labeler-v2[grounded-sam2]' \
  -e './interaction-auto-labeler-v3[models,parquet]'
```

V3 不在运行期间静默下载模型。GroundingDINO、SAM2、SAM3 和本地 VLM 的权重路径都需显式传入。

## 启动互动系统

```bash
.venv-auto-labeler-v3/bin/auto-labeler-v3 run \
  --data /ssd/hhw/zhuomian/lerobot \
  --format lerobot \
  --workspace /ssd/hhw/annotations/zhuomian_v3 \
  --task interaction-auto-labeler-v3/configs/example_task.yaml \
  --concept-model /ssd/hhw/depth-process/models/grounding-dino-base \
  --sam2-checkpoint /ssd/hhw/depth-process/models/sam2/checkpoints/sam2.1_hiera_large.pt \
  --vlm-model /ssd/hhw/models/internvla_a1_5/Qwen3.5-2B \
  --port 8773
```

页面允许在自动处理前画目标框和修正接触帧，也允许处理后逐帧纠正。每次修改除更新当前标注外，
还追加到 `v3/corrections.jsonl`，后续可持续蒸馏领域检测器、跟踪器、边界模型和语言模型。
重新打开工作区使用：

```bash
auto-labeler-v3 serve --workspace /ssd/hhw/annotations/zhuomian_v3 --port 8773
```

## P0 数据底座

统一事件图 schema 位于 `schemas/embodied_event_graph_v1.schema.json`，每个 episode 明确记录：

- 数据集 ID/版本、实体、跨视角 observation、bbox/mask、3D 质心和可选 6D pose。
- task/subtask/event/phase 时间段、父子关系、边界不确定度。
- 交互事件、执行手、目标实例、证据来源、置信度和复核状态。
- task/subtask/event 三级语言、实体引用和数据 lineage。

LeRobot v2.1 到 v3.0 使用官方转换模块，不自行猜测分片布局：

```bash
auto-labeler-v3 convert-lerobot-v3 --repo-id owner/dataset --dry-run
auto-labeler-v3 validate-v3 --root /path/to/lerobot-v3
auto-labeler-v3 version-dataset --root /path/to/lerobot-v3 \
  --dataset-id dataset --version v3.0.0 --output versions/v3.0.0.json
```

千万级元数据采用流式稳定分片，内存占用不随 episode 数线性增长：

```bash
auto-labeler-v3 shard-episodes --input episodes.jsonl --output shards --shards 4096
auto-labeler-v3 queue-enqueue --database jobs.sqlite --jobs jobs.jsonl
auto-labeler-v3 queue-status --database jobs.sqlite
```

SQLite WAL 队列提供幂等 job ID、独占租约、心跳、宕机回收、指数重试和死信。建议每个节点或数据分片
使用独立队列，跨节点调度交给 Ray/Kubernetes/Slurm；不建议让数百台机器争用一个网络文件系统上的
SQLite 文件。

## P1 分层边界和语言

`segment` 接受时间对齐 NPZ。每个数组形状为 `[time, feature]` 或 `[time]`，支持
`scene/visual/object/eef/joint/action/gripper/force/pause`。算法先产生鲁棒多模态变点，
再强制 task > subtask > event 边界嵌套。`boundary_model.py` 提供可由金标边界训练的 PyTorch
Transformer refiner 和分层 BCE loss。

```bash
auto-labeler-v3 segment --graph episode.event_graph.json \
  --modalities episode.npz --output episode.segmented.json \
  --proposal-output boundaries.json
auto-labeler-v3 language --graph episode.segmented.json --output episode.language.json
```

默认语言后端是实体约束模板。也可连接 OpenAI-compatible 本地 VLM；模型返回的实体 ID 不在事件图中时
会直接报错，防止语言幻觉进入训练集。

## P2 可靠度与质量门控

可靠度聚合不会把缺失证据当作 0 或 1，而是同时报告 `raw_score`、`evidence_coverage` 和
`missing_required`。在双人复核加仲裁的金标集上拟合 isotonic calibrator：

```bash
auto-labeler-v3 sample-gold --episodes episodes.jsonl --size 500 \
  --output gold_manifest.json
auto-labeler-v3 fit-calibrator --gold gold_results.jsonl \
  --output reliability_calibrator.json --minimum-precision 0.95
```

GVL 检查会打乱同一 episode 的抽样帧，要求 VLM 恢复进度、阶段和可见实体，再计算 Spearman、
成对时序正确率、进度 MAE 与实体 grounding recall。缺少 GVL 响应的 episode 不会自动通过。
异常检测覆盖时间戳断裂、NaN/Inf、关节/动作/末端突变、深度掉线和“动作仍在变化但视频冻结”。

## P3 三维几何

`kinematics.py` 从 URDF 计算 fixed/revolute/continuous/prismatic FK；动态腕部相机位姿为：

```text
world_from_camera = world_from_base @ base_from_eef(q_t) @ eef_from_camera
```

```bash
auto-labeler-v3 fk-camera-trajectory --urdf robot.urdf --joints joints.jsonl \
  --base-link base_link --eef-link right_wrist \
  --eef-from-camera right_hand_eye.json --output right_camera_poses.npz

auto-labeler-v3 reproject-mask --source-depth depth_mm.png --source-mask target.png \
  --source-intrinsics cam_l.json --target-from-source head_from_cam_l.json \
  --target-intrinsics cam_h.json --target-mask head_target.png \
  --output-mask projected.png --output-report reprojection.json
```

`geometry.py` 支持物理尺度反投影、SE(3) 变换、z-buffer 重投影、IoU/质心误差和体素点云融合。
`pose_tracking.py` 可读取 FoundationPose 轨迹。仅由深度 mask 得到的质心/PCA 结果明确标记
`canonical_rotation=false`，不能冒充有 CAD/模板约束的规范 6D 姿态。

批处理时将 `v3.geometry.mode` 设为 `calibrated` 并填写 JSONL `manifest`。每行格式由
`schemas/calibrated_geometry_manifest_v1.schema.json` 定义，至少包含 episode/frame/camera、
物理深度、目标 mask、相机内参和该帧 `world_from_camera`。腕部位姿应来自上一条 FK 命令，
不能把录制开始时的静态外参重复用于整段运动。V3 会把 3D 质心和 pose proxy 写回 observation，
并输出每对相机的重投影 IoU、质心像素误差和可选融合 PLY。

## P4 训练闭环

```bash
auto-labeler-v3 distill-corrections \
  --corrections WORK/v3/corrections.jsonl --output WORK/v3/distillation
auto-labeler-v3 export-sidecar --graph-root WORK/v3/event_graphs \
  --output DATA/meta/interaction_annotations.jsonl
```

消融默认只生成 A/B，不自动占用 GPU：A 是原始输入基线，B 只增加 ROI/mask 辅助定位损失且推理
不需要检测器。C/D 需要显式 `--include-input-variants`，并要求因果在线 ROI 或部署时检测器。

```bash
auto-labeler-v3 create-ablation --framework act --experiment-id act_roi_ab \
  --dataset /data/lerobot --base-config act.yaml --split-manifest episode_split.json \
  --output-root /runs --steps 5000 --gpus 0,1 \
  --train-command 'python train.py --config {config}' --output ablations/act_roi_ab
auto-labeler-v3 run-ablation --bundle ablations/act_roi_ab
```

最后一条默认 dry-run。只有检查训练进程、显存、数据 split 和命令后才添加 `--execute`。
评估必须按 episode 划分，并同时报告动作 MAE、ROI IoU 和真机任务成功率；注意力图不能替代策略指标。

## 输出目录

| 路径 | 内容 |
| --- | --- |
| `v3/event_graphs/` | 每 episode 的统一事件图 |
| `v3/quality/` | 可靠度、GVL、异常和最终门控 |
| `v3/corrections.jsonl` | 追加式人工纠错审计日志 |
| `v3/distillation/` | COCO、tracker prompts、边界和语言偏好样本 |
| `v3/summary.json` | 页面和批处理共同读取的汇总 |

## 验证

```bash
cd interaction-auto-labeler-v3
pytest -q
ruff check src tests
ruff format --check src tests
```

单元测试覆盖 schema、迁移、版本、金标抽样、分层边界、语言 grounding、可靠度校准、GVL、异常、
动态 FK、重投影、点云融合、6D proxy 限制、队列、分片、纠错蒸馏、训练消融和互动工作区。
