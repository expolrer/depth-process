# Official ACT RGB-D Benchmark

> **执行约束：** [EXECUTION_PLAN_ZH.md](EXECUTION_PLAN_ZH.md) 与
> [`execution_plan.json`](execution_plan.json) 是后续训练和评测的唯一计划。所有作业必须先通过
> `scripts/workflow_guard.py`；计划变更必须先提交 Git，再部署和运行。

本仓库是 `/ssd/hhw/depth-model` 的第二代策略基准。它只回答一个问题：在保持官方
RoboTwin ACT 动作模型不变时，不同深度表示与视觉融合方式是否能稳定提高机器人任务成功率。

旧的 `FairACT B0-B4` 结果保留为探索性证据，不再作为架构结论。新实验统一以官方 ACT 的
`DETRVAE + action CVAE + Transformer encoder/decoder + 50-step action chunk` 为核心，深度实验
只替换每个相机的视觉骨干。快速阶段让 `ACT0` 与 `ACT1` 使用相同预算并行训练，先获得加深度后的
方向性变化；完整复现与深度有效性验证随后执行。

## 架构矩阵

| 编号 | 输入与视觉前端 | ACT 动作核心 | 用途 |
| --- | --- | --- | --- |
| `ACT0_RGB` | 官方 RGB ResNet18 | 官方原样 | 唯一主基线 |
| `ACT1_EARLY_RGBD` | RGB + metric Z 四通道早期拼接 | 官方原样 | 最低成本深度基线 |
| `ACT2_DUAL_SHARED` | 共享 RGB ResNet18 + 共享单通道 Depth ResNet18 | 官方原样 | 公平双流主方案 |
| `ACT3_DUAL_PER_VIEW` | 三个 RGB ResNet18 + 三个单通道 Depth ResNet18 | 官方原样 | 六分支逐视角方案 |
| `ACT4_XYZMAP` | RGB ResNet18 + 相机坐标 XYZ 点图 | 官方原样 | 显式稠密几何 |
| `ACT5_POINT_TOKENS` | RGB ResNet18 + 无序 XYZ point tokens | 官方原样 | 显式点云几何 |
| `ACT6_LINGBOT_DEPTH` | RGB ResNet18 + 冻结 LingBot-Depth v0.5 tokens | 官方原样 | 预训练深度先验 |
| `ACT7_DEPTH_TRANSFORMER` | RGB ResNet18 + Depth Transformer tokens | 官方原样 | 非 CNN 深度编码器 |

所有后融合版本都以 RGB token 为 Query、深度/几何 token 为 Key/Value，输出与官方 ResNet18
相同的 `B x 512 x H/32 x W/32` 特征图。于是官方 `input_proj`、正弦位置编码、动作 CVAE、
Transformer、损失和动作头均无需改写。门控残差只决定深度增量，不改变 token 数量。

## 不变量

- 分辨率固定为官方 `640 x 480`，不再使用旧实验的 `224 x 224` 方形缩放。
- 相机顺序在受控实验中固定为 `head, left_wrist, right_wrist`，训练与部署一致。
- `chunk_size=50` 且无 temporal aggregation 时，连续执行整段 50 个动作后才重新查询。
- `kl_weight=10`，KL 按官方实现对 32 个 latent 维度求和。
- `hidden_dim=512`、`enc_layers=4`、`dec_layers=7`、正弦位置编码不变。
- AdamW、`lr=1e-5`、`lr_backbone=1e-5`、`weight_decay=1e-4` 不变。
- 相同任务、轨迹、划分、随机种子、训练 epoch、checkpoint 选择规则和评测种子。
- RGB 数据增强、深度噪声和深度修复在架构筛选阶段全部关闭。

## 实验阶段

当前执行 `Q0_OFFICIAL6000_TRAIN_THEN_EVAL`：八种架构均在同一份
`stack_blocks_two/depth_master_clean` 数据上按官方 ACT 预算训练 6000 epochs，batch size 8、seed 0，
每个 epoch 验证并按验证总 loss 选择 `policy_best.ckpt`。所有深度方案只输入 clean metric depth，
不输入 validity mask。训练全部完成后自动使用官方 temporal aggregation 和同一组 100 seeds 评测。
噪声或处理方法，只回答“直接加入 metric depth 是否值得继续”。30 episodes 不作为最终结论。

1. `P0_UPSTREAM_PARITY`：锁定未经修改的 RoboTwin ACT 源码，并引用已有的三项官方复现结果。
   代码级 parity 检查要求 `ACT0` 与官方模型 state-dict key、shape 和初始参数逐元素一致。
   受控训练会修复官方训练/部署腕部顺序不一致，因此历史成功率只作参照，不作逐百分点硬匹配。
2. `P1_ARCH_SMOKE`：在 `stack_blocks_two`、`handover_mic`、
   `pick_diverse_bottles + cluttered_table` 上筛选 `ACT1-ACT7`，每种架构使用同一小预算。
3. `P2_CLEAN_GT`：只保留通过门槛的两种深度架构，用干净 GT Depth 完整训练并做三随机种子评测。
4. `P3_DEPTH_ROBUSTNESS`：固定架构，构造 clean/noisy/processed 的训练输入与部署输入交叉矩阵。
5. `P4_ALL_TASKS`：扩展到六项任务，并加入深度置零、跨样本打乱和跨视角打乱反事实测试。
6. `P5_VLA_TRANSFER`：小模型胜出的深度前端才迁移到 PI0.5、InternVLA 或 LingBot-VLA。

## 关键门槛

- `ACT0` 必须先完成同一套受控训练与 100-seed 评测；若明显低于历史参照，暂停深度实验并审计，
  但不因修复相机顺序后高于历史值而判失败。
- 每个模型至少评测 100 个 episode，报告 Wilson 95% 置信区间和三随机种子结果。
- 深度模型必须同时满足：成功率优于 `ACT0`；真实 Depth 优于 shuffled/zero Depth；提升不只
  出现在 imitation loss。
- checkpoint 用完整、确定性的 validation frame grid 上的 prior-action L1 选择，不能只看
  CVAE posterior loss，也不能只评验证集开头 400 帧。
- 评测必须记录越界动作比例、最大关节跳变、控制器拒绝/TOPP 异常和分阶段成功率。

## 目录

```text
official_act_rgbd/
  variants.py       # 架构注册表与输入契约
  frontends.py      # 可替换深度视觉骨干
  policy.py         # 官方 DETRVAE 包装；动作核心不变
  data.py           # 官方 ACT 对齐的数据加载与完整验证采样
  deploy_policy.py  # 保持官方 50-step chunk 语义的 RoboTwin 接口
train.py            # 受控训练入口
verify_parity.py    # ACT0 与官方构造逐参数一致性检查
experiment_matrix.json
UPSTREAM_LOCK.json
```

服务器目标路径为 `/ssd/hhw/depth-model/repos/official-act-rgbd`。该目录独立于当前正在运行的
`rgbd-act-v2/FairACT`，部署和验证完成前不会替换现有作业。

运行时分工：`ACT0-5/7` 训练使用 `aloha`，`ACT6` 训练使用 `dp3`；普通在线评测使用
`robotwin`，`ACT6` 使用已经验证可同时运行 SAPIEN 与 xFormers 的 `RoboTwin-A3 + a3-eval`。

## 训练与评测机器

正式训练固定在 56 服务器的 H100 上，checkpoint 转移到 AutoDL RTX 4090 D 评测。普通 `ACT0-5/7`
正式评测最低按 16GB 显存准入，`ACT6_LINGBOT_DEPTH` 最低按 24GB 准入；为统一全部架构，推荐
直接使用 24GB。当前 56 服务器 SAPIEN 评测单进程实测约 7.0-7.6 GiB，但 8GB 没有足够的渲染、
视频和任务峰值余量，不进入正式结果队列。完整硬件表、跨机器 artifact 契约、阶段门槛和命令顺序见
`EXECUTION_PLAN_ZH.md`。

平台编号规则不能混用：当前 56 H100 的 GPU4-7 用于 π0.5 四卡训练；GPU1-3 的临时评测已按
北京时间 07:00 停止。旧的“GPU7 续训 ACT5、GPU4-6 评测”接力器已经停用；π0.5 完成后，
GPU4-7 整体切换到 LingBot-VLA 2.0 的 `stack_blocks_two` 四卡 post-training。ACT5 和未完成评测
保留可续状态，等待 LingBot-VLA 2.0 阶段结束后重新排队。AutoDL 4090 D 单卡实例允许使用逻辑
`cuda:0`。分别设置 `OFFICIAL_ACT_PLATFORM=server56_h100|autodl_4090d`。

`scripts/start_pi05_now_gpu4_7.sh` 会立即启动标准 RGB-only π0.5 基线：使用 RoboTwin 官方
`pi05_aloha_full_base`、20,000 steps、global batch 64，在 GPU4-7 上进行四卡 FSDP 全参微调。
`scripts/prepare_and_train_pi05_stack.sh` 先按官方流程将同一批 50 条轨迹转换成不含深度字段的
LeRobot 数据，计算 norm stats，再以每 1,000 steps checkpoint 自动续训。

当前 ACT5 已在 epoch 4708 保存完整 `training_last.pt` 后暂停。LingBot-VLA 2.0 后继任务使用官方
RoboTwin 配置、MoE action expert、FSDP2、三路 RGB、joint/action/prompt，以及 MoGe +
LingBot-Depth + DINO-Video 的在线几何/未来视频蒸馏，训练 30,000 steps 并每 5,000 steps 保存。
需要特别区分：官方 native-depth 路线从 RGB 在线生成 depth teacher targets，不直接读取传感器 GT
Depth；原始三视角 GT Depth 数据完整保留，后续用于显式 depth-input 消融。完整资源、状态机和路径见
[`LINGBOT_VLA2_SUCCESSOR_ZH.md`](LINGBOT_VLA2_SUCCESSOR_ZH.md)。
