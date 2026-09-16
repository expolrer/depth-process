# LingBot-VLA 2.0 四卡接力计划

## 目标

π0.5 `stack_blocks_two` RGB-only 四卡全参微调完成 20,000 steps 后，在 56 服务器物理 GPU4-7
启动官方 LingBot-VLA 2.0 native-depth post-training。训练使用相同 50 条 RoboTwin 原始 RGB-D
轨迹、任务指令、joint 和 action，预算 30,000 steps，可从 DCP checkpoint 续训。

## 官方 depth 语义

官方 `robotwin.yaml` 只映射 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路 RGB。训练时，
MoGe 从 RGB 估计 metric geometry，LingBot-Depth 生成 current/future depth teacher features，
DINO-Video 生成时序 teacher features，再监督 VLA 的 query tokens。因此这是“原生几何蒸馏 VLA”，
不是“传感器 GT Depth 作为输入”。原始 HDF5 的三视角 GT Depth 必须保留，但本基线不读取它；
后续显式 depth-input 版本必须另设实验名并与本基线做消融。

## 资源与路径

| 内容 | 路径 |
| --- | --- |
| 6 服务器下载根目录 | `/home/zzx23457/models/lingbot-vla-v2` |
| 56 模型根目录 | `/ssd/hhw/depth-model/models/lingbot-vla-v2` |
| 56 官方仓库 | `/ssd/hhw/depth-model/repos/lingbot-vla-v2` |
| 56 独立环境 | `/ssd/hhw/depth-model/envs/lingbotvla2-official` |
| 原始 RGB-D 数据 | `/ssd/hhw/depth-model/datasets/master/stack_blocks_two/depth_master_clean` |
| 官方训练 LeRobot | `/ssd/hhw/depth-model/datasets/lerobot/robotwin/stack_blocks_two_rgb_50` |
| 输出目录 | `/ssd/hhw/depth-model/experiments/OfficialLingBotVLA2/stack_blocks_two/native_depth_official_4gpu_seed0` |
| 状态目录 | `/ssd/hhw/depth-model/scheduler/lingbot_vla2_stack_blocks_two` |

模型资产包括 LingBot-VLA 2.0 6B 主权重、Qwen3-VL-4B tokenizer/base、MoGe、LingBot-Depth、
DINO-Video。所有公网下载只能发生在 6 服务器；传输完成后，56 用 SHA256 验证三个归档。

## 状态机

1. 6 服务器断点下载全部官方资产，并创建 Python 3.12 / Torch 2.8.0 / FlashAttention 2.8.3 环境。
2. 资产、环境和官方代码分别归档，拆分为 768、256、16 个分块。
3. 本地隐藏中继使用 64 个长连接 worker 续传到 56；本地电脑在此阶段必须保持在线。
4. 56 合并分块、校验 SHA256、解包环境，并验证 CUDA、关键 Python 包和源 HDF5 depth 字段。
5. 后继 coordinator 同时等待 `ASSETS_ENV_DATA_READY`、π0.5 final step 20,000、π0.5 进程退出，
   以及 GPU4-7 无计算进程。
6. 在 GPU4 上计算 `stack_blocks_two` 专属 norm stats，然后以 FSDP2 在 GPU4-7 启动训练。

## 训练参数

- 官方 RoboTwin 配置与 55 维统一 action/state 表示。
- 三路 RGB、joint、action、global prompt；使用 future image。
- MoE action expert、Muon optimizer、官方 depth/video auxiliary losses。
- `micro_batch_size=1`、`global_batch_size=4`、`max_steps=30000`。
- 每 5,000 steps 保存，`enable_resume=true`，关闭 WandB 和所有在线下载。

`ACT5_POINT_TOKENS` 保留 epoch 4708 的 `training_last.pt`；未完成 ACT 评测保留原始结果与 claim
状态。它们不与 LingBot-VLA 2.0 并发占用 GPU4-7。
